from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import threading
import time
import pulp


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # セッション管理のために必要

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 必要であればアップロードフォルダを作成
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ファイルクリーンアップ機能
def schedule_cleanup(file_path, delay=180):
    def cleanup():
        time.sleep(delay)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed file: {file_path}")

    threading.Thread(target=cleanup, daemon=True).start()



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # アップロードされたCSVファイルと選択された曜日を取得
        csv_file = request.files.get('csv_file')
        selected_day = request.form.get('day')

        if csv_file and csv_file.filename.endswith('.csv'):
            # ファイルを保存
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
            csv_file.save(file_path)

            # ファイルクリーンアップをスケジュール
            schedule_cleanup(file_path)

            # セッションに保存して次の画面で使用
            session['csv_file_path'] = file_path
            session['selected_day'] = selected_day

            return redirect(url_for('members'))

    return render_template('day_select.html')

@app.route('/members', methods=['GET'])
def members():
    # セッションからCSVファイルパスと選択した曜日を取得
    file_path = session.get('csv_file_path')
    day = session.get('selected_day')

    if not file_path or not day:
        return redirect(url_for('home'))  # 必要な情報がなければホームに戻る

    # CSVファイルを読み込む
    base = pd.read_csv(file_path)
    preferences = base[day].tolist()
    members = base['name'].tolist()

    return render_template('members.html', day=day, members=zip(members, preferences))



# 最適化を実行して結果を表示
@app.route('/assign', methods=['POST'])
def optimize():
    # セッションからCSVファイルパスを取得
    file_path = session.get('csv_file_path')

    if not file_path:
        return redirect(url_for('home'))  # 必要な情報がなければホームに戻る

    preferences = request.form.to_dict()
    day = preferences.pop('day', None)

    # アップロードされたCSVを読み込む
    base = pd.read_csv(file_path)

    # 最適化を実行
    results, message, failure_data = run_optimization(preferences, base)

    return render_template('results.html', results=results, message=message, failure_data=failure_data)

def run_optimization(preferences, base):
    # 入力に基づき 'today' の列を更新
    for name, preference in preferences.items():
        if name in base['name'].values:
            base.loc[base['name'] == name, 'today'] = preference

    # 線形最適化問題のセットアップ
    prob = pulp.LpProblem('carassignment', pulp.LpMinimize)

    # ドライバーと参加者のリスト
    drivers = base[(base['car'] > 0) & (base['today'] != '不参加')]['name'].tolist()
    participants = base[base['today'] != '不参加']['name'].tolist()

    # 地区データと各車の定員
    area = {row['name']: row['area'] for _, row in base.iterrows()}
    car_capacity = {row['name']: int(row['car']) for _, row in base.iterrows() if row['car'] > 0 and row['today'] != '不参加'}

    # バイナリ変数の定義
    x = pulp.LpVariable.dicts('x', [(m, c) for m in participants for c in drivers], cat='Binary')
    y = pulp.LpVariable.dicts('y', [(c, g) for c in drivers for g in ['先発', '後発', '直帰']], cat='Binary')
    use_car = pulp.LpVariable.dicts('use_car', drivers, cat='Binary')
    optimal_occupancy_bonus = pulp.LpVariable.dicts('optimal_occupancy_bonus', drivers, cat='Binary')
    first_departure_bonus = pulp.LpVariable.dicts("first_departure_bonus", drivers, cat="Binary")

    # 制約1: 各部員が1台の車に割り当てられる
    for m in participants:
        prob += pulp.lpSum(x[(m, c)] for c in drivers) == 1

    # 制約2: 各車が1つのグループのみを選択
    for c in drivers:
        prob += pulp.lpSum(y[(c, g)] for g in ['先発', '後発', '直帰']) == use_car[c]

    # 制約3: 希望するグループへの割り当て
    for m in participants:
        desired_group = base.loc[base['name'] == m, 'today'].values[0]
        for c in drivers:
            if desired_group == 'どちらでも':
                prob += x[(m, c)] <= pulp.lpSum(y[(c, g)] for g in ['先発', '後発', '直帰'])
            else:
                prob += x[(m, c)] <= y[(c, desired_group)]

    # 制約4: ドライバーは自分の車を運転する
    for c in drivers:
        if c in participants:
            prob += x[(c, c)] >= use_car[c]

    # 制約5: 各車の乗車人数が定員を超えない
    for c in drivers:
        prob += pulp.lpSum(x[(m, c)] for m in participants) <= car_capacity[c]

    # 制約6: 各車の乗車人数が3～4人の場合にボーナスを付与
    for c in drivers:
        prob += pulp.lpSum(x[(m, c)] for m in participants) >= 3 * optimal_occupancy_bonus[c]
        prob += pulp.lpSum(x[(m, c)] for m in participants) <= 4 * optimal_occupancy_bonus[c]

    # 目的関数
    prob += (
        pulp.lpSum(use_car[c] for c in drivers)
        + pulp.lpSum(x[(m, c)] * 0.01 for c in drivers for m in participants if area[c] != area[m])
        - pulp.lpSum(optimal_occupancy_bonus[c] * 5 for c in drivers)
        + pulp.lpSum(first_departure_bonus[c] * 5 for c in drivers)
    )

    # 最適化実行
    prob.solve(pulp.PULP_CBC_CMD(msg=True))



        # 結果の整形
    if pulp.LpStatus[prob.status] == 'Optimal':
        results = []
        grouped_results = {'直帰': [], '先発': [], '後発': []}

        for group in ['直帰', '先発', '後発']:
            for c in drivers:
                if pulp.value(y[(c, group)]) > 0.5:
                    assigned_members = [m for m in participants if pulp.value(x[(m, c)]) > 0.5]
                    grouped_results[group].append((c, assigned_members))  # グループごとにドライバーと割り当てメンバーを追加

        # LINE投稿用フォーマットの生成
        message = "【本日の車割】\n四乗カーはミニバッグでお願いします。\n\n"
        for group, entries in grouped_results.items():
            if entries:  # 該当グループにエントリーが存在する場合のみ出力
                message += f"（{group}）\n"
                for c, members in entries:
                    # メンバーリストの先頭にドライバーを明示的に追加
                    formatted_members = [c] + [m for m in members if m != c]  # ドライバーを先頭にし、重複を防ぐ
                    message += "、".join(formatted_members) + "\n"
                message += "\n"


        # 結果をリスト形式で整形
        results = [(c, group, members) for group, entries in grouped_results.items() for c, members in entries]


        # 全参加者情報の整理（名前、希望グループ、定員）
        participant_details = sorted(
            [
                (row['name'], row['today'], car_capacity.get(row['name'], ''))  # ドライバーには定員を表示
                for _, row in base.iterrows()
                if row['today'] != '不参加'
            ],
            key=lambda x: ['直帰', '先発', '後発', 'どちらでも'].index(x[1])
        )

        participant_table = [
            {'name': p[0], 'group': p[1], 'capacity': p[2]} for p in participant_details
        ]

        return results, message.strip(), participant_table











    else:
        # 失敗時の詳細情報を作成
        total_participants = len(participants)
        total_capacity = sum(car_capacity.values())  # 不参加の車を含まない

        # 希望ごとに参加者を並べ替え
        sorted_participants = sorted(
            [
                (row['name'], row['today'], car_capacity.get(row['name'], ''))  # ドライバーには定員を表示
                for _, row in base.iterrows()
                if row['today'] != '不参加'
            ],
            key=lambda x: ['直帰', '先発', '後発', 'どちらでも'].index(x[1])
        )

        failure_data = {
            'total_participants': total_participants,
            'total_capacity': total_capacity,
            'participants': [{'name': p[0], 'group': p[1], 'capacity': p[2]} for p in sorted_participants],
        }

        return [], "Optimization failed!", failure_data


if __name__ == '__main__':
    # Render の環境変数 PORT を取得し、デフォルト値は 5000 に設定
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


