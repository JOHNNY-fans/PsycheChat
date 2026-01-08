import os
import openai
import json
import time
from tqdm import tqdm
import regex as re
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompt import prompt_card

client = openai.OpenAI(
    api_key='your_api_key',
    base_url='your_base_url'
)

def call_llm(messages, model_name, temperature=0.7, sleep_time=1.0):
    while True:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            time.sleep(sleep_time)
            continue

def get_json(text):
    pattern = r'\{(?:[^{}]|(?R))*\}'
    try:
        match = re.search(pattern, text)
        if not match:
            return {}
        json_str = match.group(0)
        data = json.loads(json_str)
        return data
    except Exception:
        return {}
    
def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = []
    return data

def write_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_history_text(history):
    history_lsit = []
    for h in history:
        if h['role'] == 'user':
            history_lsit.append('来访者：' + h['content'])
        elif h['role'] == 'assistant':
            history_lsit.append('咨询师：' + h['content'])
    return '\n'.join(history_lsit)

emotion_groups = [
    ["平静", "快乐", "狂喜"],
    ["接受", "信任", "崇敬"],
    ["担心", "恐惧", "惊悚"],
    ["不解", "惊讶", "惊诧"],
    ["伤感", "悲伤", "悲痛"],
    ["厌倦", "厌恶", "憎恨"],
    ["烦躁", "生气", "暴怒"],
    ["关心", "期待", "警惕"]
]
emotion_to_group = {emo: idx for idx, group in enumerate(emotion_groups) for emo in group}

def validate_emotion(emotion_list):
    if not isinstance(emotion_list, list):
        return False
    if not (1 <= len(emotion_list) <= 3):
        return False
    used_groups = set()
    for emo in emotion_list:
        if not isinstance(emo, str):
            return False
        if emo not in emotion_to_group:
            return False
        group = emotion_to_group[emo]
        if group in used_groups:
            return False
        used_groups.add(group)
    return True

def validate_card(role_card):
    required_fields = ["性别", "年龄", "职业", "人格", "性格", "语言风格", "爱好",
                       "问题", "目标", "内心独白", "初始情绪", "事件触发情绪"]
    for field in required_fields:
        if field not in role_card:
            return False
    goals = role_card["目标"]
    if not isinstance(goals, list) or len(goals) != 2:
        return False
    for g in goals:
        if "gid" not in g or "goal" not in g:
            return False
    ok = validate_emotion(role_card["初始情绪"])
    if not ok:
        return False
    events = role_card["事件触发情绪"]
    if not isinstance(events, list):
        return False
    for e in events:
        if "eid" not in e or "event" not in e or "emotion" not in e:
            return False
        ok = validate_emotion(e["emotion"])
        if not ok:
            return False
    return True

def generate_role_card(dialogue_data, model_name):
    idx = dialogue_data['id']
    topic = dialogue_data['normalizedTag']
    dialogue = get_history_text(dialogue_data['messages'][1:])
    user_card = prompt_card.replace('{{original_info}}', dialogue)
    messages_card = [{'role': 'user', 'content': user_card}]
    validate_flag = False
    while not validate_flag:
        role_card = call_llm(messages_card, model_name, temperature=0.7)
        role_card = get_json(role_card)
        validate_flag = validate_card(role_card)
    return {
        'id': idx,
        'topic': topic,
        'role_card': role_card
    }

if __name__ == '__main__':
    input_file = '/path/to/your_seed_data.json'
    output_file = './datasets_example/role_card/role_cards.json'

    model_name = 'gemini-2.5-flash'
    max_workers = 16

    psydt_corpus = read_json(input_file)
    result_json = read_json(output_file)
    done_result = {item["id"]: {"topic": item.get("topic"), "role_card": item["role_card"]} for item in result_json}

    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}

    for dialogue_data in psydt_corpus:
        idx = dialogue_data["id"]
        if idx in done_result:
            continue

        future = pool.submit(generate_role_card, dialogue_data, model_name)
        futures[future] = idx

    for future in tqdm(as_completed(futures), total=len(futures)):
        idx = futures[future]
        result = future.result()

        topic = result.get("topic")
        role_card = result.get("role_card")

        done_result[idx] = {
            "topic": topic,
            "role_card": role_card
        }

        result_json = [
            {"id": k, "topic": v["topic"], "role_card": v["role_card"]}
            for k, v in sorted(done_result.items(), key=lambda x: x[0])
        ]
        write_json(result_json, output_file)
        if len(result_json) % 200 == 0:
            base_dir = os.path.dirname(output_file)
            filename = os.path.splitext(os.path.basename(output_file))[0]
            backup_name = f"{filename}_{len(result_json)}.json"
            backup_file = os.path.join(base_dir, backup_name)
            write_json(result_json, backup_file)

    pool.shutdown(wait=True)
