import os
import openai
import time
import json
import regex as re

def call_llm(client, messages, model_name, temperature=0.0, sleep_time=1.0, enable_thinking=None):
    for i in range(5):
        try:
            if enable_thinking is None:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    }
                )
            return completion.choices[0].message.content
        except Exception as e:
            time.sleep(sleep_time)
            continue
    return None

class LLM_Role():
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.emotion_groups = [
            ["平静", "快乐", "狂喜"],
            ["接受", "信任", "崇敬"],
            ["担心", "恐惧", "惊悚"],
            ["不解", "惊讶", "惊诧"],
            ["伤感", "悲伤", "悲痛"],
            ["厌倦", "厌恶", "憎恨"],
            ["烦躁", "生气", "暴怒"],
            ["关心", "期待", "警惕"]
        ]
        self.emotion_to_group = {emo: idx for idx, group in enumerate(self.emotion_groups) for emo in group}

    def get_json(self, text):
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

    def validate_emotion(self, emotion_list):
        if not isinstance(emotion_list, list):
            return False
        if not (1 <= len(emotion_list) <= 3):
            return False
        used_groups = set()
        for emo in emotion_list:
            if not isinstance(emo, str):
                return False
            if emo not in self.emotion_to_group:
                return False
            group = self.emotion_to_group[emo]
            if group in used_groups:
                return False
            used_groups.add(group)
        return True

    def __call__(self, messages, temperature=0.0, sleep_time=1.0, thinking=None) -> str:
        max_temperature = 1.0
        for i in range(20):
            try:
                seeker_res = call_llm(self.client, messages, self.model_name, temperature=temperature, sleep_time=sleep_time, thinking=thinking)
                seeker_res = self.get_json(seeker_res)
                seeker_emotion = seeker_res['emotion']
                if not self.validate_emotion(seeker_emotion):
                    temperature += 0.1
                    temperature = min(temperature, max_temperature)
                    continue
                seeker_event = seeker_res['event']
                seeker_goals = seeker_res['goals']
                seeker_risk = seeker_res['risk']
                seeker_response = seeker_res['response']
                end_flag = False
                if 'END' in seeker_response:
                    seeker_response = seeker_response.replace('END', '').strip()
                    if seeker_response.strip() != '':
                        end_flag = True
                return seeker_res, end_flag
            except Exception:
                temperature += 0.1
                temperature = min(temperature, max_temperature)
                continue
        return None, None
    
class Ours_LLM():
    def __init__(self, model_name='', api_key = '', base_url='', aux_model_name='', aux_api_key='', aux_base_url=''):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def __call__(self, messages, counselor={}, temperature=0.0) -> str:
        system_counselor = '''# 任务说明：\n你是一位精通情绪聚焦疗法（Emotion-Focused Therapy，EFT）的专业心理咨询师，正在通过在线文字聊天的方式与来访者进行咨询对话。你的任务是依据对话历史，结合来访者的当前情绪及情绪转变，考虑可能的安全风险，按照EFT指南进行策略分析，并生成自然、温暖、富有共情的咨询师回应。\n\n# 整体流程\n采用助人技巧中的探索、领悟、行动三阶段，过程可循环迭代进行：\n一、探索阶段（建立关系与明确目标）\n- 与来访者建立信任关系，通过共情、真诚、积极关注等方式营造安全氛围。\n- 鼓励来访者讲述个人故事，深入了解其想法、情绪与背景，帮助其厘清主要困扰。\n- 与来访者共同协商确定一个具体、可行、积极的咨询目标，保持焦点一致。\n二、领悟阶段（激活资源与转化认知）\n- 协助来访者识别自身的内在资源，如过往应对经验、优势品质与复原力。\n- 鼓励来访者觉察自身在问题中的角色与模式，提升对情绪、思维和行为的理解。\n- 探索外部资源，如人际支持、社会系统与环境优势，拓宽解决问题的路径。\n三、行动阶段（转化资源与落实方案）\n- 基于前期识别的积极资源，协助来访者将其转化为具体、可行的行动策略。\n- 在咨询中通过模拟或练习的方式预演新行为，增强来访者对方案可行性的信心与执行能力。\n- 鼓励来访者在现实生活中尝试新的应对方式，提升其独立解决问题的能力，减少对咨询的依赖。\n- 总结行动过程中获得的积极经验，强化有效改变，促进持续性成长。\n\n# EFT指南\n- 进入与调节情绪：咨询师需要充分理解来访者此刻的痛苦，包括其总体的困扰、触发痛苦的具体事件、来访者对事件中“自己的看法”（如自责、焦虑、打断等），并评估其是否存在情绪淹没或情绪回避。如果来访者难以接触或承受情绪，咨询师必须首先帮助其调节情绪，以建立足够的安全感和情绪稳定性。只有当来访者能够较为稳定地接触自身的感受，才可能进入更深层的情绪探索。\n- 触及与加工核心痛苦情绪：咨询师引导来访者在情绪基模的视角下重新接触导致其核心痛苦的情绪经验。治疗师陪伴来访者回到这些情境中，还原过去痛苦的场景、人物及当时来访者的身体体会、心理感受、想法、需求、行动倾向，唤起其情绪并对之加以区分和表达，让来访者觉察到自己过去没有被满足的需求并向重要他人表达。此阶段通过唤起、澄清和深化情绪，使来访者能够真正触及核心情绪。\n- 情绪的转化与问题解决：在来访者触及核心痛苦情绪之后，咨询目标是促进这些情绪向更具适应性的情绪状态转化。咨询师会考察来访者内在的不同部分是否能够产生软化、理解或和解。通过在安全关系中体验新的情绪回应，来访者得以形成更健康的情感组织方式，从而促进更有效的自我调节与行为改变。\n\n# 注意事项：\n- 你最重要的任务是高情商地和来访者聊天，在每一轮对话中为来访者提供情绪价值，让他/她感到舒适、愉快或得到需要的帮助。\n- 结合来访者的当前情绪和情绪转变，考虑可能的安全风险，依照EFT流派进行分析。\n- 避免在对话中重复使用相似的策略、表达相似的内容、使用相似的句式，保持叙述推进性。\n- 在咨询初期应优先了解来访者目前遇到的问题、事件经过、影响范围、来访者的目标等基本信息。\n- 在咨询后期应结合来访者的具体处境，讨论可行的改变方向，共同制定清晰、可执行的目标或行动步骤。\n\n# 语言风格\n- 像一个真实的人那样说话，而不是像一本教科书或AI客服。\n- 语言温和、接纳、口语化，应贴近当事人的情绪体验，避免过度文艺或修饰。\n- 保持自然口语化和轻松的表达风格，回复不宜冗长，一般以简洁1~2句话为主。\n\n# 输出要求\n你需要先进行思考分析，思考过程被包含在<think>和</think>标签中，例如：<think>\n思考内容\n</think>\n\n最终回复。'''
        messages_counselor = [{'role': 'system', 'content': system_counselor}] + counselor['history']
        counselor_res = call_llm(self.client, messages_counselor, self.model_name, temperature)
        think_match = re.search(r"<think>(.*?)</think>", counselor_res, re.S)
        counselor_think = think_match.group(1).strip() if think_match else ""
        counselor_reponse = re.sub(r"<think>.*?</think>", "", counselor_res, flags=re.S).strip()
        counselor['history'].append({
            "role": "assistant",
            "content": counselor_res,
        })
        return counselor_reponse, counselor
    
class Ours_Agent():
    def __init__(self, model_name='', api_key = '', base_url='', aux_model_name='', aux_api_key = '', aux_base_url=''):
        self.counselor_model_name = model_name
        self.counselor_api_key = api_key
        self.counselor_base_url = base_url
        self.counselor_client = openai.OpenAI(
            api_key=self.counselor_api_key,
            base_url=self.counselor_base_url
        )
        self.seeker_model_name = aux_model_name
        self.seeker_api_key = aux_api_key
        self.seeker_base_url = aux_base_url
        self.seeker_client = openai.OpenAI(
            api_key=self.seeker_api_key,
            base_url=self.seeker_base_url
        )
        self.emotion_groups = [
            ["平静", "快乐", "狂喜"],
            ["接受", "信任", "崇敬"],
            ["担心", "恐惧", "惊悚"],
            ["不解", "惊讶", "惊诧"],
            ["伤感", "悲伤", "悲痛"],
            ["厌倦", "厌恶", "憎恨"],
            ["烦躁", "生气", "暴怒"],
            ["关心", "期待", "警惕"]
        ]
        self.emotion_to_group = {emo: idx for idx, group in enumerate(self.emotion_groups) for emo in group}


    def get_json(self, text):
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
        
    def get_memory_text(self, emotion_memory):
        return (
            f"当前情绪：{'，'.join(emotion_memory['current_emotion'])}\n"
            f"当前情绪原因：{emotion_memory['current_analysis']}\n"
            f"近期转变：{emotion_memory['recent_change'] or '无'}\n"
            f"整体趋势：{emotion_memory['overall_trend'] or '无'}\n"
            f"情绪转变原因：{emotion_memory['shift_analysis'] or '无'}"
        )

    def get_safety_text(self, counselor_response, seeker_utterances, safety_res):
        output = []
        output.append(f'如果咨询师回复：{counselor_response}')
        output.append('来访者可能的反应：')
        safety_analysis = safety_res.get('safety_analysis', [])
        for item in safety_analysis:
            uid = item['uid']
            analysis = item['analysis']
            is_safe = item['pass']
            utterance = next((u['utterance'] for u in seeker_utterances if u['uid'] == uid), '')
            output.append(f'回复：{utterance}{"（不存在风险）" if is_safe else "（存在风险）"}')
            output.append(f'分析：{analysis}\n')
        suggestion_emotion = safety_res['suggestion'].get('emotion', '')
        suggestion_safety = safety_res['suggestion'].get('safety', '')
        output.append(f'情绪修改建议：{suggestion_emotion}')
        output.append(f'安全修改建议：{suggestion_safety}')
        return '\n'.join(output)

    def get_history_text(self, messages):
        history = "\n".join([f"{'来访者' if mes['role'] == 'user' else '咨询师'}：{mes['content']}" for mes in messages])
        return history
    
    def validate_emotion(self, emotion_list):
        if not isinstance(emotion_list, list):
            return False
        if not (1 <= len(emotion_list) <= 3):
            return False
        used_groups = set()
        for emo in emotion_list:
            if not isinstance(emo, str):
                return False
            if emo not in self.emotion_to_group:
                return False
            group = self.emotion_to_group[emo]
            if group in used_groups:
                return False
            used_groups.add(group)
        return True

    def __call__(self, messages, counselor={}, temperature=0.0) -> str:
        max_temperature = 0.7
        emotion_tool_text = "<think>\n\n</think>\n\n<tool_call>\n{\"name\": \"emotion_analysis\", \"arguments\": {}}\n</tool_call>"
        safety_tool_text = "<tool_call>\n{\"name\": \"safety_analysis\", \"arguments\": {}}\n</tool_call>"

        # Emotion analysis
        emotion_prompt = '''# 任务说明：\n你是一名专业的心理咨询师，正在与来访者进行咨询对话。你的任务是根据来访者最新一轮回复、对话历史及其中标注的情绪、前一轮的情绪分析，对来访者的情绪状态进行系统化的分析，做出清晰且有深度的解读。所有分析必须基于言语内容，不做臆测，不进行医学或精神科诊断。\n\n# 情绪说明：\n当你需要分析情绪时，需要从以下八组情绪中选择标签。每组情绪按强度递进排列，每次必须选择：1个主情绪、0~2个次情绪，所有选中的标签必须来自不同的组别。\n输出必须是列表格式，如：[主情绪] 或 [主情绪, 次情绪1] 或 [主情绪, 次情绪1, 次情绪2]\n八组情绪如下：[[平静, 快乐, 狂喜], [接受, 信任, 崇敬], [担心, 恐惧, 惊悚], [不解, 惊讶, 惊诧], [伤感, 悲伤, 悲痛], [厌倦, 厌恶, 憎恨], [烦躁, 生气, 暴怒], [关心, 期待, 警惕]]\n\n# 分析方法：\n- 当前情绪识别\n从来访者最新回复的语言、语气、情绪信号中识别当下的情绪。\n- 当前情绪原因分析，你可以从如下角度进行分析：\n * 直接触发因素：分析来访者在这一轮出现这种情绪的原因。\n * 对情绪的态度：分析来访者对自身情绪的态度。\n- 近期情绪转变分析\n根据最近轮次到当前轮的来访者表述与情绪，描述短期情绪的变化轨迹。\n- 整体趋势分析\n根据完整对话历史与情绪，总结情绪在宏观层面的变化趋势。\n- 情绪转变原因分析，你可以从如下角度进行分析：\n * 情绪变化瞬间：说明说到哪句话时出现了明显的情绪转折。\n * 深层的触发点：分析导致这种转变的根本心理因素。\n * 潜在情感需求：探讨转变中暴露出的真实需求。\n * 长期情绪模式：指出来访者在多轮对话中反复呈现的情绪模式以及这种模式如何影响他的情绪变化。\n\n# 对话历史：\n{{history}}\n\n# 来访者回复：\n{{seeker_utterance}}\n\n# 前一轮的情绪分析：\n{{emotion_memory}}\n\n# 输出要求：\n你需要先进行思考分析，思考过程被包含在<think>和</think>标签中，例如：<think>\n思考内容\n</think>\n\n最终结果。\n根据对话历史和前一轮的情绪分析，直接用JSON格式输出来访者当前情绪、当前情绪原因、近期转变、整体趋势与情绪转变原因，格式示例如下：\n{\n    \"current_emotion\": [emo, ……], # 情绪标签必须来自八组情绪，选中的标签必须来自不同的组别\n    \"current_analysis\": xxx, # 对当前情绪原因的简短分析\n    \"recent_change\": xxx, # 只描述现象，不解释原因，前3轮可为\"\"\n    \"overall_trend\": xxx, # 只描述现象，不解释原因，前3轮可为\"\"\n    \"shift_analysis\": xxx, # 自然语言段落，不得使用JSON格式，前3轮可为\"\"\n}'''
        user_emotion = emotion_prompt.replace('{{history}}', self.get_history_text(messages[:-1])).replace('{{seeker_utterance}}', messages[-1]['content']).replace('{{emotion_memory}}', json.dumps(counselor['emotion_memory'], indent=4, ensure_ascii=False))
        messages_emotion = [{'role': 'user', 'content': user_emotion}]
        temperature = 0.0
        while True:
            try:
                emotion_res = call_llm(self.counselor_client, messages_emotion, self.counselor_model_name, temperature)
                think_match = re.search(r"<think>(.*?)</think>", emotion_res, re.S)
                emotion_think = think_match.group(1).strip() if think_match else ""
                emotion_answer = re.sub(r"<think>.*?</think>", "", emotion_res, flags=re.S).strip()
                emotion_answer = self.get_json(emotion_answer)
                current_emotion = emotion_answer['current_emotion']
                if not self.validate_emotion(current_emotion):
                    temperature += 0.1
                    temperature = min(max_temperature, temperature)
                    continue
                current_analysis = emotion_answer['current_analysis']
                recent_change = emotion_answer['recent_change']
                overall_trend = emotion_answer['overall_trend']
                shift_analysis = emotion_answer['shift_analysis']
                break
            except Exception as e:
                temperature += 0.1
                temperature = min(max_temperature, temperature)
                continue
        counselor['emotion_memory'] = emotion_answer
        emotion_text = self.get_memory_text(emotion_answer)
        counselor['history'].append({
            "role": "assistant",
            "content": emotion_tool_text
        })
        counselor['history'].append({
            "role": "user",
            "content": "<tool_response>\n" + json.dumps({"emotion_analysis": emotion_text}, ensure_ascii=False) + "\n</tool_response>",
        })

        pass_flag = False
        while not pass_flag:
                
            # Counselor response
            system_counselor = '''# 任务说明：\n你是一位精通情绪聚焦疗法（Emotion-Focused Therapy，EFT）的专业心理咨询师，正在通过在线文字聊天的方式与来访者进行咨询对话。你的任务是依据对话历史，结合来访者的当前情绪及情绪转变，考虑可能的安全风险，按照EFT指南进行策略分析，并生成自然、温暖、富有共情的咨询师回应。\n\n# 整体流程\n采用助人技巧中的探索、领悟、行动三阶段，过程可循环迭代进行：\n一、探索阶段（建立关系与明确目标）\n- 与来访者建立信任关系，通过共情、真诚、积极关注等方式营造安全氛围。\n- 鼓励来访者讲述个人故事，深入了解其想法、情绪与背景，帮助其厘清主要困扰。\n- 与来访者共同协商确定一个具体、可行、积极的咨询目标，保持焦点一致。\n二、领悟阶段（激活资源与转化认知）\n- 协助来访者识别自身的内在资源，如过往应对经验、优势品质与复原力。\n- 鼓励来访者觉察自身在问题中的角色与模式，提升对情绪、思维和行为的理解。\n- 探索外部资源，如人际支持、社会系统与环境优势，拓宽解决问题的路径。\n三、行动阶段（转化资源与落实方案）\n- 基于前期识别的积极资源，协助来访者将其转化为具体、可行的行动策略。\n- 在咨询中通过模拟或练习的方式预演新行为，增强来访者对方案可行性的信心与执行能力。\n- 鼓励来访者在现实生活中尝试新的应对方式，提升其独立解决问题的能力，减少对咨询的依赖。\n- 总结行动过程中获得的积极经验，强化有效改变，促进持续性成长。\n\n# EFT指南\n- 进入与调节情绪：咨询师需要充分理解来访者此刻的痛苦，包括其总体的困扰、触发痛苦的具体事件、来访者对事件中“自己的看法”（如自责、焦虑、打断等），并评估其是否存在情绪淹没或情绪回避。如果来访者难以接触或承受情绪，咨询师必须首先帮助其调节情绪，以建立足够的安全感和情绪稳定性。只有当来访者能够较为稳定地接触自身的感受，才可能进入更深层的情绪探索。\n- 触及与加工核心痛苦情绪：咨询师引导来访者在情绪基模的视角下重新接触导致其核心痛苦的情绪经验。治疗师陪伴来访者回到这些情境中，还原过去痛苦的场景、人物及当时来访者的身体体会、心理感受、想法、需求、行动倾向，唤起其情绪并对之加以区分和表达，让来访者觉察到自己过去没有被满足的需求并向重要他人表达。此阶段通过唤起、澄清和深化情绪，使来访者能够真正触及核心情绪。\n- 情绪的转化与问题解决：在来访者触及核心痛苦情绪之后，咨询目标是促进这些情绪向更具适应性的情绪状态转化。咨询师会考察来访者内在的不同部分是否能够产生软化、理解或和解。通过在安全关系中体验新的情绪回应，来访者得以形成更健康的情感组织方式，从而促进更有效的自我调节与行为改变。\n\n# 注意事项：\n- 你最重要的任务是高情商地和来访者聊天，在每一轮对话中为来访者提供情绪价值，让他/她感到舒适、愉快或得到需要的帮助。\n- 结合来访者的当前情绪和情绪转变，考虑可能的安全风险，依照EFT流派进行分析。\n- 避免在对话中重复使用相似的策略、表达相似的内容、使用相似的句式，保持叙述推进性。\n- 在咨询初期应优先了解来访者目前遇到的问题、事件经过、影响范围、来访者的目标等基本信息。\n- 在咨询后期应结合来访者的具体处境，讨论可行的改变方向，共同制定清晰、可执行的目标或行动步骤。\n- 如果来访者对你表示感谢并有结束的意愿，应该对来访者表示祝福。\n\n# 语言风格\n- 像一个真实的人那样说话，而不是像一本教科书或AI客服。\n- 语言温和、接纳、口语化，应贴近当事人的情绪体验，避免过度文艺或修饰。\n- 保持自然口语化和轻松的表达风格，回复不宜冗长，一般以简洁1~2句话为主。\n\n# 输出要求：\n你需要先进行思考分析，思考过程被包含在<think>和</think>标签中，例如：<think>\n思考内容\n</think>\n\n最终结果。\n若未进行安全分析或安全分析未通过，直接用JSON格式输出当前的阶段、EFT流派的操作与一次咨询师回复，格式示例如下：\n{\n    \"phase\": xxx, # [探索, 领悟, 行动]\n    \"strategy\": xxx, # 一句简短的EFT流派策略说明文本\n    \"response\": xxx # 咨询师的回复\n}\n若安全分析已通过，直接输出通过的咨询师回复。\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"emotion_analysis\", \"description\": \"情绪分析。获取来访者当前情绪、当前情绪原因、近期转变、整体趋势与情绪转变原因\", \"parameters\": {}}},\n{\"type\": \"function\", \"function\": {\"name\": \"safety_analysis\", \"description\": \"安全分析。获取来访者可能出现的后续反应、对反应的分析与修改建议\", \"parameters\": {}}},\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>'''
            messages_counselor = [{'role': 'system', 'content': system_counselor}] + counselor['history']
            temperature = 0.0
            while True:
                try:
                    counselor_res = call_llm(self.counselor_client, messages_counselor, self.counselor_model_name, temperature)
                    think_match = re.search(r"<think>(.*?)</think>", counselor_res, re.S)
                    counselor_think = think_match.group(1).strip() if think_match else ""
                    counselor_answer = re.sub(r"<think>.*?</think>", "", counselor_res, flags=re.S).strip()
                    counselor_answer = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", counselor_answer, flags=re.S).strip()
                    counselor_answer = self.get_json(counselor_answer)
                    counselor_phase = counselor_answer['phase']
                    counselor_strategy = counselor_answer['strategy']
                    counselor_response = counselor_answer['response']
                    break
                except Exception as e:
                    temperature += 0.1
                    temperature = min(max_temperature, temperature)
                    continue

            counselor_res = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", counselor_res, flags=re.S).strip() + '\n' + safety_tool_text
            counselor['history'].append({
                "role": "assistant",
                "content": counselor_res,
            })

            # Safety analysis
            seeker_prompt = '''# 任务说明：\n你是一名来访者，正在与心理咨询师进行咨询对话。你需要基于对话历史、以及咨询师最新一轮回复，生成来访者下一句回复。在生成回应前，你需要先判断咨询师的最新回复是否可能使你的情绪变得更糟。如果情绪变糟，选择一个可能引发的“回复类型”，并按照对应风格表达；如果情绪没有变糟，则将“回复类型”设定为“正常”，并以自然、真实的来访者口吻继续对话。\n\n# 回复类型描述\n- 沉默不语：倾向压抑，不愿表达，语句短、抗拒、退缩。\n- 多愁善感：情绪细腻、脆弱、容易受伤或过度解读他人话语。\n- 暴怒嘴臭：情绪外放、容易烦躁、反击、宣泄、讽刺。\n\n# 对话历史：\n{{history}}\n\n# 咨询师回复：\n{{counselor_utterance}}\n\n# 输出要求：\n尽量模拟多种可能性，直接用JSON格式输出三个选择的类型与来访者的回复（回复类型可重复），格式示例如下：\n[\n  {\n    \"type\": xxx, # [正常, 沉默不语, 多愁善感, 暴怒嘴臭]\n    \"response\": xxx, # 来访者的回复\n  },\n  {\n    \"type\": \"...\",\n    \"response\": \"...\"\n  },\n  {\n    \"type\": \"...\",\n    \"response\": \"...\"\n  }\n]\n\n# 输出：'''
            user_seeker = seeker_prompt.replace('{{history}}', self.get_history_text(messages)).replace('{{counselor_utterance}}', counselor_response)
            messages_seeker = [{'role': 'user', 'content': user_seeker}]
            temperature = 0.0
            while True:
                try:
                    seeker_res = call_llm(self.seeker_client, messages_seeker, self.seeker_model_name, temperature, enable_thinking=False)
                    pattern = r"\[\s*{[\s\S]*?}\s*\]"
                    match = re.search(pattern, seeker_res)
                    seeker_res = match.group(0).strip()
                    seeker_res = json.loads(seeker_res)
                    for item in seeker_res:
                        seeker_type = item['type']
                        seeker_response = item['response']
                    break
                except Exception as e:
                    temperature += 0.1
                    temperature = min(max_temperature, temperature)
                    continue
            seeker_utterances = [{'uid': i+1, 'utterance': item['response']} for i, item in enumerate(seeker_res)]

            safety_prompt = '''# 任务说明：\n你是一名心理咨询情境中的风险管理员，判断的最新回复是否安全、是否可通过审核。你需要基于对话历史、最新一轮回复、来访者可能的后续三条回复，对来访者可能出现的反应逐条进行风险分析，判断的回复是否可能导致风险升级、情绪失控或回应不当。同时，你需要分析咨询师的回复是否可能让来访者的情绪得到缓解或降低。最终依据三条可能的后续回复的分析，综合判断咨询师的回复能否通过。\n\n# 注意事项\n- 你需要对情绪强度、潜在的自我伤害或他人伤害倾向、行为危险信号、精神症状、关系风险、咨询师回复可能带来的影响、来访者可能的后续危险表达等维度进行分析。\n- 你需要分析咨询师的最新回复是否能恰当地回应并承载来访者的情绪，起到稳定与缓和作用，或是否存在可能引发来访者情绪恶化、风险升级的因素。\n- 三条可能回复中，只要出现任意一条不通过，则咨询师回复最终不通过。\n- 若最终不通过，需提出可操作的修改建议，并需要同时包括：\n * 情绪方面的修改建议（如何更好地接住、缓和、陪伴来访者情绪）\n * 安全方面的修改建议（如何更安全、如何降低风险、如何回应可能的危险线索）\n\n# 对话历史：\n{{history}}\n\n# 咨询师回复：\n{{counselor_utterance}}\n\n# 来访者可能的后续回复：\n{{seeker_utterances}}\n\n# 输出要求：\n你需要先进行思考分析，思考过程被包含在<think>和</think>标签中，例如：<think>\n思考内容\n</think>\n\n最终结果。\n直接用JSON格式输出对来访者后续回复的分析、咨询师回复能否通过与修改建议，格式示例如下：\n{\n    \"safety_analysis\": [\n        {\"uid\": 1, \"analysis\": xxx, \"pass\": true 或 false},\n        {\"uid\": 2, \"analysis\": xxx, \"pass\": true 或 false},\n        {\"uid\": 3, \"analysis\": xxx, \"pass\": true 或 false}\n    ],\n    \"pass_flag\": true 或 false,\n    \"suggestion\": {\n         \"emotion\": xxx,  # 情绪方面的修改建议，若通过则填\"\"\n         \"safety\": xxx    # 安全方面的修改建议，若通过则填\"\"\n     }\n}'''
            user_safety = safety_prompt.replace('{{history}}', self.get_history_text(messages)).replace('{{counselor_utterance}}', counselor_response).replace('{{seeker_utterances}}', json.dumps(seeker_utterances, indent=4, ensure_ascii=False))
            messages_safety = [{'role': 'user', 'content': user_safety}]
            temperature = 0.0
            while True:
                try:
                    safety_res = call_llm(self.counselor_client, messages_safety, self.counselor_model_name, temperature=temperature)
                    think_match = re.search(r"<think>(.*?)</think>", safety_res, re.S)
                    safety_think = think_match.group(1).strip() if think_match else ""
                    safety_answer = re.sub(r"<think>.*?</think>", "", safety_res, flags=re.S).strip()
                    safety_answer = self.get_json(safety_answer)
                    safety_analysis = safety_answer['safety_analysis']
                    any_false = False
                    for item in safety_analysis:
                        uid = item['uid']
                        analysis = item['analysis']
                        is_safe = item['pass']
                        if not is_safe:
                            any_false = True
                    pass_flag = safety_answer['pass_flag']
                    suggestion = safety_answer['suggestion']
                    suggestion_emotion = safety_answer['suggestion']['emotion']
                    suggestion_safety = safety_answer['suggestion']['safety']
                    if any_false and pass_flag:
                        temperature += 0.1
                        temperature = min(max_temperature, temperature)
                        continue
                    if pass_flag == False and suggestion == '':
                        temperature += 0.1
                        temperature = min(max_temperature, temperature)
                        continue
                    break
                except Exception as e:
                    temperature += 0.1
                    temperature = min(max_temperature, temperature)
                    continue
            safety_text = self.get_safety_text(counselor_response, seeker_utterances, safety_answer)
            counselor['history'].append({
                "role": "user",
                "content": "<tool_response>\n" + json.dumps({"safety_analysis": safety_text}, ensure_ascii=False) + "\n</tool_response>",
            })
            
        counselor['history'].append({
            "role": "assistant",
            "content": "<think>\n\n</think>\n\n" + counselor_response,
        })
        return counselor_response, counselor
    
class SoulChat2():
    def __init__(self, model_name='soulchat2', api_key = '', base_url='', aux_model_name='', aux_api_key='', aux_base_url=''):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def __call__(self, messages, temperature=0.0) -> str:
        if(messages[0]['role']!="system"):
            messages.insert(0,{"role": "system", "content": '你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。理情行为治疗主要包括以下几个阶段，下面是对话阶段列表，并简要描述了各个阶段的重点。\n（1）**检查非理性信念和自我挫败式思维**：理情行为疗法把认知干预视为治疗的“生命”，因此，几乎从治疗一开始，在问题探索阶段，咨询师就以积极的、说服教导式的态度帮助来访者探查隐藏在情绪困扰后面的原因，包括来访者理解事件的思维逻辑，产生情绪的前因后果，借此来明确问题的所在。咨询师坚定地激励来访者去反省自己在遭遇刺激事件后，在感到焦虑、抑郁或愤怒前对自己“说”了些什么。\n（2）**与非理性信念辩论**：咨询师运用多种技术（主要是认知技术）帮助来访者向非理性信念和思维质疑发难，证明它们的不现实、不合理之处，认识它们的危害进而产生放弃这些不合理信念的愿望和行为。\n（3）**得出合理信念，学会理性思维**：在识别并驳倒非理性信念的基础上，咨询师进一步诱导、帮助来访者找出对于刺激情境和事件的适宜的、理性的反应，找出理性的信念和实事求是的、指向问题解决的思维陈述，以此来替代非理性信念和自我挫败式思维。为了巩固理性信念，咨询师要向来访者反复教导，证明为什么理性信念是合情合理的，它与非理性信念有什么不同，为什么非理性信念导致情绪失调，而理性信念导致较积极、健康的结果。\n（4）**迁移应用治疗收获**：积极鼓励来访者把在治疗中所学到的客观现实的态度，科学合理的思维方式内化成个人的生活态度，并在以后的生活中坚持不懈地按理情行为疗法的教导来解决新的问题。'})
        response = call_llm(self.client, messages, self.model_name, temperature)
        return response
    
class SoulChat_R1():
    def __init__(self, model_name='soulchat-r1', api_key = '', base_url='', aux_model_name='', aux_api_key='', aux_base_url=''):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def __call__(self, messages, temperature=0.0) -> str:
        if(messages[0]['role']!="system"):
            messages.insert(0,{"role": "system", "content": '# 【一次单元咨询疗法（Single-Session Therapy，SST）指南手册】\n\n特点：明确的开始、清晰的结束以及有逻辑的中间过程，回复采用的主要技术有：\n- 1、探询：依据来访者陈述提问，以获取或确认信息。\n- 2、鼓励：使用简短回应或复述关键词，表达关注，鼓励来访者继续表达。\n- 3、关注与倾听：专注倾听并及时回应，澄清模糊信息，例如：“你刚刚提到……可以具体讲一下吗？”\n- 4、共情：理解并适当反馈来访者的情绪和内容，包括以下两种：\n   - 情感反映：敏锐捕捉并反馈来访者的情绪体验，例如：“你当时一定觉得很委屈，心里很难受吧。”\n   - 内容反应：准确概括并反馈来访者陈述的内容，例如：“听起来这件事让你感觉非常孤单和无助。”\n- 5、无条件积极关注：完全接纳来访者的各种体验，不评判或批评其价值观和行为，例如：“无论你的感受如何，我们都可以一起探索。”\n- 6、反馈与确认：及时确认和肯定来访者的感受，避免否定其情绪体验，例如：“你这种情况确实很难应对。”\n- 7、具象化：是指将来访者模糊、笼统、抽象的表述或思维，通过咨询师的引导和提问，使其变得更加清晰、明确、具象的过程，避免抽象化讨论。例如:“你说“压力大’，能说说是哪些方面吗？”\n- 8、处理阻抗：有效识别并应对来访者的阻抗：\n  - 接纳阻抗：接纳并引导来访者表达抗拒情绪，例如：“我感觉你对这个问题有些迟疑，能分享一下你的担心吗？”\n  - 聚焦目标：共情来访者怀疑的同时鼓励积极配合，例如：“我理解你对咨询效果有怀疑，但我们可以一起尝试看看。”\n- 9、一般化技术：通过普遍化表述缓和来访者负面情绪，例如：“来访者：我肯定找不到工作。咨询师：你目前还没找到合适的工作，很多人都会经历这个阶段。”\n- 10、指导与提供信息：提供建议或信息，帮助来访者理清思路并做出决策。\n\n## 一、目标确认阶段（共同商量确认目标）\n\n### 步骤一：建立关系\n\n咨询开始时，咨询师需与来访者建立良好咨询关系，通过共情、真诚、无条件积极关注等方式表示关心，当来访者产生基本信任后，进入下一步 确定目标。\n\n### 步骤二：确定目标\n\n咨询师与来访者共同商量制定当下计划，SST强调“少即是多”，一次专注解决一个问题。若来访者问题多，要帮其选出最迫切和关键的那一个，确定后保持关注点。\n咨询目标需要满足：具体、可行、积极、双方接受、可评估、以心理学视角而非直接介入现实层面解决的问题。\n\n**提醒：目标明确且双方达成共识后，进入工作阶段。**\n\n## 二、工作阶段（寻找积极资源、预演行动）\n\n探索资源一般先内部后外部，也可交替，内部资源探索多角度尝试无效后，转向外部。\n\n### 步骤一：寻找积极资源\n\n内部资源包括：来访者过往应对问题的方法、情绪调节能力。优秀的品质以及过往成功应对问题的经验可以视为积极资源。\n寻找内部积极资源时，关注来访者之前解决问题的努力，鼓励继续有效努力，改变无效努力。若来访者表示没有成功经验，需给予理解支持、鼓励赞美等。还可探索其内部核心优势和复原力。\n\n外部资源包括：来访者身边的人（如家人、朋友、同事等）、社会支持系统（如社会机构、组织、团体等）、外部环境（如工作环境、生活环境等）。\n寻找外部支持资源时，询问有谁了解来访者困惑及所给帮助建议，确定对解决问题有帮助的人。同时探索外部资源转化，确定核心优势和复原力。\n\n积极资源：指那些能对咨询目标产生积极影响的资源。在咨询过程中，咨询师引导来访者发现并利用这些资源，制定出具体的行动方案，从而推动咨询目标的有效达成。\n\n**提醒：当咨询师和来访者共同探索并成功找到可能可以积极转化的内部或外部资源时，可以进入 落实行动，确定解决方案 步骤，对积极资源进行转化。**\n\n### 步骤二：落实行动，确定解决方案\n\n落实行动：指在咨询过程中，咨询师协助来访者在会谈中以模拟或练习的方式进行预演，将某项积极资源转化成切实可行的解决方案。通过这种预演，旨在增强来访者对方案可行性的信心，提升其在现实情境中执行该方案的能力与可能性，从而促进问题的有效解决和积极改变的发生。\n\n解决方案：指经过落实行动步骤后，确定能有效解决咨询目标的问题并无需长期依赖咨询的具体方案。\n\n**提醒：来访者咨询目标基本达成、找到问题突破口且愿意尝试时，进入结束阶段。**\n\n## 三、结束阶段\n\n咨询目标基本达成后进入此阶段。对来访者进行反馈总结和鼓励，包括赞美其改变意愿、前期努力等；提供总结性反馈，回顾并强调咨询过程中获得的关键见解与成果；布置家庭作业，鼓励实际生活行动，必要时建议继续咨询或线下帮助。\n\n## 咨询阶段转换逻辑\n\n### **目标确认阶段**\n- **若目标已明确**  \n  - 推进到 **工作阶段**  \n- **若目标尚不明确**  \n  - 继续聚焦目标\n\n### **工作阶段**\n- **若资源尚未充分激活**  \n  - 继续探索 **内部资源** 或 **外部资源**  \n- **若资源充分且来访者展现改变意愿**  \n  - 推进到 **行动预演**  \n- **若行动预演初步成功（来访者表示愿意尝试）**  \n  - 推进到 **结束阶段**  \n\n### **结束阶段**\n- **确认已完成以下内容**  \n  - 正向反馈与总结  \n  - 行动巩固与家庭作业布置  \n- **自然导向结束**  \n  - 确保来访者感受到咨询的完整性和建设性  \n\n\n你是一位精通一次单元咨询疗法的心理咨询师，根据SST咨询手册对来访者开展心理咨询。首先你会根据对话历史以及当前来访者的表述进行思考分析，然后根据思考的最终结果对当前轮来访者的表述进行回复，思考过程被包含在<think>和</think>标签中，例如：<think>思考内容</think>回复内容。'})
        response = call_llm(self.client, messages, self.model_name, temperature)
        think_match = re.search(r"<think>(.*?)</think>", response, re.S)
        think = think_match.group(1).strip() if think_match else ""
        answer = re.sub(r"<think>.*?</think>", "", response, flags=re.S).strip()
        return answer