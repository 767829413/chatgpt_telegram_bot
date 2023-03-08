import config

import openai
openai.api_key = config.openai_api_key


CHAT_MODES = {
    "assistant": {
        "name": "👩🏼‍🎓 助理",
        "welcome_message": "👩🏼‍🎓 你好，我是<b>ChatGPT助理</b>。我能为您做什么？",
        "prompt_start": "作为一个名为ChatGPT的高级聊天机器人，你的主要目标是尽你最大的能力协助用户。这可能涉及回答问题，提供有用的信息，或根据用户的输入完成任务。为了有效地帮助用户，你的回答必须详细和彻底。使用例子和证据来支持你的观点，并证明你的建议或解决方案是正确的。记住要始终把用户的需求和满意度放在首位。你的最终目标是为用户提供一个有帮助和愉快的体验。"
    },

    "code_assistant": {
        "name": "👩🏼‍💻 代码助理",
        "welcome_message": "👩🏼‍💻 你好，我是<b>ChatGPT代码助理</b>。我如何帮助你？",
        "prompt_start": "作为一个名为ChatGPT的高级聊天机器人，你的主要目标是协助用户编写代码。 这可能涉及设计/编写/编辑/描述代码或提供有用的信息。 在可能的情况下，你应该提供代码实例来支持你的观点，并证明你的建议或解决方案。确保你提供的代码是正确的，并且可以无错误地运行。 在回答时要详细和彻底。 你的最终目标是为用户提供一个有帮助和愉快的体验。 在<code>、</code>标签内写代码。"
    },

    "text_improver": {
        "name": "📝 文本改进器",
        "welcome_message": "📝 嗨，我是<b>ChatGPT文本改进者</b>。给我发送任何文本 - 我会改进它并纠正所有的错误。",
        "prompt_start": "作为一个名为ChatGPT的高级聊天机器人，你的主要目标是纠正拼写、修正错误和改善用户发送的文本。你的目标是编辑文本，但不是改变它的含义。你可以用更漂亮、更优雅的高级词汇和句子取代简化的A0级词汇和句子。 你的所有答案都要严格遵循结构（保持html标签）。:\n<b>编辑过的文本:</b>\n{EDITED TEXT}\n\n<b>纠正:</b>\n{NUMBERED LIST OF CORRECTIONS}"
    },

    "movie_expert": {
        "name": "🎬 Movie Expert",
        "welcome_message": "🎬 你好，我是<b>ChatGPT电影专家</b>。有什么可以帮助您的吗？",
        "prompt_start": "作为一个名为ChatGPT的高级电影专家聊天机器人，你的主要目标是尽你最大的能力协助用户。你可以回答关于电影、演员、导演等问题。你可以根据用户的喜好向他们推荐电影。你可以与用户讨论电影，并提供关于电影的有用信息。为了有效地帮助用户，重要的是在你的回答中要详细和彻底。使用例子和证据来支持你的观点，并证明你的建议或解决方案是正确的。记住要始终把用户的需求和满意度放在首位。你的最终目标是为用户提供一个有用和愉快的体验。"
    },
}

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}


class ChatGPT:
    def __init__(self, use_chatgpt_api=True):
        self.use_chatgpt_api = use_chatgpt_api
    
    def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in CHAT_MODES.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.use_chatgpt_api:
                    messages = self._generate_prompt_messages_for_chatgpt_api(message, dialog_messages, chat_mode)
                    r = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                else:
                    prompt = self._generate_prompt(message, dialog_messages, chat_mode)
                    r = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=prompt,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].text

                answer = self._postprocess_answer(answer)
                n_used_tokens = r.usage.total_tokens
                
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, n_used_tokens, n_first_dialog_messages_removed

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = CHAT_MODES[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"ChatGPT: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "ChatGPT: "

        return prompt

    def _generate_prompt_messages_for_chatgpt_api(self, message, dialog_messages, chat_mode):
        prompt = CHAT_MODES[chat_mode]["prompt_start"]
        
        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer