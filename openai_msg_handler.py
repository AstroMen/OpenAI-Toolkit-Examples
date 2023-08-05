import re
import openai
import logging


class MessageHandler:
    def __init__(self, max_tokens=2000, model_name='gpt-4'):
        self.max_tokens = max_tokens
        self.messages = []
        self.model_name = model_name

    def __generate_message(self, role, content):
        return {"role": role, "content": content}

    def append_message(self, role, content):
        new_msg = self.__generate_message(role, content)
        self.messages.append(new_msg)
        # message_list.append(new_msg)
        # return message_list

    def generate_and_append_message(self, role, content):
        self.append_message(role, content)

    def get_msg_list(self):
        return self.messages

    async def get_ans(self, max_tokens, temperature=1, resp_cnt=1, stop_sentence=None, presence_penalty=0):
        """Get answer from GPT-4 model"""
        res = ''
        usage = {'total_tokens': 0, 'completion_tokens': 0, 'prompt_tokens': 0}
        status = 1
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=resp_cnt,  # 对每条prompt生成多少条结果
                stream=False,  # Whether to stream partial results
                stop=stop_sentence,
                presence_penalty=presence_penalty
            )
            answer = response.choices[0].message.content
            res = re.sub(r'\n[\s]*\n+', '\n', answer)  # Remove extra blank newlines
            usage = response["usage"]
            return res, usage, status
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            # openai.error.InvalidRequestError: This model's maximum context length is 8192 tokens. However, you requested 8322 tokens (4322 in the messages, 4000 in the completion). Please reduce the length of the messages or completion.
            if 'Limit: 3 / min' in str(e):
                status = '-2'
            elif '8192 tokens' in str(e) or "This model's maximum context length is 4097 tokens" in str(e):
                status = 0
            elif 'Limit: 40000 / min' in str(e) or 'on tokens per min. Limit: 90000 / min':
                status = -3
            else:
                # An error occurred: Request timed out: HTTPSConnectionPool(host='api.openai.com', port=443): Read timed out. (read timeout=600)
                status = -1
            return res, usage, status
