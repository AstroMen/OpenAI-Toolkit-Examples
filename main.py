import argparse
import os
import time
import random
import asyncio
import openai
from dotenv import load_dotenv
import logging

from openai_image_processor import OpenaiImageProcessor
from openai_msg_handler import MessageHandler
from prompt_preprocessor import PromptPreprocessor
from token_manager import TokenManager
from utils.file_ops import FileOperations

logging.basicConfig(
    level=logging.INFO,
    format='\033[93m%(levelname)s [%(filename)s Line: %(lineno)d] %(asctime)s\033[0m %(message)s'
)


class OpenAIWrapper:
    def __init__(self, model_name):
        self.load_environment()
        os.environ['OPENAI_API_KEY'] = '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'
        # self.api_key = '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'
        self.api_key = os.getenv('OPENAI_API_KEY') #if 'OPENAI_KEY' in os.getenv('OPENAI_KEY')
        if self.api_key is None:
            raise Exception("Please set the 'OPENAI_KEY' environment variable.")
        openai.api_key = self.api_key
        self.model_name = model_name
        self.model_id = self.get_model_id(self.model_name)
        self.max_tokens = 2000
        self.max_tokens_long = 4000
        self.tokens_limit = 6000
        self.sleep_time = 1  # for no binding card case: 18
        self.dir_chat_data = './data/chat/'
        if not os.path.exists(self.dir_chat_data):
            os.makedirs(self.dir_chat_data)
        # self.dir_image_data = './data/image/'
        self.gen_file_name = None
        self.gen_bk_file = None
        self.total_token1 = [0]
        self.token1 = [0]
        self.total_token2 = [0]
        self.token2 = [0]
        self.role_name1 = ''
        self.role_name2 = ''
        self.total_rounds, self.remaining_rounds = 30, 30
        self.end_hint = 'Closing of the discussion'
        self.prompter = PromptPreprocessor(self.gen_file_name, self.gen_bk_file, self.role_name1, self.role_name2, self.dir_chat_data, self.model_name, self.model_id)

    def load_environment(self):
        load_dotenv()
        # if 'OPENAI_API_KEY' in os.getenv('OPENAI_API_KEY'):
        #     os.environ['OPENAI_API_KEY'] = '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'
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise Exception("Please set the 'OPENAI_KEY' environment variable.")
        openai.api_key = self.api_key
    #
    # def save_and_print(self, message, file):
    #     file.write(message + '\n')
    #     file.flush()
    #     logging.info(message)

    def get_model_id(self, model_name):
        model_dict = {
            'gpt-4': 1,
            'gpt-4-0314': 2,
            'gpt-4-0613': 3,
            'gpt-3.5-turbo': 4,
            'gpt-3.5-turbo-16k': 5,
            'gpt-3.5-turbo-16k-0613': 6
        }
        return model_dict[model_name]
    #
    # def get_num_tokens_from_msgs_by_scratch(self, messages, model):
    #     # Define the function num_tokens_from_messages, which returns the number of tokens used by a set of messages.
    #     # Attempt to get the model's encoding
    #     try:
    #         encoding = tiktoken.encoding_for_model(model)
    #     except KeyError:
    #         # If the model is not found, use cl100k_base encoding and give a warning
    #         logging.warning("Warning: model not found. Using cl100k_base encoding.")
    #         encoding = tiktoken.get_encoding("cl100k_base")
    #     # Set the number of tokens for different models
    #     if model in {
    #         "gpt-3.5-turbo-0613",
    #         "gpt-3.5-turbo-16k-0613",
    #         "gpt-4-0314",
    #         "gpt-4-32k-0314",
    #         "gpt-4-0613",
    #         "gpt-4-32k-0613",
    #     }:
    #         tokens_per_message = 3
    #         tokens_per_name = 1
    #     elif model == "gpt-3.5-turbo-0301":
    #         tokens_per_message = 4  # Each message follows {role/name}\n{content}\n format
    #         tokens_per_name = -1  # If there is a name, the role will be omitted
    #     elif "gpt-3.5-turbo" in model:
    #         # For gpt-3.5-turbo model there may be updates, here return the number of tokens assumed to be gpt-3.5-turbo-0613 and give a warning
    #         logging.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
    #         return self.get_num_tokens_from_msgs_by_scratch(messages, model="gpt-3.5-turbo-0613")
    #     elif "gpt-4" in model:
    #         # For gpt-4 model there may be updates, here return the number of tokens assumed to be gpt-4-0613 and give a warning
    #         logging.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
    #         return self.get_num_tokens_from_msgs_by_scratch(messages, model="gpt-4-0613")
    #     elif model in {"davinci", "curie", "babbage", "ada"}:
    #         logging.warning("Warning: gpt-3 related model is used. Returning num tokens assuming gpt2.")
    #         encoding = tiktoken.get_encoding("gpt2")
    #         num_tokens = 0
    #         # Only calculate the content
    #         for message in messages:
    #             for key, value in message.items():
    #                 if key == "content":
    #                     num_tokens += len(encoding.encode(value))
    #         return num_tokens
    #     else:
    #         # For models that are not implemented, throw a not implemented error
    #         raise NotImplementedError(
    #             f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
    #         )
    #     num_tokens = 0
    #     # Calculate the number of tokens for each message
    #     for message in messages:
    #         num_tokens += tokens_per_message
    #         for key, value in message.items():
    #             num_tokens += len(encoding.encode(value))
    #             if key == "name":
    #                 num_tokens += tokens_per_name
    #     num_tokens += 3  # Each response begins with the assistant
    #     return num_tokens
    #
    # def remove_earliest_messages_if_needed(self, msg, total_token, token):
    #     """Remove the earliest two messages from msg1 and msg2 if num_token is greater than a threshold."""
    #     try:
    #         if total_token[-1] > self.tokens_limit:
    #             logging.info('Removing tokens...')
    #             logging.info(f"Removing tokens: total_token={total_token}, token={token}")
    #             if len(msg) >= 5:
    #                 msg.pop(2)
    #                 msg.pop(2)
    #             if len(total_token) >= 3:
    #                 total_token[-1] -= total_token.pop(1)
    #                 total_token[-1] -= total_token.pop(1)
    #             if len(token) >= 5:
    #                 token[-1] -= token.pop(3)
    #                 token[-1] -= token.pop(3)
    #             logging.info(f"Removed tokens: total_token={total_token}, token={token}")
    #     except Exception as e:
    #         logging.error(f'An error occurred: {e}')
    #         logging.info(f'msg={msg}')
    #         logging.info(f'total_token={total_token}, token={token}')
    #     return msg, total_token, token

    # async def get_ans(self, msg, max_tokens, temperature=1, resp_cnt=1, stop_sentence=None, presence_penalty=0):
    #     """Get answer from GPT-4 model"""
    #     res = ''
    #     usage = {'total_tokens': 0, 'completion_tokens': 0, 'prompt_tokens': 0}
    #     status = 1
    #     try:
    #         response = openai.ChatCompletion.create(
    #             model=self.model_name,
    #             messages=msg,
    #             max_tokens=max_tokens,
    #             temperature=temperature,
    #             n=resp_cnt,  # 对每条prompt生成多少条结果
    #             stream=False,  # 是否回流部分结果
    #             stop=stop_sentence,
    #             presence_penalty=0
    #         )
    #         answer = response.choices[0].message.content
    #         res = re.sub(r'\n[\s]*\n+', '\n', answer)  # Remove extra blank newlines
    #         usage = response["usage"]
    #         return res, usage, status
    #     except Exception as e:
    #         logging.error(f"An error occurred: {e}")
    #         # openai.error.InvalidRequestError: This model's maximum context length is 8192 tokens. However, you requested 8322 tokens (4322 in the messages, 4000 in the completion). Please reduce the length of the messages or completion.
    #         if 'Limit: 3 / min' in str(e):
    #             status = '-2'
    #         elif '8192 tokens' in str(e) or "This model's maximum context length is 4097 tokens" in str(e):
    #             status = 0
    #         elif 'Limit: 40000 / min' in str(e) or 'on tokens per min. Limit: 90000 / min':
    #             status = -3
    #         else:
    #             # An error occurred: Request timed out: HTTPSConnectionPool(host='api.openai.com', port=443): Read timed out. (read timeout=600)
    #             status = -1
    #         return res, usage, status
    #
    # def get_params_for_interview(self):
    #     role_info = {
    #         0: {'short': 'sde', 'position': 'Software Engineer', 'question_related': 'algorithm and system design'},
    #         1: {'short': 'backend', 'position': 'Back-End Developer', 'question_related': ''},
    #         2: {'short': 'frontend', 'position': 'Front-End Developer', 'question_related': ''},
    #         3: {'short': 'full_stack', 'position': 'Full Stack Developer', 'question_related': ''},
    #         4: {'short': 'sys', 'position': 'Systems Engineer', 'question_related': ''},
    #         5: {'short': 'android', 'position': 'Android Developer', 'question_related': ''},
    #         6: {'short': 'ios', 'position': 'IOS Developer', 'question_related': ''},
    #         7: {'short': 'security', 'position': 'Security Engineer', 'question_related': ''},
    #         8: {'short': 'security_detect', 'position': 'Security Engineer - Detection and Response',
    #             'question_related': ''},
    #         9: {'short': 'devops', 'position': 'DevOps Engineer', 'question_related': 'Linux'},
    #         10: {'short': 'data', 'position': 'Data Engineer', 'question_related': ''},
    #         11: {'short': 'ml', 'position': 'Machine Learning Engineer', 'question_related': ''},
    #         12: {'short': 'db', 'position': 'Database Administrator', 'question_related': ''},
    #         13: {'short': 'test', 'position': 'Quality Assurance Engineer', 'question_related': ''},
    #         14: {'short': 'cloud', 'position': 'Cloud Engineer', 'question_related': ''},
    #         15: {'short': 'pm', 'position': 'Project Manager', 'question_related': 'agile and scrum'},
    #     }
    #     print("Available roles:")
    #     for key, value in role_info.items():
    #         print(f"ID: {key}, Position: {value['position']}")
    #
    #     inst = input("Please input the ID for the role: ")
    #     return role_info[int(inst)]

    # def generate_message(self, role, content):
    #     return {"role": role, "content": content}

    # def append_message(self, message_list, role, content):
    #     new_msg = self.generate_message(role, content)
    #     message_list.append(new_msg)
    #     return message_list

    # def generate_and_append_message(self, message_list, role, content):
    #     self.append_message(message_list, role, content)
    #     return message_list

    async def chat_round(self, msg1_handler, msg2_handler, round_number=0):
        # ask
        executed = True
        ans1 = ''
        while executed:
            # usage: total_tokens, completion_tokens, prompt_tokens
            ans1, usage, status = await msg1_handler.get_ans(self.max_tokens)
            if len(ans1) == 0 and status == 0:
                msg1, self.total_token1, self.token1 = TokenManager.remove_earliest_messages_if_needed(msg1_handler.get_msg_list(), self.total_token1, self.token1)
                continue
            if len(ans1) == 0 and status == -1:
                time.sleep(60 + random.randint(0, 5))
                continue
            if len(ans1) == 0 and status == -2:
                time.sleep(self.sleep_time + random.randint(2, 5))
                continue
            if len(ans1) == 0 and status == -3:
                time.sleep(50 + random.randint(2, 5))
                continue

            if status > 0:
                # append message
                msg1_handler.generate_and_append_message("assistant", ans1)
                # msg1 = self.generate_and_append_message(msg1, "assistant", ans1)
                self.token1.append(self.token1[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg1_handler.get_msg_list()[-1]], model=self.model_name))
                msg2_handler.generate_and_append_message("user", ans1)
                # msg2 = self.generate_and_append_message(msg2, "user", ans1)
                self.token2.append(self.token2[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg2_handler.get_msg_list()[-1]], model=self.model_name))
                executed = False

                # calculate token
                if self.total_token1[0] == 0:
                    self.total_token1[0] = usage['prompt_tokens']
                self.total_token1.append(usage['total_tokens'])
                # self.total_token1.append(self.total_token1[-1] + usage['completion_tokens'])
            msg1, self.total_token1, self.token1 = TokenManager.remove_earliest_messages_if_needed(msg1_handler.get_msg_list(), self.total_token1, self.token1)
            logging.info(f'current usage: {usage}, total_token1: {self.total_token1[-1]}, token1: {self.token1[-1]}')

        # finish ask
        time.sleep(self.sleep_time + random.randint(0, 5))

        # answer
        executed = True
        ans2 = ''
        while executed:
            ans2, usage, status = await msg2_handler.get_ans(self.max_tokens_long)
            if len(ans2) == 0 and status == 0:
                msg2, self.total_token2, self.token2 = TokenManager.remove_earliest_messages_if_needed(msg2_handler.get_msg_list(), self.total_token2, self.token2)
                continue
            if len(ans2) == 0 and status == -1:
                time.sleep(600 + random.randint(0, 5))
                continue
            if len(ans2) == 0 and status == -2:
                time.sleep(self.sleep_time + random.randint(2, 5))
                continue
            if len(ans2) == 0 and status == -3:
                time.sleep(50 + random.randint(2, 5))
                continue

            if status > 0:
                msg2_handler.generate_and_append_message("assistant", ans2)
                self.token2.append(self.token2[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg2_handler.get_msg_list()[-1]], model=self.model_name))
                msg1_handler.generate_and_append_message("user", ans2)
                self.token1.append(self.token1[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg1_handler.get_msg_list()[-1]], model=self.model_name))
                executed = False

                # calculate token
                if self.total_token2[0] == 0:
                    self.total_token2[0] = usage['prompt_tokens']
                self.total_token2.append(usage['total_tokens'])
                # self.total_token2.append(self.total_token2[-1] + usage['completion_tokens'])
            msg2, self.total_token2, self.token2 = TokenManager.remove_earliest_messages_if_needed(msg2_handler.get_msg_list(), self.total_token2, self.token2)
            logging.info(f'current usage: {usage}, total_token2: {self.total_token2[-1]}, token2: {self.token2[-1]}')

        # finish answer
        time.sleep(self.sleep_time + random.randint(0, 5))

        with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
            FileOperations.save_and_print(
                '================================================================================================', file)
            FileOperations.save_and_print(f'round: {round_number}', file)
            FileOperations.save_and_print(f'{self.role_name1}: {ans1}', file)
            FileOperations.save_and_print(f'{self.role_name2}: {ans2}', file)

        return msg1_handler.get_msg_list(), msg2_handler.get_msg_list()

#     def set_beginning_for_interview(self):
#         params = prompter.get_params_for_interview()
#         self.gen_file_name = f"gen_interview_{params['short']}.txt"
#         self.role_name1 = 'Interviewer'
#         self.role_name2 = 'Candidates'
#
#         position, question_related = params['position'], f" focused on {params['question_related']}"
#
#         interviewer_content = f"""As a senior interviewer for the {position} position, you are tasked with conducting a mock interview that should span approximately one hour.
#                     Your main goal is to ask a series of technical interview questions{question_related}.
#                     It is important to ensure that the number and complexity of the questions are enough to fill the entire duration.
#                     Additionally, you are responsible for providing corrections, prompts, engaging in discussions, and evaluating the candidate's responses as needed.
#                     Please proceed by asking one question at a time.
#                     """
#         candidate_content = f"""As a candidate for the position of a highly experienced {position}, you are participating in a one-hour mock interview.
#                     You are expected to respond to the interviewer's inquiries with precise and detailed technical information.
#                     If applicable, please include examples to enhance the clarity and support of your answers."
#                     """
#         greetings = "Let's start today's interview."
#         greetings_resp = "Okay."
#
#         with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
#             FileOperations.save_and_print('****************************************************', file)
#             FileOperations.save_and_print(f"* {self.model_name} *", file)
#             FileOperations.save_and_print(f"* {position},{question_related} *", file)
#             FileOperations.save_and_print('****************************************************', file)
#         return interviewer_content, candidate_content, greetings, greetings_resp
#
#     def set_beginning_for_trend(self):
#         self.gen_file_name = 'gen_trend_economic.txt'
#         self.role_name1 = 'Partner1'
#         self.role_name2 = 'Partner2'
#
#         partner1 = """Engage in a thorough and in-depth discussion with your partner to explore and analyze the economic trends that you both predict will significantly influence and mold the global landscape in the upcoming decade.
#         Indicate the end of the discussion by saying 'Closing of the discussion, exit.'.
#         """
#         partner2 = """Engage in a thorough and in-depth discussion with your partner to explore and analyze the economic trends that you both predict will significantly influence and mold the global landscape in the upcoming decade.
#         Your discussion should include but not be limited to topics such as technological advancements, shifts in global trade, changes in population demographics, and the impact of climate change on the economy.
#         Be sure to provide supporting evidence for your predictions and consider different perspectives, including potential challenges and opportunities.
#         Additionally, discuss how these trends might affect different sectors such as technology, manufacturing, finance, and services.
#         Indicate the end of the discussion by saying 'Closing of the discussion, exit.'.
#         """
#         greetings = "Let's start our talk."
#         greetings_resp = "Okay."
#
#         with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
#             FileOperations.save_and_print('****************************************************', file)
#             FileOperations.save_and_print(f"* {self.model_name} *", file)
#             FileOperations.save_and_print('****************************************************', file)
#         return partner1, partner2, greetings, greetings_resp
#
#     def set_beginning_for_entrepreneurship(self, lang='en'):
#         self.gen_file_name = f'gen_entrepreneurship_plan{self.model_id}.txt'
#         self.role_name1 = 'Partner1'
#         self.role_name2 = 'Partner2'
#         if lang == 'en':
#             lang = ''
#         elif lang == 'zh':
#             lang = 'It is important to conduct the discussion in Chinese.'
#
#         partner1 = f"""Please allocate a continuous four-hour period with your partner to fully immerse yourselves in the realm of software industry entrepreneurship.
#         The primary aim of this session is to generate groundbreaking ideas and formulate a viable startup concept that is closely tied to any facet of software development.
#         Throughout the conversation, carefully analyze the advantages and disadvantages of the current idea and delve deep into the specifics.
#         Your ultimate objective is to create a comprehensive business plan that encompasses all pertinent aspects.
#         Once the discussion reaches its conclusion, please condense your findings into a succinct version of the business plan.
#         It is crucial to engage in a meticulous and exhaustive discussion, ensuring that no topic is left unexplored and avoiding premature conclusions.
#         Utilize the entire four-hour timeframe to maximize this opportunity for exploration and exploitation.
#         {lang}
#         Signal the end of the discussion by uttering the word 'Closing of the discussion, exit.'.
#         """
#         partner2 = f"""Please dedicate a complete four-hour session with your partner to fully immerse yourselves in the software industry entrepreneurship.
#         The main goal of this session is to generate innovative ideas and develop a feasible startup concept that is closely related to any aspect of software development.
#         Throughout the conversation, carefully analyze the advantages and disadvantages of the current idea and delve deep into the specifics.
#         Your ultimate objective is to create a comprehensive business plan that covers all relevant aspects.
#         Once the discussion is complete, please condense your findings into a concise version of the business plan.
#         It is essential to engage in a thorough and exhaustive discussion, ensuring that no topic is left unexplored and avoiding drawing premature conclusions.
#         Utilize the entire four-hour duration to maximize this opportunity for exploration and exploitation.
#         {lang}
#         Indicate the end of the discussion by saying 'Closing of the discussion, exit.'.
#         """
#         greetings = "Let's start our talk."
#         greetings_resp = "Okay."
#
#         with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
#             FileOperations.save_and_print('****************************************************', file)
#             FileOperations.save_and_print(f"* {self.model_name} *", file)
#             FileOperations.save_and_print('****************************************************', file)
#         return partner1, partner2, greetings, greetings_resp
#
#     def set_beginning_for_story(self):
#         self.gen_file_name = f'gen_story_plots.txt'
#         self.role_name1 = 'Writer'
#         self.role_name2 = ''
#
#         partner1 = f"""Please write a story for children aged 3-6, composed of exactly 50 short and simple sentences.
#         Begin by providing a title for the story, and then follow these guidelines:
# Title: (Provide a creative and engaging title for the story here. The title must not contain any punctuation and should be as concise as possible.)
# 1. Avoid Pronouns: Do not use pronouns like "he," "she," "it," "they," "their," etc. Refer to characters by their names or descriptions.
# 2. Descriptive Nouns: Use more descriptive nouns for objects and settings.
# 3. Character Emotions: Include the emotions of the characters.
# 4. Background Details: Provide detailed backgrounds, such as "on the busy city street with a little giraffe."
#
# For example:
# Title: The Little Bear's Joyful Day
# 1. The happy little bear found a colorful umbrella.
# 2. The little bear walked to the sparkling river.
# 3. On the busy city street, a curious little giraffe saw a fish.
# 4. The fish, feeling playful, smiled at the little giraffe.
# 5. The little giraffe felt joy and danced.
# 6. ...
# 7. ...
#
# Continue this story or create a new one, ensuring that the entire story consists of 50 brief, clear sentences, following the guidelines above, and is suitable for children aged 3-6 to understand.
#         """
#         greetings = "Please create one story."
#         greetings_resp = ""
#         self.gen_bk_file = 'image_prompts.csv'
#
#         with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
#             FileOperations.save_and_print('****************************************************', file)
#             FileOperations.save_and_print(f"* {self.model_name} *", file)
#             FileOperations.save_and_print('****************************************************', file)
#         return partner1, greetings

    # async def dalle_gen_img(self, prompt, style='cartoon', img_size='512x512'):
    #     # 使用DALL-E API生成图片
    #     response = openai.Image.create(
    #         model="image-alpha-001",
    #         prompt=prompt + ', ' + style,
    #         n=1,  # Number of images to generate
    #         size=img_size,  # Output image size
    #         response_format="url"
    #     )
    #     # 获取生成的图片URL
    #     image_url = response['data'][0]['url']
    #     return image_url
    #
    # async def generate_story_images(self, src_file, dst_dir):
    #     with open(src_file, 'r') as file:
    #         data_list = file.read().strip().split('\n')
    #     data = pd.DataFrame(data_list, columns=['prompt'])
    #
    #     title = re.sub(r'^Title: ', '', data.iloc[1]['prompt'])
    #     logging.info(f'title: {title}')
    #     story_path = os.path.join(dst_dir, title.replace(' ', '_') + '/')
    #     logging.info(f'story_path: {story_path}')
    #     if not os.path.exists(story_path):
    #         os.makedirs(story_path)
    #
    #     # generate images based on text prompt from csv
    #     for index, row in data.iloc[2:].iterrows():
    #         prompt = re.sub(r'^\d+\.\s', '', row['prompt'])
    #         plot_id = index - 1
    #         image_url = await self.dalle_gen_img(prompt)
    #
    #         # save images
    #         if not os.path.exists(dst_dir):
    #             os.makedirs(dst_dir)
    #         image_response = requests.get(image_url)
    #         img = Image.open(BytesIO(image_response.content))
    #         img.save(f"{story_path}generated_image_{plot_id}.png")
    #         logging.info(f"Generating image {plot_id}: {prompt}")

    async def main(self, inst=3):
        sys_content1, sys_content2, greetings, greetings_resp = 'You are an assistant.', 'You are an assistant.', 'Hello', 'Hello'
        __inst_fork = {
            1: self.prompter.set_beginning_for_interview,
            2: self.prompter.set_beginning_for_trend,
            3: self.prompter.set_beginning_for_entrepreneurship
        }
        sys_content1, sys_content2, greetings, greetings_resp = __inst_fork[inst]()

        msg1_handler = MessageHandler(model_name=self.model_name)
        msg2_handler = MessageHandler(model_name=self.model_name)

        msg1_handler.generate_and_append_message("system", sys_content1)
        msg2_handler.generate_and_append_message("system", sys_content2)
        # msg1 = [self.generate_message("system", sys_content1), ]
        # msg2 = [self.generate_message("system", sys_content2), ]
        self.token1 = [TokenManager.get_num_tokens_from_msgs_by_scratch(msg1_handler.get_msg_list(), model=self.model_name)]
        self.token2 = [TokenManager.get_num_tokens_from_msgs_by_scratch(msg2_handler.get_msg_list(), model=self.model_name)]

        msg1_handler.generate_and_append_message("assistant", greetings)
        msg2_handler.generate_and_append_message("user", greetings)
        # msg1 = self.generate_and_append_message(msg1, "assistant", greetings)
        # msg2 = self.generate_and_append_message(msg2, "user", greetings)
        self.token1.append(self.token1[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg1_handler.get_msg_list()[-1]], model=self.model_name))
        self.token2.append(self.token2[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg2_handler.get_msg_list()[-1]], model=self.model_name))

        msg1_handler.generate_and_append_message("user", greetings_resp)
        msg2_handler.generate_and_append_message("assistant", greetings_resp)
        # msg1 = self.generate_and_append_message(msg1, "user", greetings_resp)
        # msg2 = self.generate_and_append_message(msg2, "assistant", greetings_resp)
        self.token1.append(self.token1[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg1_handler.get_msg_list()[-1]], model=self.model_name))
        self.token2.append(self.token2[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg2_handler.get_msg_list()[-1]], model=self.model_name))
        logging.info(f'current usage: total_token1: {self.total_token1[-1]}, token1: {self.token1[-1]}, total_token2: {self.total_token2[-1]}, token2: {self.token2[-1]}')

        offset = 0

        with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
            while self.remaining_rounds:
                round_number = self.total_rounds - self.remaining_rounds + 1 + offset
                msg1, msg2 = await self.chat_round(msg1_handler, msg2_handler, round_number)
                self.remaining_rounds -= 1
                if self.end_hint in msg1[-1]['content'] or self.end_hint in msg2[-1]['content']:
                    break

    async def run_bot(self):
        img_processor = OpenaiImageProcessor()
        msg_handler = MessageHandler()

        sys_content, greetings = self.prompter.set_beginning_for_story()
        # msg = [self.generate_message("system", sys_content), ]
        # self.generate_and_append_message(msg, "user", greetings)
        msg_handler.generate_and_append_message("system", sys_content)
        msg_handler.generate_and_append_message("user", greetings)

        executed = True
        ans = ''
        while executed:
            ans, num_token, status = await msg_handler.get_ans(self.max_tokens)
            # if len(ans) == 0 and num_token == 0:
            #     msg1, self.total_token1, self.token1 = TokenManager.remove_earliest_messages_if_needed(msg1, self.total_token1, self.token1)
            #     continue
            if len(ans) == 0 and status == -1:
                time.sleep(600 + random.randint(0, 5))
                continue
            if len(ans) == 0 and status == -2:
                time.sleep(self.sleep_time + random.randint(2, 5))
                continue
            if len(ans) == 0 and status == -3:
                time.sleep(50 + random.randint(2, 5))
                continue
            executed = False

        with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
            FileOperations.save_and_print(
                '================================================================================================', file)
            FileOperations.save_and_print(f'{self.role_name1}: \n{ans}', file)

        with open(os.path.join(self.gen_bk_file), 'w') as file:
            FileOperations.save_and_print(f'prompt\n{ans}', file)

        subsequent_task_fork = {

        }

        await img_processor.generate_story_images(self.gen_bk_file)

        time.sleep(random.randint(2, 5))


def parse_arguments():
    parser = argparse.ArgumentParser(description="ChatBot Script")
    parser.add_argument('-t', "--type", type=int, default=1, help="Type of mode to run: 1. AI pair, 2. stand-alone")
    parser.add_argument('-m', "--model_name", type=str, default='gpt-4', help="Model name: gpt-4, gpt-4-0314, gpt-4-0613")
    parser.add_argument('-r', "--round_cnt", type=int, default=50, help="Number of rounds to run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    mode = args.type
    model_name = args.model_name
    round_cnt = args.round_cnt

    while round_cnt > 0:
        # Initialize the bot with the desired model
        bot = OpenAIWrapper(model_name)
        if mode == 1:
            print("1: interview, 2: economic trends, 3: entrepreneurship")
            inst = 3
            # inst = int(input("Please input the task ID for the discussion: "))
            # Run the main function
            asyncio.run(bot.main(inst))
            time.sleep(60)
            round_cnt -= 1
        elif mode == 2:
            asyncio.run(bot.run_bot())
