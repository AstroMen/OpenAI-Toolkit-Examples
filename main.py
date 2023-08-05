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
        self.api_key = os.getenv('OPENAI_API_KEY')
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
        self.api_key = os.getenv('OPENAI_API_KEY')
        if self.api_key is None:
            raise Exception("Please set the 'OPENAI_KEY' environment variable.")
        openai.api_key = self.api_key
    
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
                self.token1.append(self.token1[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg1_handler.get_msg_list()[-1]], model=self.model_name))
                msg2_handler.generate_and_append_message("user", ans1)
                self.token2.append(self.token2[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg2_handler.get_msg_list()[-1]], model=self.model_name))
                executed = False

                # calculate token
                if self.total_token1[0] == 0:
                    self.total_token1[0] = usage['prompt_tokens']
                self.total_token1.append(usage['total_tokens'])
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
        self.token1 = [TokenManager.get_num_tokens_from_msgs_by_scratch(msg1_handler.get_msg_list(), model=self.model_name)]
        self.token2 = [TokenManager.get_num_tokens_from_msgs_by_scratch(msg2_handler.get_msg_list(), model=self.model_name)]

        msg1_handler.generate_and_append_message("assistant", greetings)
        msg2_handler.generate_and_append_message("user", greetings)
        self.token1.append(self.token1[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg1_handler.get_msg_list()[-1]], model=self.model_name))
        self.token2.append(self.token2[-1] + TokenManager.get_num_tokens_from_msgs_by_scratch([msg2_handler.get_msg_list()[-1]], model=self.model_name))

        msg1_handler.generate_and_append_message("user", greetings_resp)
        msg2_handler.generate_and_append_message("assistant", greetings_resp)
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
        msg_handler.generate_and_append_message("system", sys_content)
        msg_handler.generate_and_append_message("user", greetings)

        executed = True
        ans = ''
        while executed:
            ans, num_token, status = await msg_handler.get_ans(self.max_tokens)
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
            inst = int(input("Please input the task ID for the discussion: "))
            # Run the main function
            asyncio.run(bot.main(inst))
            time.sleep(60)
            round_cnt -= 1
        elif mode == 2:
            asyncio.run(bot.run_bot())
