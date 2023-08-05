import logging
import tiktoken


class TokenManager:
    tokens_limit = 6000
    # def __init__(self, max_tokens):
    #     self.tokens_limit = 6000
        # self.max_tokens = max_tokens
        # self.messages = []

    @staticmethod
    def get_num_tokens_from_msgs_by_scratch(messages, model):
        # Define the function num_tokens_from_messages, which returns the number of tokens used by a set of messages.
        # Attempt to get the model's encoding
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # If the model is not found, use cl100k_base encoding and give a warning
            logging.warning("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        # Set the number of tokens for different models
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # Each message follows {role/name}\n{content}\n format
            tokens_per_name = -1  # If there is a name, the role will be omitted
        elif "gpt-3.5-turbo" in model:
            # For gpt-3.5-turbo model there may be updates, here return the number of tokens assumed to be gpt-3.5-turbo-0613 and give a warning
            logging.warning("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return TokenManager.get_num_tokens_from_msgs_by_scratch(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            # For gpt-4 model there may be updates, here return the number of tokens assumed to be gpt-4-0613 and give a warning
            logging.warning("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return TokenManager.get_num_tokens_from_msgs_by_scratch(messages, model="gpt-4-0613")
        elif model in {"davinci", "curie", "babbage", "ada"}:
            logging.warning("Warning: gpt-3 related model is used. Returning num tokens assuming gpt2.")
            encoding = tiktoken.get_encoding("gpt2")
            num_tokens = 0
            # Only calculate the content
            for message in messages:
                for key, value in message.items():
                    if key == "content":
                        num_tokens += len(encoding.encode(value))
            return num_tokens
        else:
            # For models that are not implemented, throw a not implemented error
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        # Calculate the number of tokens for each message
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # Each response begins with the assistant
        return num_tokens

    @staticmethod
    def remove_earliest_messages_if_needed(msg, total_token, token):
        """Remove the earliest two messages from msg1 and msg2 if num_token is greater than a threshold."""
        try:
            if total_token[-1] > TokenManager.tokens_limit:
                logging.info('Removing tokens...')
                logging.info(f"Removing tokens: total_token={total_token}, token={token}")
                if len(msg) >= 5:
                    msg.pop(2)
                    msg.pop(2)
                if len(total_token) >= 3:
                    total_token[-1] -= total_token.pop(1)
                    total_token[-1] -= total_token.pop(1)
                if len(token) >= 5:
                    token[-1] -= token.pop(3)
                    token[-1] -= token.pop(3)
                logging.info(f"Removed tokens: total_token={total_token}, token={token}")
        except Exception as e:
            logging.error(f'An error occurred: {e}')
            logging.info(f'msg={msg}')
            logging.info(f'total_token={total_token}, token={token}')
        return msg, total_token, token
