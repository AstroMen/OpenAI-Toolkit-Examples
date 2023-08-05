import openai

class ChatBot:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        self.add_system_message()

    def add_system_message(self):
        if self.system:
            self.messages.append({"role": "system", "content": self.system})

    def __call__(self, message):
        self.add_user_message(message)
        response = self.execute()
        self.add_bot_message(response)
        return response

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def execute(self):
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages)
        print(completion.usage)  # {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        return completion.choices[0].message.content


ai = ChatBot("You are a chatbot imitating Story Writer.")
ai("Tell me a story")
reply = ai.execute()
