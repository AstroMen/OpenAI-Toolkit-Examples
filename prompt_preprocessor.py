import os
from utils.file_ops import FileOperations


class PromptPreprocessor:
    def __init__(self, gen_file_name, gen_bk_file, role_name1, role_name2, dir_chat_data, model_name, model_id):
        self.gen_file_name = gen_file_name
        self.gen_bk_file = gen_bk_file
        self.role_name1 = role_name1
        self.role_name2 = role_name2
        self.dir_chat_data = dir_chat_data
        self.model_name = model_name
        self.model_id = model_id

    def get_params_for_interview(self):
        role_info = {
            0: {'short': 'sde', 'position': 'Software Engineer', 'question_related': 'algorithm and system design'},
            1: {'short': 'backend', 'position': 'Back-End Developer', 'question_related': ''},
            2: {'short': 'frontend', 'position': 'Front-End Developer', 'question_related': ''},
            3: {'short': 'full_stack', 'position': 'Full Stack Developer', 'question_related': ''},
            4: {'short': 'sys', 'position': 'Systems Engineer', 'question_related': ''},
            5: {'short': 'android', 'position': 'Android Developer', 'question_related': ''},
            6: {'short': 'ios', 'position': 'IOS Developer', 'question_related': ''},
            7: {'short': 'security', 'position': 'Security Engineer', 'question_related': ''},
            8: {'short': 'security_detect', 'position': 'Security Engineer - Detection and Response',
                'question_related': ''},
            9: {'short': 'devops', 'position': 'DevOps Engineer', 'question_related': 'Linux'},
            10: {'short': 'data', 'position': 'Data Engineer', 'question_related': ''},
            11: {'short': 'ml', 'position': 'Machine Learning Engineer', 'question_related': ''},
            12: {'short': 'db', 'position': 'Database Administrator', 'question_related': ''},
            13: {'short': 'test', 'position': 'Quality Assurance Engineer', 'question_related': ''},
            14: {'short': 'cloud', 'position': 'Cloud Engineer', 'question_related': ''},
            15: {'short': 'pm', 'position': 'Project Manager', 'question_related': 'agile and scrum'},
        }
        print("Available roles: ")
        for key, value in role_info.items():
            print(f"ID: {key}, Position: {value['position']}")

        inst = input("Please input the ID for the role: ")
        return role_info[int(inst)]

    def set_beginning_for_interview(self):
        params = self.get_params_for_interview()
        self.gen_file_name = f"gen_interview_{params['short']}.txt"
        self.role_name1 = 'Interviewer'
        self.role_name2 = 'Candidates'

        position, question_related = params['position'], f" focused on {params['question_related']}"

        interviewer_content = f"""As a senior interviewer for the {position} position, you are tasked with conducting a mock interview that should span approximately one hour. 
                    Your main goal is to ask a series of technical interview questions{question_related}. 
                    It is important to ensure that the number and complexity of the questions are enough to fill the entire duration.
                    Additionally, you are responsible for providing corrections, prompts, engaging in discussions, and evaluating the candidate's responses as needed.
                    Please proceed by asking one question at a time.
                    """
        candidate_content = f"""As a candidate for the position of a highly experienced {position}, you are participating in a one-hour mock interview. 
                    You are expected to respond to the interviewer's inquiries with precise and detailed technical information. 
                    If applicable, please include examples to enhance the clarity and support of your answers."
                    """
        greetings = "Let's start today's interview."
        greetings_resp = "Okay."

        with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
            FileOperations.save_and_print('****************************************************', file)
            FileOperations.save_and_print(f"* {self.model_name} *", file)
            FileOperations.save_and_print(f"* {position},{question_related} *", file)
            FileOperations.save_and_print('****************************************************', file)
        return interviewer_content, candidate_content, greetings, greetings_resp

    def set_beginning_for_trend(self):
        self.gen_file_name = 'gen_trend_economic.txt'
        self.role_name1 = 'Partner1'
        self.role_name2 = 'Partner2'

        partner1 = """Engage in a thorough and in-depth discussion with your partner to explore and analyze the economic trends that you both predict will significantly influence and mold the global landscape in the upcoming decade.
        Indicate the end of the discussion by saying 'Closing of the discussion, exit.'.
        """
        partner2 = """Engage in a thorough and in-depth discussion with your partner to explore and analyze the economic trends that you both predict will significantly influence and mold the global landscape in the upcoming decade. 
        Your discussion should include but not be limited to topics such as technological advancements, shifts in global trade, changes in population demographics, and the impact of climate change on the economy. 
        Be sure to provide supporting evidence for your predictions and consider different perspectives, including potential challenges and opportunities. 
        Additionally, discuss how these trends might affect different sectors such as technology, manufacturing, finance, and services.
        Indicate the end of the discussion by saying 'Closing of the discussion, exit.'.
        """
        greetings = "Let's start our talk."
        greetings_resp = "Okay."

        with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
            FileOperations.save_and_print('****************************************************', file)
            FileOperations.save_and_print(f"* {self.model_name} *", file)
            FileOperations.save_and_print('****************************************************', file)
        return partner1, partner2, greetings, greetings_resp

    def set_beginning_for_entrepreneurship(self, lang='en'):
        self.gen_file_name = f'gen_entrepreneurship_plan{self.model_id}.txt'
        self.role_name1 = 'Partner1'
        self.role_name2 = 'Partner2'
        if lang == 'en':
            lang = ''
        elif lang == 'zh':
            lang = 'It is important to conduct the discussion in Chinese.'

        partner1 = f"""Please allocate a continuous four-hour period with your partner to fully immerse yourselves in the realm of software industry entrepreneurship. 
        The primary aim of this session is to generate groundbreaking ideas and formulate a viable startup concept that is closely tied to any facet of software development. 
        Throughout the conversation, carefully analyze the advantages and disadvantages of the current idea and delve deep into the specifics. 
        Your ultimate objective is to create a comprehensive business plan that encompasses all pertinent aspects. 
        Once the discussion reaches its conclusion, please condense your findings into a succinct version of the business plan. 
        It is crucial to engage in a meticulous and exhaustive discussion, ensuring that no topic is left unexplored and avoiding premature conclusions. 
        Utilize the entire four-hour timeframe to maximize this opportunity for exploration and exploitation. 
        {lang}
        Signal the end of the discussion by uttering the word 'Closing of the discussion, exit.'.
        """
        partner2 = f"""Please dedicate a complete four-hour session with your partner to fully immerse yourselves in the software industry entrepreneurship. 
        The main goal of this session is to generate innovative ideas and develop a feasible startup concept that is closely related to any aspect of software development. 
        Throughout the conversation, carefully analyze the advantages and disadvantages of the current idea and delve deep into the specifics.
        Your ultimate objective is to create a comprehensive business plan that covers all relevant aspects. 
        Once the discussion is complete, please condense your findings into a concise version of the business plan. 
        It is essential to engage in a thorough and exhaustive discussion, ensuring that no topic is left unexplored and avoiding drawing premature conclusions. 
        Utilize the entire four-hour duration to maximize this opportunity for exploration and exploitation. 
        {lang}
        Indicate the end of the discussion by saying 'Closing of the discussion, exit.'.
        """
        greetings = "Let's start our talk."
        greetings_resp = "Okay."

        with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
            FileOperations.save_and_print('****************************************************', file)
            FileOperations.save_and_print(f"* {self.model_name} *", file)
            FileOperations.save_and_print('****************************************************', file)
        return partner1, partner2, greetings, greetings_resp

    def set_beginning_for_story(self):
        self.gen_file_name = f'gen_story_plots.txt'
        self.role_name1 = 'Writer'
        self.role_name2 = ''

        partner1 = f"""Please write a story for children aged 3-6, composed of exactly 50 short and simple sentences. 
        Begin by providing a title for the story, and then follow these guidelines:
Title: (Provide a creative and engaging title for the story here. The title must not contain any punctuation and should be as concise as possible.)  
1. Avoid Pronouns: Do not use pronouns like "he," "she," "it," "they," "their," etc. Refer to characters by their names or descriptions.
2. Descriptive Nouns: Use more descriptive nouns for objects and settings.
3. Character Emotions: Include the emotions of the characters.
4. Background Details: Provide detailed backgrounds, such as "on the busy city street with a little giraffe."

For example:
Title: The Little Bear's Joyful Day
1. The happy little bear found a colorful umbrella.
2. The little bear walked to the sparkling river.
3. On the busy city street, a curious little giraffe saw a fish.
4. The fish, feeling playful, smiled at the little giraffe.
5. The little giraffe felt joy and danced.
6. ...
7. ...

Continue this story or create a new one, ensuring that the entire story consists of 50 brief, clear sentences, following the guidelines above, and is suitable for children aged 3-6 to understand.
        """
        greetings = "Please create one story."
        greetings_resp = ""
        self.gen_bk_file = 'image_prompts.csv'

        with open(os.path.join(self.dir_chat_data + self.gen_file_name), 'a') as file:
            FileOperations.save_and_print('****************************************************', file)
            FileOperations.save_and_print(f"* {self.model_name} *", file)
            FileOperations.save_and_print('****************************************************', file)
        return partner1, greetings
