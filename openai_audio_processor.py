import os
import openai
import logging


class OpenaiAudioProcessor:
    def __init__(self, api_key, system_prompt=''):
        self.system_prompt = system_prompt
        openai.api_key = api_key

    def transcribe(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text = transcript['text']
            logging.info(f'transcript: \n{text}')
            return text

    def translate(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.translate("whisper-1", audio_file)
            text = transcript['text']
            logging.info(f'translate: \n{text}')
            return text

    def correct_transcript(self, temperature, audio_file):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": self.translate(audio_file)
                }
            ]
        )
        return response['choices'][0]['message']['content']


system_prompt = '''You are a helpful assistant. Your task is to correct any spelling discrepancies in the transcribed text. 
Make sure that the names of the following products are spelled correctly. 
Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided.
'''

api_key = os.getenv('OPENAI_API_KEY')
processor = OpenaiAudioProcessor(api_key, system_prompt)

file_path = './audio/qilixiang.mp3'
processor.transcribe(file_path)
corrected_text = processor.correct_transcript(0, file_path)
print(f'corrected_text: \n{corrected_text}')
