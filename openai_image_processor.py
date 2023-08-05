import os
import re
import openai
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import logging


class OpenaiImageProcessor:
    def __init__(self):
        self.dir_image_data = './data/image/'
        if not os.path.exists(self.dir_image_data):
            os.makedirs(self.dir_image_data)

    async def dalle_gen_img(self, prompt, style='cartoon', img_size='512x512'):
        response = openai.Image.create(
            model="image-alpha-001",
            prompt=prompt + ', ' + style,
            n=1,  # Number of images to generate
            size=img_size,  # Output image size
            response_format="url"
        )
        # Get image URL
        image_url = response['data'][0]['url']
        return image_url

    async def generate_story_images(self, src_file):
        with open(src_file, 'r') as file:
            data_list = file.read().strip().split('\n')
        data = pd.DataFrame(data_list, columns=['prompt'])
        title = re.sub(r'^Title: ', '', data.iloc[1]['prompt'])
        logging.info(f'title: {title}')
        story_path = os.path.join(self.dir_image_data, title.replace(' ', '_') + '/')
        logging.info(f'story_path: {story_path}')
        if not os.path.exists(story_path):
            os.makedirs(story_path)

        # generate images based on text prompt from csv
        for index, row in data.iloc[2:].iterrows():
            prompt = re.sub(r'^\d+\.\s', '', row['prompt'])
            plot_id = index - 1
            image_url = await self.dalle_gen_img(prompt)

            # save images
            image_response = requests.get(image_url)
            img = Image.open(BytesIO(image_response.content))
            img.save(f"{story_path}generated_image_{plot_id}.png")
            logging.info(f"Generating image {plot_id}: {prompt}")
