
# OpenAI Toolkit Examples

This repository contains a comprehensive toolkit for OpenAI processing with application examples, including text, image, audio processing, automatic fine-tuning, and some utility tools.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Clone this repository to your local machine and navigate to the project directory:

```bash
git clone https://github.com/AstroMen/OpenAI-Toolkit-Examples.git
cd OpenAI-Toolkit-Examples
```

Ensure you have the necessary Python packages installed:

```bash
pip install -r requirements.txt
```

## Usage

This toolkit contains several modules. Here's how to use them:

* `main.py`: This is the main entry point of the application. Run this script to start the program with the command `main.py -t 1 -m "gpt-4" -r 50`. The arguments are:

  - `-t`: Type of mode to run: 1 for AI pair, 2 for stand-alone. Default is 1.
  - `-m`: Model name. Options include 'gpt-4', 'gpt-4-0314', 'gpt-4-0613'. Default is 'gpt-4'.
  - `-r`: Number of rounds to run. Default is 50.

  If the mode is 1, you will be asked to input the task ID for the discussion: 1 for interview, 2 for economic trends, 3 for entrepreneurship.

* `token_manager.py`: This module is responsible for managing tokens for text processing. It automatically removes old messages to ensure the number of tokens does not exceed the limit.

* `auto_finetune.py`: Use this module for automatic fine-tuning of the OpenAI models. Run it with the command `python3 auto_finetune.py -n mytest -p test.csv -f`. The arguments are:

  - `-n`: Name of the task to be fine-tuned.
  - `-p`: Path to the preprocessed data file.
  - `-f`: If specified, finetuning is needed. Default is False.

After a successful fine-tuning, the status will be printed to the console, showing the fine-tuning process and the result. For example:

  ```bash
  openai api fine_tunes.follow -i  <finetune_id>
  [2023-07-31 13:33:19] Created fine-tune: <finetune_id>
  ...
  [2023-07-31 16:28:00] Fine-tune succeeded
  Job complete! Status: succeeded 🎉
  Try out your fine-tuned model:
  openai api completions.create -m ada:ft-myft-2023-07-31-23-27-59 -p <YOUR_PROMPT>
  ```

* `chat_bot.py`: This module is a chatbot that uses OpenAI's GPT models for generating responses. Initialize and call it as follows:

  ```python
  ai = ChatBot("You are a chatbot imitating Story Writer.")
  ai("Tell me a story")
  reply = ai.execute()
  ```

* `openai_image_processor.py`: This module processes images using OpenAI's models. It includes a function `generate_story_images` that generates story images.

* `openai_audio_processor.py`: This module processes audio using OpenAI's models. Initialize and call it as follows:

  ```python
  system_prompt = '''You are a helpful assistant. Your task is to correct any spelling discrepancies in the transcribed text. 
  Make sure that the names of the following products are spelled correctly. 
  Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided.
  '''

  api_key = os.getenv('OPENAI_API_KEY')
  processor = OpenaiAudioProcessor(api_key, system_prompt)

  file_path = './audio/my_video.mp3'
  processor.transcribe(file_path)
  corrected_text = processor.correct_transcript(0, file_path)
  print(f'corrected_text: {corrected_text}')
  ```

* `prompt_preprocessor.py`: Use this module for preprocessing prompts for text generation. The `image_prompts.csv` file should have the following format: the first line is the prompt, the second line starts with "Title: " and contains the story title, and the subsequent lines contain the story plot with numbers.

* `openai_msg_handler.py`: This module handles messages in the context of an OpenAI chat.

* `prompt_preprocessor.py`: This module is used for preprocessing prompts for text generation. It provides different prompt examples using both zero-shot and few-shot learning techniques. The module can also simulate dialogues with different AI roles interacting with each other. The `image_prompts.csv` file, which is used by this module, should have the following format: the first line is the prompt, the second line starts with "Title: " and contains the story title, and the subsequent lines contain the story plot with numbers.

* `utils/`: This directory contains utility functions and classes that support the other modules.

* `data/`: This directory is used to store datasets and other relevant data files required for the functioning of the various modules in this toolkit.

For detailed usage of each module, please refer to the comments and documentation in the respective Python files.

## Contributing

Contributions are welcome! Please read the contributing guidelines to get started.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
