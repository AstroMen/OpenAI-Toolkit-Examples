import subprocess
import os, re, logging
import openai
import pandas as pd
import argparse


logging.basicConfig(
    level=logging.INFO,
    format='\033[93m%(levelname)s [%(filename)s Line: %(lineno)d] %(asctime)s\033[0m %(message)s'
)


class SystemUtil:
    @staticmethod
    def exec_shell(command, shell=True):
        if not shell:
            command = command.split(' ')
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=shell)

        if result.returncode == 0:
            logging.info(f"Success: {result.stdout}")
            return result.stdout
        else:
            logging.error(f"Fail: {result.stderr}")
            return result.stderr

    @staticmethod
    def extract_id(text):
        pattern = r'ft-\w+'
        match = re.search(pattern, text)
        return match.group(0) if match else None


class DataSetPreparation:
    def prepare_dataset(self, dir_data, data_file, task_name):
        df = pd.read_csv(data_file, sep=',')
        column_names = df.columns.tolist()
        df.rename(columns={column_names[0]: 'prompt', column_names[1]: 'completion'}, inplace=True)
        logging.info(f'Dataset: \n{df.head()}')
        df.to_json(f"{dir_data}{task_name}.jsonl", orient='records', lines=True)
        cmd = f'openai tools fine_tunes.prepare_data -f {dir_data}{task_name}.jsonl -q'
        res = SystemUtil.exec_shell(cmd)
        pattern = r'\`\.(.*?\.jsonl)\`'
        file_paths = re.findall(pattern, res)
        logging.info(f'Prepared files: {file_paths}')
        return file_paths


class ModelFineTuning:
    def __init__(self, dir_finetune, task_name):
        self.dir_finetune = dir_finetune
        self.model_name = 'ada'
        self.task_name = task_name
        self.fine_tuning_id = None

    def fine_tune_model(self, prepared_file_paths, verified=False, positive=None):
        verified_cmd = f'-v "{self.dir_finetune}{self.task_name}_prepared_valid.jsonl" --compute_classification_metrics --classification_positive_class "{positive}"'
        # verified_cmd = verified_cmd if verified else ""
        cmd = f'openai api fine_tunes.create -t ".{prepared_file_paths[0]}" {verified_cmd if verified else ""} -m {self.model_name}'
        # cmd = f'openai api fine_tunes.create -t "{self.dir_finetune}{self.task_name}_prepared_train.jsonl" -v "{self.dir_finetune}{self.task_name}_prepared_valid.jsonl" --compute_classification_metrics --classification_positive_class "{self.task_name}" -m {self.model_name}'
        res = SystemUtil.exec_shell(cmd)
        self.fine_tuning_id = SystemUtil.extract_id(res)
        logging.info(f'Fine tuning id: {self.fine_tuning_id}')
        return self.fine_tuning_id if self.fine_tuning_id else None

    def check_status(self):
        cmd = f'openai api fine_tunes.follow -i {self.fine_tuning_id}'
        log_data = SystemUtil.exec_shell(cmd)
        logging.info(f'Status: \n{log_data}')


class ModelEvaluation:
    @staticmethod
    def get_predictions(fine_tuning_id, res_path):
        cmd = f'openai api fine_tunes.results -i {fine_tuning_id} > {res_path}result_{fine_tuning_id}.csv'
        res = SystemUtil.exec_shell(cmd)
        logging.info(f'get_predictions result: \n{res}')

    @staticmethod
    def evaluate_model():
        cmd = 'openai api fine_tunes.list'
        res = SystemUtil.exec_shell(cmd)
        logging.info(f'Evaluate result: \n{res}')


class Inference:
    @staticmethod
    def run_inference(prompt_text, ft_model):
        try:
            res = openai.Completion.create(model=ft_model, prompt=prompt_text + '\n\n###\n\n', max_tokens=1,
                                           temperature=0)
            return res['choices'][0]['text']
        except Exception as e:
            logging.error(f'An error occurred: {e}')
            return 'None'


def main(task_name, preprocess_file, finetune=False):
    if os.getenv('OPENAI_API_KEY') is None:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    logging.info('OPENAI_API_KEY: ' + os.getenv('OPENAI_API_KEY'))

    dir_src_data = './data/source/'
    dir_finetune = './data/finetune/' + task_name + '/'
    if not os.path.exists(dir_finetune):
        os.makedirs(dir_finetune)
    data_prep = DataSetPreparation()
    model_finetuning = ModelFineTuning(dir_finetune, task_name)
    model_evaluation = ModelEvaluation()

    if finetune:
        # Preparing the dataset
        prepared_file_paths = data_prep.prepare_dataset(dir_finetune, os.path.join(dir_src_data, preprocess_file), task_name)
        # Model Fine-Tuning
        ft_model_id = model_finetuning.fine_tune_model(prepared_file_paths, verified=True, positive='1')
    else:
        ft_model_id = input('Please input finetune id: ')

    # Check status
    model_finetuning.check_status()
    # Get predictions
    model_evaluation.get_predictions(ft_model_id, dir_finetune)
    # Model Evaluation
    # Model Accuracy during Training
    model_evaluation.evaluate_model()

    # To run inference using our fine-tuned model
    while True:
        # prompt_text = '6 Triggers People With Zero Patience Will Recognize'
        prompt_text = input("Please input the prompt('exit' to exit): ")
        if prompt_text.lower() == 'exit':
            break
        result_text = Inference.run_inference(prompt_text, ft_model_id)
        logging.info("Inference Result: " + result_text)


class FineTuneCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="OpenAI Fine-Tuning Tool",
                                               formatter_class=argparse.RawTextHelpFormatter)
        self._add_arguments()

    def _add_arguments(self):
        self.parser.add_argument('-n', '--task_name', type=str, required=True,
                                 help="Name of the task to be fine-tuned.\n"
                                      "Example: python auto_finetune.py -n TaskName --preprocess_file data.csv")

        self.parser.add_argument('-p', '--preprocess_file', type=str, required=True,
                                 help="Path to the preprocessed data file.\n"
                                      "Example: python auto_finetune.py --task_name TaskName -p data.csv")

        self.parser.add_argument('-f', '--need_finetune', action='store_true',
                                 help="Specify if finetuning is needed (default: False).\n"
                                      "Example: python auto_finetune.py -n TaskName -p data.csv -f")

    def parse_args(self):
        return self.parser.parse_args()


if __name__ == '__main__':
    cli = FineTuneCLI()
    args = cli.parse_args()

    main(task_name=args.task_name, preprocess_file=args.preprocess_file, finetune=args.need_finetune)
