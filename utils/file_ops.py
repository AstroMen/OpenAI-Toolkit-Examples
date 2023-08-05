import logging


class FileOperations:
    @staticmethod
    def read_file(file_name):
        try:
            with open(file_name, 'r') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return "NotFound"
        except Exception as e:
            return str(e)

    @staticmethod
    def write_file(file_name, content):
        try:
            with open(file_name, 'w') as file:
                file.write(content)
            return "Success"
        except Exception as e:
            return str(e)

    @staticmethod
    def append_content(file_name, content):
        try:
            with open(file_name, 'a') as file:
                file.write(content)
            return "Append"
        except Exception as e:
            return str(e)

    @staticmethod
    def save_and_print(message, file):
        file.write(message + '\n')
        file.flush()
        logging.info(message)
