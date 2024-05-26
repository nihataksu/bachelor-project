class AppendLogger:
    def __init__(self, file_name):
        self.file_name = file_name

    def print(self, text):
        print(text)

        with open(self.file_name, "a") as log_file:
            log_file.write(text + "\n")
