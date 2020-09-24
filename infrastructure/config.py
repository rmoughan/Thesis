
class Config:

    def __init__(self, file_path):
        self.params = {}
        self.process_file(file_path)

    def process_file(self, file_path):
        file = open(file_path, "r")
        for line in file:
            key, value = line.rstrip().split(":")
            try:
                self.params[key] = int(value)
            except ValueError as e:
                if "Schedule" in value:
                    self.params[key] = value.strip()
                elif '[' in value:
                    self.params[key] = [int(e) for e in value.strip()[1:-1].split(',')]
                elif '0.' in value:
                    self.params[key] = float(value)
                else:
                    self.params[key] = value.strip()
