class Log:
    def __init__(self, filename_track, filename_general):
        self.filename_track = filename_track
        self.filename_general = filename_general
    def log_track(self, message):
        with open(self.filename_track, 'a') as f:
            f.write(message + '\n')
    def log_general(self, message):
        with open(self.filename_general, 'a') as f:
            f.write(message + '\n')