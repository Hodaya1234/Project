
class Settings:
    def __init__(self, setting_file):
        file = open(setting_file, 'r')
        lines = file.readlines()
        self.stages = lines[0].split()[1:]


s = Settings('temp_outputs/settings_format')