
class Settings:
    def __init__(self, setting_file):
        file = open(setting_file, 'r')
        lines = file.readlines()
        self.stages = []
        self.input_files = dict()
        self.output_files = dict()
        self.flags = []
        self.sizes = dict()
        for line in lines:
            words = line.split()
            line_key = words[0].strip()
            if line_key == 'frames':
                self.frames = list(range(int(words[1].strip()), int(words[2].strip())))
            elif line_key == 'flag':
                self.flags.append(words[1].strip())
            elif line_key == 'sizes':
                self.sizes[words[1].strip()] = int(words[2].strip())
                self.sizes[words[3].strip()] = int(words[4].strip())
                self.sizes[words[5].strip()] = int(words[6].strip())
            elif line_key == 'vis':
                self.stages.append('vis')
                self.input_files['vis_net'] = words[1].strip()
                self.input_files['vis_set'] = words[2].strip()
                self.input_files['vis_seg'] = words[3].strip()
            else:
                self.stages.append(line_key)
                self.input_files[line_key] = words[1].strip()
                self.output_files[line_key] = words[2].strip()


# EXAMPLE
"""
seg temp_outputs\clean.mat temp_outputs\seg.npz
set temp_outputs\seg.npz temp_outputs\set.npz
net temp_outputs\set.npz temp_outputs\net.pt
res temp_outputs\net.pt temp_outputs\res.npz
frames 27 53
sizes train 50 valid 10 test 1
flag cv
flag down_sample
"""