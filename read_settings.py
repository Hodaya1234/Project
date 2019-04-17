
class Settings:
    def __init__(self, setting_file):
        file = open(setting_file, 'r')
        lines = file.readlines()
        self.stages = []
        self.files = dict()
        self.flags = []
        self.sizes = dict()
        self.stages = [w.strip() for w in lines[0].split()]
        for line in lines[1:]:
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
            elif line_key == 'frame_groups':
                frame_groups = []
                first_frame = self.frames[0]
                prev_num = first_frame
                numbers = [int(n.strip()) for n in words[1:]]
                numbers.append(self.frames[-1] + 1) # make the last frame included
                for num in numbers:
                    frame_groups.append(list(range(prev_num - first_frame, num - first_frame)))
                    prev_num = num
                self.frame_groups = frame_groups
                self.frame_groups_string = ['frames {}-{}'.format(g[0] + first_frame, g[-1] + first_frame) for g in frame_groups]

            else:
                self.files[line_key] = words[1].strip()


# EXAMPLE
"""
mask seg set net los calc_vis show_vis
mask temp_outputs/0212-a/mask.npy
seg temp_outputs/0212-a/seg.npz
set temp_outputs/0212-a/set.npz
net temp_outputs/0212-a/net.pt
los temp_outputs/0212-a/los.npz
vis_both temp_outputs/0212-a/vis_both.npy
vis_seg temp_outputs/0212-a/vis_seg.npy
vis_frame temp_outputs/0212-a/vis_frame.npy
frames 27 50
sizes train 50 valid 10 test 1
flag cv
frame_groups 33 41

"""
