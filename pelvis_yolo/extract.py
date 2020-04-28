import os
import numpy as np

from results_analyzer import save_annotation, load_annotation


DATA_PATH = os.path.join('pelvis_scan','EXTRACT_TEST','charleroi_1')
GT_FILE = 'boxes.p'
PRED_FILE = 'pred_boxes.p'
ORGANS = ['bladder','rectum','prostate']
def _main():
    gt = load_annotation(os.path.join(DATA_PATH,GT_FILE))
    pred = load_annotation(os.path.join(DATA_PATH,PRED_FILE))
    convert(pred,ORGANS)


def outliers(dic,organs):
    '''Identifies potential outliers in the predicted data.'''

def convert(dic,organs):
    '''Takes a dic that has bb in terms of files and returns a dic that has the organs as keys and all the bb as items.'''
    new_dic = {}
    for o in organs:
        new_dic.update({o:[]})

    for file,box in dic.items():
        for organ,bbs in box.items():
            for box in bbs:
                new_dic[organ].append(box)

    return new_dic


if __name__ == '__main__':
    _main()
