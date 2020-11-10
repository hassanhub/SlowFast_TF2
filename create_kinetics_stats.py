import cv2
import numpy as np
import os, pickle
import argparse

def _list_files(data_path,extension):
    l=[]
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.isfile(os.path.join(root, file)) and extension in file:
                l.append(os.path.join(root, file))
    return l


def main(vid_path, annot_path):
    files = _list_files(vid_path, '.mp4')
    annotations = {'classes': {}, 'stats': dict(zip(files, [{}]*len(files)))}
    classes = set()
    faulty = []
    for n,file in enumerate(files):
        print('{}/{}: {}'.format(n+1,len(files),file))
        clss = file.split('/')[-2]
        classes.add(clss)
        video = cv2.VideoCapture(file)
        nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(np.round(video.get(cv2.CAP_PROP_FPS)))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if nb_frames==0:
            faulty.append(file)
            continue
        annotations['stats'][file] = {'class': clss,
                                      'nb_frames': nb_frames,
                                      'frame_rate': frame_rate,
                                      'width': width,
                                      'height': height}
        
    class_indices = {}
    cnt = -1
    for key in sorted(classes):
        cnt+=1
        class_indices[key] = cnt
    annotations['classes'] = class_indices
    pickle.dump(annotations, open(os.path.join(annot_path,'annotations.pkl'), 'wb'), protocol=4)
    if faulty:
        pickle.dump(faulty, open(os.path.join(annot_path,'faulties.pkl'), 'wb'), protocol=4)
    print('Done. {} corrupt files in total.'.format(len(faulty)))
    return

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Script for reading stats of video dataset.')
    p.add_argument('-i', '--vid-path', type=str,
                   help=('Path to raw videos (split by class labels).'))
    p.add_argument('-o', '--annot-path', type=str,
                   help='Output annotation dictionary.')

    main(**vars(p.parse_args()))