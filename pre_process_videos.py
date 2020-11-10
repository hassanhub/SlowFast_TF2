import argparse
import pickle
import os
import subprocess
import uuid
from multiprocessing import Pool
import cv2

def set_none(arg_list):
    arg_out = []
    for arg in arg_list:
        if int(arg) == -1:
            arg_out.append(None)
        else:
            arg_out.append(arg)
    return arg_out

def list_files_dirs(source_root,target_root,extension):
    source_files = []
    source_dirs = []
    if extension is None:
        extensions = ['.mp4', '.webm', '.mkv', '.avi']
    else:
        extensions = [extension]
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if os.path.isfile(os.path.join(root, file)):
                for extension in extensions:
                    if extension in file:
                        source_files.append(os.path.join(root, file))
        for dir_ in dirs:
            source_dirs.append(os.path.join(root,dir_))
    target_dirs = [dir_.replace(source_root, target_root) for dir_ in source_dirs]
    return source_files, target_dirs

def crop(in_file,dim_scale):
    video = cv2.VideoCapture(in_file)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dim = min(width, height)

    if height >= width:
        x = 0
        y = int((height - out_dim) * 0.5)
    else:
        x = int((width - out_dim) * 0.5)
        y = 0

    return '{}:{}:{}:{}'.format(out_dim,out_dim,x,y)

def scale_dim_shorter(in_file,dim_scale):
    video = cv2.VideoCapture(in_file)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width<height:
        return "'trunc({}/hsub)*hsub:trunc(ih/vsub)*vsub:flags=bicubic'".format(dim_scale)
    else:
        return "'trunc(iw/hsub)*hsub:trunc({}/vsub)*vsub:flags=bicubic'".format(dim_scale)

def scale_dim(in_file,dim_scale):
    return '{}:{}'.format(dim_scale,dim_scale)

def process_clip(in_file, out_file, dim_scale, frame_rate, codec, preset, crf):
    status = False
    # Construct command line for processing the file.
    command = ['ffmpeg',
               '-i', '"%s"' % in_file,
               '-c:v', codec, '-c:a', 'copy',
               '-tune', 'fastdecode',
               '-crf', str(crf),
               '-movflags', 'faststart',
               '-preset', preset]
    
    if frame_rate is not None:
        command.extend(['-r', str(frame_rate)])

    if dim_scale is not None:
        crops = crop(in_file, dim_scale)
        scales = scale_dim(in_file, dim_scale)
        command.extend(['-filter:v', 'crop={},scale={}'.format(crops, scales)])
        
        #command.extend(['-vf', 'scale={}'.format(scales)])

    command.extend(['-loglevel', 'quiet', '"%s"' % out_file])
    command = ' '.join(command)
    
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(command)
        return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(out_file)
    return status, 'Processed'


def process_clip_wrapper(i, in_file, n_files, in_dir, out_dir, dim_scale, frame_rate, codec, preset, crf):
    """Wrapper for parallel processing purposes."""
    out_id = os.path.splitext(in_file.replace(in_dir, out_dir))[0]
    clip_id = os.path.basename(out_id)
    out_file = out_id+'.mp4'
    if os.path.exists(out_file):
        status = tuple([clip_id, True, 'Exists'])
        return status

    processed, log = process_clip(in_file, out_file, dim_scale, frame_rate, codec, preset, crf)

    status = tuple([clip_id, processed, log])
    print('{}/{}: {}'.format(i+1,n_files,status))
    return status


def main(in_dir, out_dir, dim_scale, frame_rate, codec, preset, num_jobs, extension, crf):
    dim_scale, frame_rate, extension = set_none([dim_scale, frame_rate, extension])

    source_files, target_dirs = list_files_dirs(in_dir, out_dir, extension)
    for dir_ in target_dirs:
        os.makedirs(dir_, exist_ok=True)
    n_files = len(source_files)

    if num_jobs == 1:
        status_lst = []
        for i, file in enumerate(source_files):
            status_lst.append(process_clip_wrapper(i, file, n_files, in_dir, out_dir, dim_scale, frame_rate, codec, preset, crf))
    else:
        pool = Pool(processes=num_jobs)
        processes = []
        for i, file in enumerate(source_files):
            processes.append(pool.apply_async(process_clip_wrapper, (i, file, n_files, in_dir, out_dir, dim_scale, frame_rate, codec, preset, crf)))
        status_lst = [process.get() for process in processes]

    # Save download report.
    with open(os.path.join(out_dir,'process_report.pkl'), 'wb') as fobj:
        pickle.dump(status_lst, fobj)


if __name__ == '__main__':
    description = 'Rescaling all videos under in_dir.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('-i', '--in-dir', type=str)
    p.add_argument('-o', '--out-dir', type=str)
    p.add_argument('-d', '--dim-scale', type=int, default=-1)
    p.add_argument('-f', '--frame-rate', type=int, default=-1)
    p.add_argument('-c', '--codec', type=str, default='libx265')
    p.add_argument('-p', '--preset', type=str, default='ultrafast')
    p.add_argument('-n', '--num-jobs', type=int, default=24)
    p.add_argument('-e', '--extension', type=str, default=-1)
    p.add_argument('-r', '--crf', type=int, default=23)

    main(**vars(p.parse_args()))
