import os, sys, yaml, time, argparse
import numpy as np
import pickle, cv2, re
import decord as de
import tensorflow as tf
from datetime import timedelta
from dataflow import RNGDataFlow, MultiProcessRunnerZMQ, BatchData
from transformers import T5Tokenizer
from slowfast import SlowFast

pkl_to_native = [
                
                [r"^fast_res([0-99]+).([0-99]+).(.*)", r"SlowFast/Fast/Res_Block_\1/Res_Layer_\2/\3"],
                [r"^slow_res([0-99]+).([0-99]+).(.*)", r"SlowFast/Slow/Res_Block_\1/Res_Layer_\2/\3"],
                
                [r"^(.*)fast_conv1(.*)", r"\1SlowFast/Fast/Pre_Block/Conv_p\2"],
                [r"^(.*)fast_bn1(.*)", r"\1SlowFast/Fast/Pre_Block/BatchNorm_p\2"],
                [r"^(.*)slow_conv1(.*)", r"\1SlowFast/Slow/Pre_Block/Conv_p\2"],
                [r"^(.*)slow_bn1(.*)", r"\1SlowFast/Slow/Pre_Block/BatchNorm_p\2"],

                [r"^(.*)lateral_p1.0(.*)", r"\1SlowFast/Slow/Fuse/Fuse_pool1/Conv_Fuse\2"],
                [r"^(.*)lateral_p1.1(.*)", r"\1SlowFast/Slow/Fuse/Fuse_pool1/BatchNorm_Fuse\2"],
                [r"^(.*)lateral_res([0-99]+).0(.*)", r"\1SlowFast/Slow/Fuse/Fuse_res\2/Conv_Fuse\3"],
                [r"^(.*)lateral_res([0-99]+).1(.*)", r"\1SlowFast/Slow/Fuse/Fuse_res\2/BatchNorm_Fuse\3"],

                [r"^(.*)conv1(.*)", r"\1Conv_a\2"],
                [r"^(.*)conv2(.*)", r"\1Conv_b\2"],
                [r"^(.*)conv3(.*)", r"\1Conv_c\2"],
                [r"^(.*)bn1(.*)", r"\1BatchNorm_a\2"],
                [r"^(.*)bn2(.*)", r"\1BatchNorm_b\2"],
                [r"^(.*)bn3(.*)", r"\1BatchNorm_c\2"],
                [r"^(.*)downsample.0(.*)", r"\1Shortcut/Conv_s\2"],
                [r"^(.*)downsample.1(.*)", r"\1Shortcut/BatchNorm_s\2"],
                [r"^(.*).weight(.*)", r"\1/conv_3d/kernel:0\2"],
                [r"^(.*).gamma(.*)", r"\1/gamma:0\2"],
                [r"^(.*).beta(.*)", r"\1/beta:0\2"],
                [r"^(.*).running_mean(.*)", r"\1/moving_mean:0\2"],
                [r"^(.*).running_var(.*)", r"\1/moving_variance:0\2"],

                ]

def convert_pkl_name_to_tf(pkl_name):
    for source, dest in pkl_to_native:
        pkl_name = re.sub(source, dest, pkl_name)
    return pkl_name
    
def assign_weight_from_pkl(layer, pkl_handle):
    pkl_to_tf_dict = {}
    for pkl_name in pkl_handle:
        tf_name = convert_pkl_name_to_tf(pkl_name)
        pkl_to_tf_dict[tf_name] = {'name': pkl_name, 'weight': pkl_handle[pkl_name].asnumpy()}
    
    skipped = []
    for n, tf_w in enumerate(layer.weights):
        tf_name = tf_w.name
        if tf_name in pkl_to_tf_dict:
            pkl_w = pkl_to_tf_dict[tf_name]['weight']
            if len(pkl_w.shape)==5:
                pkl_w = np.transpose(pkl_w, [2,3,4,1,0])
            tf_w.assign(pkl_w)
        else:
            skipped.append(tf_name)
            
    return skipped

BACKBONES = [
         'slowfast',
         ]
         
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

class VideoBatchGen(RNGDataFlow):
    def __init__(self,
                 vid_paths,
                 frame_size,
                 window_size,
                 max_bag_len=32,
                 **kwargs):

        print('Initializing generator...')
        os.system('export FFREPORT=level=quiet')
        self.files_list = vid_paths
        self.frame_size = frame_size
        self.window_size = window_size
        self.max_bag_len = max_bag_len
        self._init_stats()
        print('Initialization done.')
    
    def _init_stats(self):
        self.nbBatches = 0
        correct_files = []
        for file in self.files_list:
            video = cv2.VideoCapture(file)
            nb_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
            if nb_frames > 0:
                correct_files.append(file)
                self.nbBatches += int(nb_frames/self.window_size)

        self.bag_files = [correct_files[i:i + self.max_bag_len] for i in range(0, len(correct_files), self.max_bag_len)]

    def __len__(self):
        return self.nbBatches

    def __iter__(self):
        for bag in self.bag_files:
            vl_batches = de.VideoLoader(bag, 
                                        ctx=[de.cpu(0)], 
                                        shape=(self.window_size, self.frame_size, self.frame_size, 3), 
                                        interval=0, 
                                        skip=0, 
                                        shuffle=0)

            vl_batches.reset()
            frames = []
            for n in range(len(vl_batches)):
                de.bridge.set_bridge('native')
                vl_batch = vl_batches.next()
                file_ids = vl_batch[1].asnumpy()[:,0]
                frame_ids = vl_batch[1].asnumpy()[:,1]
                #make sure all frames in a batch come from same video
                #make sure frame_ids strictly increase
                file_id = set(file_ids)
                is_strict_increase = all(i < j for i, j in zip(frame_ids, frame_ids[1:]))
                if len(file_id) != 1 or not is_strict_increase:
                    raise NotImplementedError
                file_id = file_id.pop()
                frame_rng = '{}:{}'.format(frame_ids[0], frame_ids[-1])
                file = bag[file_id]
                label = '|'.join([file, frame_rng])
                batch_frames = vl_batch[0].asnumpy()

                yield batch_frames, label

class Evaluator():
    def __init__(self,
                 config,
                 BatchGen=VideoBatchGen,
                 **kwargs):

        print('Initializing evaluator...')
        self.config = config

        # get test/data parameters
        self.gpu_workers = ','.join([str(n) for n in range(self.config['FEATURE_EXTRACTION']['GPU_WORKERS'])])
        self.train_ckpt_path = self.config['FEATURE_EXTRACTION']['TRAIN_CKPT_PATH']
        #self.split = self.config['SPLIT']
        assert self.train_ckpt_path is not None
        self.from_mxnet_ckpt = 'pkl' in self.train_ckpt_path
        self.from_tf_ckpt = not self.from_mxnet_ckpt
        self.in_path = self.config['FEATURE_EXTRACTION']['DATA']['IN_PATH']
        self.out_path = self.config['FEATURE_EXTRACTION']['DATA']['OUT_PATH']
        self.window_size = self.config['FEATURE_EXTRACTION']['DATA']['WINDOW_SIZE']
        self.frame_size = self.config['FEATURE_EXTRACTION']['DATA']['FRAME_SIZE']
        self.batch_size = self.config['FEATURE_EXTRACTION']['DATA']['BATCH_SIZE']
        self.vid_paths, self.out_dirs = list_files_dirs(self.in_path, self.out_path, None)
        

        #config gpu workers
        gpu_ids = self.gpu_workers
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        #nvidia_smi.nvmlInit()
        
        tf.get_logger().setLevel('ERROR')
        tf.debugging.set_log_device_placement(False)

        self.gpus = tf.config.list_physical_devices('GPU')
        if False:
            pass
        else:
            for gpu in self.gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        self.strategy = tf.distribute.MirroredStrategy()
        self.num_repl = self.strategy.num_replicas_in_sync
        print('Successfully allocated {} workers.'.format(self.num_repl))

        # Define batch generator
        self.df = BatchGen(vid_paths = self.vid_paths,
                           frame_size=self.frame_size,
                           window_size=self.window_size)

        #create dataflow for faster batchgen
        self.gen = BatchData(self.df, self.batch_size)
        self.gen.reset_state()
        self.steps_max = len(self.gen)
        
        def tuple_gen():
            for d in self.gen:
                yield tuple(d)

        self.vid_shape = [self.batch_size,self.window_size,self.frame_size,self.frame_size,3]
        self.lbl_shape = [self.batch_size,]
        self.dataset = tf.data.Dataset.from_generator(tuple_gen,output_types=(tf.uint8,tf.string),output_shapes=(self.vid_shape,self.lbl_shape))
        self.dataset = self.dataset.prefetch(2048).repeat(2)
        self.dist_dataset = self.strategy.experimental_distribute_dataset(self.dataset)
        self.iter_dataset = iter(self.dist_dataset)

    def pretty_progress(self,
                        step,
                        steps_max,
                        t_step,
                        **metrics):

        eta_secs = min(2**32, int((steps_max - step) * t_step))
        progress_str = ''
        progress_str += 'Iter {}/{}'.format(step+1,steps_max)
        for metric in metrics:
            progress_str += ' - {}: {:0.4f}'.format(metric, metrics[metric])
        
        progress_str += '{:>5} //{:>5} ETA {:0>8}, {:0.2f}/step {:>10}'.format('', '', str(timedelta(seconds=eta_secs)), t_step, '')
        progress_str += '\r'
        sys.stdout.write(progress_str)
        sys.stdout.flush()

    def save_features(self, feat_dict):
        for dir_ in self.out_dirs:
            os.makedirs(dir_, exist_ok=True)
        
        for n, file in enumerate(feat_dict):
            t_step = time.time()
            out_id = os.path.splitext(file.replace(self.in_path, self.out_path))[0]
            vid_id = out_id.split('/')[-1]
            features_splits = feat_dict[file]
            features = []
            nb_t = 0

            for frame_rng in features_splits:
                feat_split = features_splits[frame_rng]
                nb_t += feat_split.shape[0]
                features.append(feat_split)

            features = np.concatenate(features, axis=0)
            np.save(out_id, features)
            t_step = time.time() - t_step
            self.pretty_progress(step=n,
                                 steps_max=len(feat_dict),
                                 t_step=t_step)


    def extract_features(self):
        @tf.function(input_signature=[self.iter_dataset.element_spec])
        def _feat_step(dist_inputs):
            def _step_fn(inputs):
                frames, labels = inputs
                features = self.model(inputs=frames, training=tf.constant(False))
                
                return features, labels

            features_batch, labels_batch = self.strategy.run(_step_fn, args=(dist_inputs,))
            #import pdb; pdb.set_trace()
            per_replica_features = self.strategy.experimental_local_results(features_batch)
            per_replica_labels = self.strategy.experimental_local_results(labels_batch)
            features = tf.concat(per_replica_features, axis=0)
            labels = tf.concat(per_replica_labels, axis=0)

            return features, labels

        with self.strategy.scope():
            feat_dict = {}
            print('Extracting visual features...')
            for step in range(self.steps_max):
                t_step = time.time()

                inputs = self.iter_dataset.next()
                features, labels = _feat_step(inputs)
                features, labels = features.numpy(), labels.numpy()
                
                for n, label in enumerate(labels):
                    label = label.decode('ascii')
                    info = label.split('|')
                    assert len(info) == 2, "Corrupt label: {}".format(label)
                    file, frame_rng = info
                    if file not in feat_dict:
                        feat_dict[file] = {}
                    feat_dict[file][frame_rng] = features[n,:]

                t_step = time.time() - t_step
                self.pretty_progress(step=step,
                                     steps_max=self.steps_max,
                                     t_step=t_step)
            
            print('\nFeature extraction finished.\nSaving features...')
            self.save_features(feat_dict)
            print('\n\nAll done.')


class SlowFast_Evaluator(Evaluator):
    def __init__(self,
                 config,
                 **kwargs):
        super(SlowFast_Evaluator, self).__init__(config=config, **kwargs)
        self.slow_config = self.config['SLOW_CONFIG']
        self.fast_config = self.config['FAST_CONFIG']
        self.slow_fast_params = self.config['SLOW_FAST_PARAMS']
        with self.strategy.scope():
            print('Building video model...')
            self.model = SlowFast(num_classes=None,
                                  slow_config=self.slow_config,
                                  fast_config=self.fast_config,
                                  temporal_sampling_rate=self.slow_fast_params['TMP_SMPL_RATE'],
                                  spatial_sampling_rate=self.slow_fast_params['SPT_SMPL_RATE'],
                                  fusion_kernel=self.slow_fast_params['FUSION_KERNEL'],
                                  pre_activation=self.slow_fast_params['PRE_ACTIVATION'],
                                  alpha=self.slow_fast_params['ALPHA'],
                                  beta=self.slow_fast_params['BETA'],
                                  tau=self.slow_fast_params['TAU'],
                                  dropout_rate=self.slow_fast_params['DROPOUT_RATE'],
                                  epsilon=self.slow_fast_params['EPSILON'],
                                  momentum=self.slow_fast_params['MOMENTUM'],
                                  data_format=self.slow_fast_params['DATA_FORMAT'])
            
            self.checkpoint = tf.train.Checkpoint(model=self.model)

            print('Initializing variables...')
            self.model.init([1]+self.vid_shape[1:])

            if self.from_tf_ckpt:
                    print('Loading variables from TF checkpoint...')
                    self.checkpoint.restore(self.train_ckpt_path)#.assert_consumed()
                
            elif self.from_mxnet_ckpt:
                print('Loading variables from MXNET checkpoint...')
                pkl_handle = pickle.load(open(self.train_ckpt_path, 'rb'))
                skipped = assign_weight_from_pkl(self.model, pkl_handle)
                print('{} weights were skipped:\n{}'.format(len(skipped), skipped))

            print('All initializations done.')

def main(backbone, config):
    assert backbone in BACKBONES
    if backbone == 'slowfast':
        evaluator = SlowFast_Evaluator(config = config)
    
    evaluator.extract_features()

if __name__ == '__main__':
    description = 'The main test call function.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--backbone', type=str, default='slowfast')
    parser.add_argument('--config_path', type=str, default='./configs/SlowFast_R50_8x8.yml')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    backbone = args.backbone
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    optional_args = args.opts
    for arg in optional_args:
        key, value = arg.split(':')
        replace_value(config, key, value)

    main(backbone, config)