import os, sys, yaml, time, re
import numpy as np
import pickle, cv2, argparse
import tensorflow as tf
from datetime import timedelta
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from dataflow import RNGDataFlow, MultiProcessRunner, MultiProcessRunnerZMQ, BatchData
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

class VideoBatchGen(RNGDataFlow):
    def __init__(self,
                 video_path, 
                 annot_path,
                 frame_size,
                 window_size,
                 extension,
                 **kwargs):

        print('Initializing generator...')
        os.system('export FFREPORT=level=quiet')
        self.video_path = video_path
        self.annot_path = annot_path
        self.extension = extension
        self.frame_size = frame_size
        self.window_size = window_size
        self._init_stats()
        print('Initialization done.')

    def _list_files(self,data_path,extension):
        l=[]
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if os.path.isfile(os.path.join(root, file)) and extension in file:
                    l.append(os.path.join(root, file))
        return l
    
    def _init_stats(self):
        self.annotations = pickle.load(open(self.annot_path, 'rb'))
        self.files_list = [file for file in self.annotations['stats'] if len(self.annotations['stats'][file])>0]
        self.class_list = list(self.annotations['classes'].keys())
        self.categorical = self.annotations['classes']
        self.nbSamples = 0
        for file in self.files_list:
            nb_frames = self.annotations['stats'][file]['nb_frames']
            if nb_frames >= self.window_size:
                self.nbSamples+=int(np.ceil(nb_frames/self.window_size))

    def __len__(self):
        return self.nbSamples

    def __iter__(self):
        while(True):
            file = self.rng.choice(self.files_list)
            video = cv2.VideoCapture(file)
            nb_frames = self.annotations['stats'][file]['nb_frames']
            if nb_frames < self.window_size:
                #this file has had less than T correct frames
                continue

            Width = self.annotations['stats'][file]['width']
            Height = self.annotations['stats'][file]['height']
            assert Width == self.frame_size
            assert Height == self.frame_size

            ####### Resize original files that are faulty #######
            #if Width - self.frame_size<=0 or Height - self.frame_size<=0:
            #    continue

            #x0 = np.random.randint(0,Width - self.frame_size)
            #y0 = np.random.randint(0,Height - self.frame_size)

            #processing frames
            frames = np.zeros((nb_frames,self.frame_size,self.frame_size,3), dtype='uint8')
            flip = self.rng.choice([False,True])
            cnt=-1
            while(video.isOpened()):
                cnt+=1
                ret, frame = video.read()
                if not ret or cnt>=nb_frames:
                    break
                if flip:
                    frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                #frame = frame[y0:y0+self.frame_size, x0:x0+self.frame_size, :]
                frames[cnt,:] = frame

            video.release()

            if cnt < self.window_size:
                #file is corrupt
                continue
            
            K = int(np.ceil(nb_frames/self.window_size))
            T = self.window_size
            label = self.annotations['stats'][file]['class']
            label = self.categorical[label]
            label = np.reshape(label , (-1,))

            for k in range(K):
                if k<K-1:
                    yield frames[k*T:(k+1)*T,:].astype('uint8'), label.astype('uint8')
                else:
                    yield frames[-T:,:].astype('uint8'), label.astype('uint8')

class LRSchedule(LearningRateSchedule):
    def __init__(self,
                 steps_max,
                 steps_max_warmup,
                 lr_0,
                 lr_0_warmup):
        super(LRSchedule, self).__init__()
        self.steps_max = steps_max
        self.steps_max_warmup = steps_max_warmup
        self.lr_0 = lr_0
        self.lr_0_warmup = lr_0_warmup
        self.lr_max_warmup = lr_0
        self.last_lr = 0

    def lr_cosine(self,step):
        return self.lr_0 * (tf.math.cos(np.pi * (step - self.steps_max_warmup) / self.steps_max) + 1.0) * 0.5

    def lr_warmup(self,step):
        alpha = (self.lr_max_warmup - self.lr_0_warmup) / self.steps_max_warmup
        lr = step * alpha + self.lr_0_warmup
        return lr
    
    def __call__(self,step):
        if self.steps_max_warmup == 0:
            self.last_lr = self.lr_cosine(step)
        else:
            self.last_lr = tf.minimum(self.lr_warmup(step), self.lr_cosine(step))
        return self.last_lr

class Trainer():
    def __init__(self,
                 config,
                 BatchGen=VideoBatchGen,
                 **kwargs):

        print('Initializing trainer...')
        self.config = config

        # get train/data parameters
        self.run_name = self.config['TRAIN']['RUN_NAME']
        self.base_lr_0 = self.config['TRAIN']['BASE_LR_0']
        self.base_lr_0_warmup = self.config['TRAIN']['BASE_LR_0_WARMUP']
        self.warmup_ratio = self.config['TRAIN']['WARMUP_RATIO']
        self.mini_batch_size = self.config['TRAIN']['MINI_BATCH_SIZE']
        self.epochs = self.config['TRAIN']['MAX_EPOCHS']
        self.gpu_workers = self.config['TRAIN']['GPU_WORKERS']
        self.pretrain_ckpt_path = self.config['TRAIN']['PRETRAIN_CKPT_PATH']
        self.pre_train = self.pretrain_ckpt_path is not None
        self.from_mxnet_ckpt = 'pkl' in self.pretrain_ckpt_path
        self.from_tf_ckpt = not self.from_mxnet_ckpt
        self.batch_size = self.mini_batch_size * self.gpu_workers
        self.lr_scale = self.config['TRAIN']['LR_SCALE']
        self.lr_0 = self.lr_scale * self.base_lr_0
        self.lr_0_warmup = self.lr_scale * self.base_lr_0_warmup

        self.video_path = self.config['TRAIN']['DATA']['VIDEO_PATH']
        self.annot_path = self.config['TRAIN']['DATA']['ANNOT_PATH']
        self.window_size = self.config['TRAIN']['DATA']['WINDOW_SIZE']
        self.frame_size = self.config['TRAIN']['DATA']['FRAME_SIZE']
        self.extension = self.config['TRAIN']['DATA']['EXTENSION']
        self.num_classes = self.config['TRAIN']['DATA']['NUM_CLASSES']
        self.num_prefetch = self.config['TRAIN']['DATA']['NUM_PREFETCH']
        self.num_process = self.config['TRAIN']['DATA']['NUM_PROCESS']

        #config gpu workers
        gpu_ids = ','.join([str(g) for g in range(self.gpu_workers)])
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        
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
        self.df = BatchGen(video_path=self.video_path, 
                            annot_path = self.annot_path,
                            frame_size=self.frame_size,
                            window_size=self.window_size,
                            extension=self.extension)

        #create dataflow for faster batchgen
        self.gen = MultiProcessRunnerZMQ(self.df, num_proc=self.num_process, hwm=self.num_prefetch)
        self.gen = BatchData(self.gen, self.batch_size)
        self.gen.reset_state()
        
        def tuple_gen():
            for d in self.gen:
                yield (d[0], tf.keras.utils.to_categorical(d[1],self.num_classes))

        self.vid_shape = [self.batch_size,self.window_size,self.frame_size,self.frame_size,3]
        self.lbl_shape = [self.batch_size,self.num_classes]
        self.dataset = tf.data.Dataset.from_generator(tuple_gen,output_types=(tf.uint8,tf.uint8),output_shapes=(self.vid_shape,self.lbl_shape))
        self.dataset = self.dataset.repeat(5000)
        self.dist_dataset = self.strategy.experimental_distribute_dataset(self.dataset)
        self.iter_dataset = iter(self.dist_dataset)
        self.steps_per_epoch = int(len(self.df)//self.batch_size)
        self.steps_max = self.epochs * self.steps_per_epoch
        self.steps_max_warmup = int(self.warmup_ratio * self.steps_max)

        self.lr_non_bn = LRSchedule(steps_max=self.steps_max,
                                    steps_max_warmup=self.steps_max_warmup,
                                    lr_0=self.lr_0,
                                    lr_0_warmup=self.lr_0_warmup)

        self.lr_bn = LRSchedule(steps_max=self.steps_max,
                                steps_max_warmup=self.steps_max_warmup,
                                lr_0=self.lr_0,
                                lr_0_warmup=self.lr_0_warmup)

        self.optimizer_non_bn = tf.keras.optimizers.SGD(learning_rate=self.lr_non_bn,
                                                             momentum=0.9,
                                                             nesterov=True, 
                                                             decay=1e-4,
                                                             name='SGD_Non_BN')

        #following arXiv:1706.02677v2, we set decay=0 for all batch_norm weights
        self.optimizer_bn = tf.keras.optimizers.SGD(learning_rate=self.lr_bn,
                                                     momentum=0.9,
                                                     nesterov=True, 
                                                     decay=0.,
                                                     name='SGD_BN')

        self.metrics = {}
        self._init_metrics()
        self.writer = tf.summary.create_file_writer(logdir=os.path.join('./logs',self.run_name))
        tf.summary.trace_on(graph=False, profiler=False)

    def add_metric(self,
                   metric):
        names = metric.name.split('/')
        if len(names)!=2:
            raise ValueError('Please pass a metric with name pattern: matric_type/metric_name')
        mtype, mname = names
        if mtype not in self.metrics:
            self.metrics[mtype] = {}
        self.metrics[mtype].update({mname: metric})

    def reset_metrics(self):
        for mtype in self.metrics:
            for mname in self.metrics[mtype]:
                self.metrics[mtype][mname].reset_states()

    def update_tf_summary(self, step):
        with self.writer.as_default():
            for mtype in self.metrics:
                for mname in self.metrics[mtype]:
                    summary_name = '{}/{}'.format(mtype, mname)
                    summary_result = self.metrics[mtype][mname].result().numpy()
                    if mtype == 'Losses' or mtype == 'Accuracies':
                            tf.summary.scalar(summary_name, summary_result, step=step)
                    elif mtype == 'Distributions':
                            tf.summary.histogram(summary_name, summary_result, step=step)
                            self.metrics[mtype][mname].reset_states()
    
    def pretty_progress(self,
                        step,
                        steps_max,
                        epoch,
                        epochs_max,
                        t_step,
                        **metrics):

        eta_secs = min(2**32, int((steps_max * (epochs_max - epoch) - step) * t_step))
        progress_str = ''
        progress_str += 'Iter {}/{}'.format(step+1,steps_max)
        for metric in metrics:
            progress_str += ' - {}: {:0.4f}'.format(metric, metrics[metric])
        
        progress_str += '{:>5} //{:>5} ETA {:0>8}, {:0.2f}/step {:>10}'.format('', '', str(timedelta(seconds=eta_secs)), t_step, '')
        progress_str += '\r'
        sys.stdout.write(progress_str)
        sys.stdout.flush()

    def _compute_loss_0(self,
                     labels,
                     predictions,
                     reduction=tf.losses.Reduction.AUTO):
        loss_object = tf.keras.losses.CategoricalCrossentropy(reduction = reduction)
        loss_ = loss_object(labels, predictions)
        return loss_

    def _init_metrics(self):
        self.add_metric(tf.keras.metrics.Mean(name='Losses/Train_Loss'))
        self.add_metric(tf.keras.metrics.CategoricalAccuracy(name='Accuracies/Train_Accuracy'))

    def _compute_loss(self, labels, predictions):
        per_example_loss = self._compute_loss_0(labels = labels,
                                                predictions = predictions,
                                                reduction = tf.keras.losses.Reduction.NONE)

        avg_loss = tf.reduce_mean(per_example_loss) #(1/n)*L ; n is size of miniBatch
        
        self.metrics['Losses']['Train_Loss'].update_state(avg_loss)
        self.metrics['Accuracies']['Train_Accuracy'].update_state(labels, predictions)

        return avg_loss


    def train(self):
        #note that average over loss is done after applying gradients (k times)
        #hence we should scale lr by k: number of workers/replica
        @tf.function(input_signature=[self.iter_dataset.element_spec])
        def _train_step(dist_inputs):
            def _step_fn(inputs):
                frames, labels = inputs
                with tf.GradientTape() as tape:
                    predictions = self.model(inputs=frames, training=tf.constant(True))
                    loss = self._compute_loss(labels, predictions)
                
                weights = self.model.trainable_variables
                gradients = tape.gradient(loss, weights)
                
                vars_non_bn = [[g,w] for g,w in zip(gradients,weights) if 'BatchNorm' not in w.name]
                vars_bn = [[g,w] for g,w in zip(gradients,weights) if 'BatchNorm' in w.name]

                #these optimizers apply gradients k times without averaging
                #hence, this k should be compensated in learning rate
                self.optimizer_non_bn.apply_gradients(vars_non_bn)
                self.optimizer_bn.apply_gradients(vars_bn)

                return loss

            per_replica_losses = self.strategy.run(_step_fn, args=(dist_inputs,))
            return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        with self.strategy.scope():
            ckpt_dir = os.path.join('./checkpoints',self.run_name)
            manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                 directory=ckpt_dir,
                                                 max_to_keep=None)
            print('Training...')
            for e in range(self.epochs):
                self.reset_metrics()
                print('\n======Epoch: {}/{}'.format(e+1,self.epochs))
                for step in range(self.steps_per_epoch):
                    t_step = time.time()

                    inputs = self.iter_dataset.next()
                    loss_ = _train_step(inputs)
                    
                    if step % 10 == 0:
                        self.update_tf_summary(step = step + e * self.steps_per_epoch)
                    
                    t_step = time.time() - t_step
                    self.pretty_progress(step=step,
                                         steps_max=self.steps_per_epoch,
                                         epoch=e,
                                         epochs_max=self.epochs,
                                         t_step=t_step,
                                         loss=self.metrics['Losses']['Train_Loss'].result().numpy(),
                                         acc=self.metrics['Accuracies']['Train_Accuracy'].result().numpy())

                #save a checkpoint every epoch
                #if (e+1)%5==0:
                prnt_loc = manager.save(checkpoint_number=e+1)

class SlowFast_Trainer(Trainer):
    def __init__(self,
                 config,
                 **kwargs):

        super(SlowFast_Trainer, self).__init__(config=config, **kwargs)
        self.freeze_backbone = self.config['TRAIN']['FREEZE_BACKBONE']
        self.slow_config = self.config['SLOW_CONFIG']
        self.fast_config = self.config['FAST_CONFIG']
        self.slow_fast_params = self.config['SLOW_FAST_PARAMS']
        
        with self.strategy.scope():
            print('Building video model...')
            self.model = SlowFast(num_classes=self.num_classes,
                                  slow_config=self.slow_config,
                                  fast_config=self.fast_config,
                                  pre_activation=self.slow_fast_params['PRE_ACTIVATION'],
                                  temporal_sampling_rate=self.slow_fast_params['TMP_SMPL_RATE'],
                                  spatial_sampling_rate=self.slow_fast_params['SPT_SMPL_RATE'],
                                  fusion_kernel=self.slow_fast_params['FUSION_KERNEL'],
                                  alpha=self.slow_fast_params['ALPHA'],
                                  beta=self.slow_fast_params['BETA'],
                                  tau=self.slow_fast_params['TAU'],
                                  dropout_rate=self.slow_fast_params['DROPOUT_RATE'],
                                  epsilon=self.slow_fast_params['EPSILON'],
                                  momentum=self.slow_fast_params['MOMENTUM'],
                                  data_format=self.slow_fast_params['DATA_FORMAT'])

            self.checkpoint = tf.train.Checkpoint(model=self.model)

            print('Initializing variables...')
            self.model.set_strategy(self.strategy)
            self.model.distributed_init(self.vid_shape)

            if self.pre_train:
                if self.from_tf_ckpt:
                    print('Loading variables from TF checkpoint...')
                    self.checkpoint.restore(self.pretrain_ckpt_path)#.assert_consumed()
                
                elif self.from_mxnet_ckpt:
                    print('Loading variables from MXNET checkpoint...')
                    pkl_handle = pickle.load(open(self.pretrain_ckpt_path, 'rb'))
                    skipped = assign_weight_from_pkl(self.model, pkl_handle)
                    print('{} weights were skipped:\n{}'.format(len(skipped), skipped))

            if self.freeze_backbone:
                self.model.freeze_backbone()

        print('All initializations done.')

def main(config):
    trainer = SlowFast_Trainer(config = config)
    trainer.train()

if __name__ == '__main__':
    description = 'The main train call function.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config_path', type=str, default='./configs/SlowFast_R50_8x8.yml')

    args = parser.parse_args()
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)

    main(config)