'''
@author: kohill

'''
from __future__ import print_function
import  os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from tensorboardX import SummaryWriter
from data_iter import getDataLoader
from resnet_v1_101_deeplab import get_symbol
import mxnet as mx
import logging,os
import numpy as np

BATCH_SIZE = 64
NUM_LINKS = 19
NUM_PARTS =  18

SAVE_PREFIX = "models/resnet-101"
PRETRAINED_PREFIX = "pre/deeplab_cityscapes"
LOGGING_DIR = "logs"
def load_checkpoint(prefix, epoch):

    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}

    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def train(retrain = True,ndata = 16,gpus = [0,1],start_n_dataset = 0):
    input_shape = (368,368)
    stride = (8,8)
    sym = get_symbol(is_train = True, numberofparts = NUM_PARTS, numberoflinks= NUM_LINKS)
    batch_size = BATCH_SIZE
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(g) for g in gpus],
                          label_names  = ['heatmaplabel'])
    model.bind(data_shapes=[('data', (batch_size, 3, 96*2, 96))], label_shapes=[
        ('heatmaplabel', (batch_size, 18, 24, 12))
        ]
    )
    summary_writer = SummaryWriter(LOGGING_DIR)
    if retrain:
        args,auxes = load_checkpoint("pre/rcnn_coco",0)
    else:
        args,auxes = load_checkpoint(SAVE_PREFIX+"final",start_n_dataset)
    data_iter = getDataLoader(batch_size = BATCH_SIZE)

    lr_scheduler = mx.lr_scheduler.FactorScheduler(step = len(data_iter)*4,factor=0.1)
    model.init_params(arg_params=args, aux_params=auxes, allow_missing=True,allow_extra = True,initializer=mx.init.Xavier())
    model.init_optimizer(optimizer='rmsprop',
                        optimizer_params={'learning_rate':1e-4,
#                                            "momentum":0.9,
#                                            "wd":0.0001,
#                                           'clip_gradient': 5,
                                          'lr_scheduler': lr_scheduler,
#                                         "rescale_grad":1/BATCH_SIZE
                                           } ,
                                          )
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.ion()

    plt.show()
    for n_data_wheel in range(ndata):
        if n_data_wheel > 0:
            model.save_checkpoint(SAVE_PREFIX + "final", n_data_wheel + start_n_dataset)
        for nbatch,data_batch in enumerate( data_iter):
            data = mx.nd.array(data_batch[0])
            label = [mx.nd.array(x) for x in data_batch[1:]]
            model.forward(mx.io.DataBatch(data = [data],label = label), is_train=True)
            predi=model.get_outputs()
            model.backward()
            model.update()
            global_step = nbatch + len(data_iter)*n_data_wheel
            print("{0} {1} {2}".format(global_step,n_data_wheel,nbatch),end = " ")
            for i in range(1):
                loss = mx.nd.sum(predi[i*2]).asnumpy()[0]/BATCH_SIZE
                summary_writer.add_scalar("heatmap_loss_{}".format(i),loss,
                                          global_step = nbatch)
                print(loss,end = " ")
            if nbatch %100 == 0:
                plt.imshow(np.max(predi[-1][0].asnumpy(),axis = 0))
                plt.pause(0.001)
            print("")
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    train(retrain = False, gpus = [0,1],start_n_dataset = 15)