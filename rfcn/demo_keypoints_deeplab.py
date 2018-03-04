# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li, Haocheng Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
import matplotlib.pyplot as plt
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/rfcn/cfgs/rfcn_coco_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Deformable ConvNets demo')
    # general
    parser.add_argument('--rfcn_only', help='whether use R-FCN only (w/o Deformable ConvNets)', default=False, action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()

def detect_humans(img_path):
    # get symbol
#     pprint.pprint(config)
    config.symbol = 'resnet_v1_101_rfcn_dcn' if not args.rfcn_only else 'resnet_v1_101_rfcn'
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_symbol(config, is_train=False)

    # set up class names
    num_classes = 81
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # load demo data
    image_names = [img_path]
    data = []
    for im_name in image_names:
#         assert os.path.exists(cur_path + '/../demo/' + im_name), ('%s does not exist'.format('../demo/' + im_name))
#         im = cv2.imread(cur_path + '/../demo/' + im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        im = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)        
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        data.append({'data': im_tensor, 'im_info': im_info})


    # get predictor
    data_names = ['data', 'im_info']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + '/../model/' + ('rfcn_dcn_coco' if not args.rfcn_only else 'rfcn_coco'), 0, process=True)
    predictor = Predictor(sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)

    # warm up
    for j in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[0]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[0])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)

    # test
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        tic()
        scores, boxes, data_dict = im_detect(predictor, data_batch, data_names, scales, config)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
            dets_nms.append(cls_dets)
        print 'testing {} {:.4f}s'.format(im_name, toc())
        # visualize
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#         show_boxes(im, [dets_nms[0]], [classes[0]], 1)
        return im,dets_nms[0]
    print 'done'
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
def get_keypoints_model():
    from resnet_v1_101_deeplab import get_symbol
    sym = get_symbol(is_train=False)
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(1)],
                          label_names  = ['heatmaplabel'])
    model.bind(data_shapes=[('data', (1, 3, 96*2, 96))])
    args,auxes = load_checkpoint("/home/kohill/Desktop/projects/Deformable-ConvNets/model/keypoints/resnet-101final",14)
    model.init_params(arg_params=args, aux_params=auxes, allow_missing=False,allow_extra = True)
    return model
def keypoints_detect(model,humans,scales):
    model.forward(mx.io.DataBatch(data = [mx.nd.array(humans)]),is_train = False)
    results = model.get_outputs()[0]
    predi = results.asnumpy()
    humans_keypoints = []
    for m in range(predi.shape[0]):
        keypoints = np.zeros(51)
#         fig,axes = plt.subplots(1,2,squeeze=True)
#         axes[0].imshow(np.transpose(humans[m],(1,2,0)))
#         axes[1].imshow(np.max(predi[m],axis = 0))
#         plt.show()
        orderCOCO = [1, 0, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        for n in range(len(orderCOCO)):
            heat = cv2.resize(predi[m,n],(96,2*96))
            heat_max = np.max(heat)
            heat_max_index = np.argwhere(heat_max == heat)
            y0,x0 = heat_max_index[0,:]
            x0 /= scales[m][0]
            y0 /= scales[m][1]
            x0 += scales[m][2]
            y0 += scales[m][3]
            v = 1 if heat_max > .9 else 0
            index = orderCOCO[n] - 1
            if index >= 0:
                keypoints[index *3 :(index*3+3)] = [x0,y0,v]
        humans_keypoints.append(keypoints.tolist())
    return humans_keypoints
def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval    
    import pprint as pp
    import tempfile,json
    images_path = "/home/kohill/hszc/data/coco/val2014"
    import matplotlib.pyplot as plt
    from random import randint
    randc =lambda :(randint(128,255),randint(128,255),randint(128,255))
    key_points_module = get_keypoints_model()
    
    annFile = '/home/kohill/hszc/data/coco/annotations/person_keypoints_val2014.json'
    cocoGt = COCO(annFile)
    cats = cocoGt.loadCats(cocoGt.getCatIds())
    catIds = cocoGt.getCatIds(catNms=['person'])
    imgIds = cocoGt.getImgIds(catIds=catIds)
    validate_result= []
    validate_ids = []
    for i in range(50):
#     for file_name in os.listdir(images_path):
        oneimg =  cocoGt.loadImgs(imgIds[i])[0]
        validate_ids.append(oneimg['id'])
        img_path = os.path.join(images_path,oneimg["file_name"])
        img_ori = cv2.imread(img_path)
        img_canvas = img_ori.copy()
        img_forwarded,human_bboxes = detect_humans(img_path)
        humans = []
        scales = []
        for m in range(human_bboxes.shape[0]):
            x0,y0,x1,y1,cls = human_bboxes[m,:].astype(np.int32)
            img_cropped = cv2.resize(img_ori[y0:y1,x0:x1,:],(96,96*2))
            humans.append(np.transpose(img_cropped,(2,0,1))[np.newaxis])
            fscale = img_forwarded.shape[0]/img_ori.shape[0]
            scales.append((96.0/(x1-x0) * fscale ,
                           96.0*2/(y1-y0) * fscale,x0,y0))
        if len(humans) == 0:
#             return []
            continue
        humans = np.concatenate(humans,axis = 0)
        humans_keypoints = keypoints_detect(key_points_module, humans,scales)
        for keypoints in humans_keypoints:
            cu_dict = {}
            cu_dict['image_id'] = oneimg['id']
            cu_dict['category_id'] = 1
            cu_dict['keypoints'] = keypoints
            cu_dict['score'] = 10
            validate_result.append(cu_dict)
#         for hum in humans_keypoints:
#             for i in range(len(hum)):
#                 x,y,v = map(lambda x:int(x),hum[i])
#                 if v:
#                     cv2.circle(img_canvas,(x,y),8,randc(),-1)
#         plt.imshow(img_canvas)
#         plt.show()
    resJsonFile = tempfile.mktemp(suffix="json", prefix="result")
    json.dump(validate_result, open(resJsonFile,"wb"))
    cocoDt2 = cocoGt.loadRes(resJsonFile)
    cocoEval = COCOeval(cocoGt, cocoDt2, "keypoints")
    cocoEval.params.imgIds = validate_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    print(cocoEval.summarize())
if __name__ == '__main__':
    main()
