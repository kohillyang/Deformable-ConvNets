import mxnet as mx
import xml.etree.ElementTree as ET
import os,logging
import numpy as np
import cv2
import sklearn
from sklearn.model_selection import train_test_split
import copy
from gluoncv.data.transforms import bbox as bbox_transform

from random import randint
from random import random
import gluoncv
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
    def __len__(self):
        return len(self.transforms)
class bbox_random_rotate():
    def __init__(self,min_angle = -3,max_angle = 3):
        self.min_angle = min_angle
        self.max_angle = max_angle
    def tranform_point(self,matrix,point):
        x,y = point
        p0 = np.array([x, y]).astype(np.float32).reshape(1, 1, 2)
        p = cv2.transform(p0, matrix).reshape(1, -1)
        x_r = p[0, 0]
        y_r = p[0, 1]
        return np.array([x_r,y_r])
    def __call__(self,img,bbox):
        angle = randint(self.min_angle, self.max_angle)
        matrix = cv2.getRotationMatrix2D((img.shape[1] // 2., img.shape[0] / 2.), angle, 1.0);
        img_r = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
        bbox_r = []
        for i in range(len(bbox)):
            xmin,ymin,xmax,ymax = bbox[i,:4]
            point0 = (xmin,ymin)
            point1 = (xmax,ymin)
            point2 = (xmax,ymax)
            point3 = (xmin,ymax)
            points = [point0,point1,point2,point3]
            points = list(map(lambda x:self.tranform_point(matrix,x),points))
            points = np.array(points)

            xmin = np.min(points[:,0])
            ymin = np.min(points[:,1])
            xmax = np.max(points[:,0])
            ymax = np.max(points[:,1])

            xmin = max(0,xmin)
            ymin = max(0,ymin)
            xmax = min(xmax,img.shape[1]-1)
            ymax = min(ymax,img.shape[0]-1)
            bb = [xmin,ymin,xmax,ymax] + bbox[i,4:].tolist()
            bbox_r.append(bb)
        bbox_r = np.array(bbox_r).conjugate()
        return img_r,bbox_r
class bbox_random_resize():
    def __init__(self,min_scale = 0.8,max_scale = 1.2):
        self.min_scale = 0.7
        self.max_scale = 1.2
    def __call__(self, img,bbox):
        fscale = random() * (self.max_scale-self.min_scale) + self.min_scale
        dshape = [int(fscale * img.shape[0]),int(fscale*img.shape[1])]
        img_r = cv2.resize(img,(dshape[1],dshape[0]))
        bbox_r = bbox_transform.resize(bbox,(img.shape[1],img.shape[0]),(dshape[1],dshape[0]))
        return img_r,bbox_r
class bbox_image_pad():
    def __init__(self,dest_shape = (1024,1024)):
        self.dest_shape = dest_shape #h,w
    def __call__(self,img_ori,bboxes):
        dshape = [1024.0,1024.0]
        fscale = min(dshape[0]/img_ori.shape[0],dshape[1]/img_ori.shape[1])
        img_resized = cv2.resize(img_ori,dsize=(0,0),fx=fscale,fy=fscale)  # type: numpy.ndarray
        bboxes = bbox_transform.resize(bboxes,(img_ori.shape[1],img_ori.shape[0]),(img_resized.shape[1],img_resized.shape[0]))
        img_padded = np.zeros(shape = (int(dshape[0]),int(dshape[1]),3),dtype=np.float32)
        img_padded[:img_resized.shape[0],:img_resized.shape[1],:img_resized.shape[2]] = img_resized
        return img_padded,bboxes

class bbox_Aug_train():
    def __init__(self):
        self.img_pad = bbox_image_pad()
        self.bbox_roate = bbox_random_rotate()
        self.bbox_resize = bbox_random_resize()
        self.aug = Compose([self.img_pad,self.bbox_roate])
    def __call__(self,img,bbox):
        return self.aug(img,bbox)
    def __len__(self):
        return len(self.aug)
class bbox_Aug_val():
    def __init__(self):
        pass
    def __call__(self,img,bbox):

        return img,bbox
    def __len__(self):
        return 0
class DetectDataset(mx.gluon.data.Dataset):
    def __init__(self,
                 anno__path,
                 images_set = None,
                 reindex = None
                 ):
        self.img_root = "/data1/zyx/yks/baidu/round2/datasets/train/"
        self.num_classes = 61
        lines = open(anno__path,"rt") .readlines()
        self.lines = lines
        for i in range(len(lines)):
            self.lines[i] = str(lines[i]).strip().split(" ")
            # print(self.lines[i])
            for j in range(5):
                self.lines[i][j+1] = int(self.lines[i][j+1])
                # assert self.lines[i][j+1] > 0
        self.obj = {}
        for l in self.lines:
            img_name,cls,x0,y0,x1,y1 = l
            img_path = img_name
            assert cls < self.num_classes
            try:
                self.obj[img_path].append([x0,y0,x1,y1,cls])
            except KeyError:
                self.obj[img_path]= [[x0,y0,x1,y1,cls]]

        # filter images
        if images_set:
            self.obj_filtered = {}
            for key in self.obj:
                if key in images_set:
                    self.obj_filtered[key] = self.obj[key]
        self.obj = self.obj_filtered

        self.keys = list(self.obj.keys())

        if reindex == None:
            self.reindex = list(range(len(self.keys)))
        else:
            self.reindex = reindex

    def __len__(self):
        return len(self.reindex)
    def __getitem__(self, idx):
        img_path = self.keys[idx]
        img_path = os.path.join(self.img_root,img_path)
        assert os.path.exists(img_path),img_path

        img_ori = cv2.imread(img_path)[:,:,::-1]
        bboxes = self.obj[self.keys[idx]]
        bboxes = np.array(bboxes)

        return img_path,mx.nd.array(img_ori),bboxes

def getDataset(txt_path,images_set):
    all_dataset = DetectDataset(txt_path,images_set)

    return all_dataset
def write_roid_db(dataset,db_prefix = "dataset_processed/train",aug = None):
    r = []
    import tqdm,shutil
    bar = tqdm.tqdm(total = len(dataset),desc=db_prefix)


    for i in range(len(dataset)):
        bar.update(1)
        img_path_ori,img,bbox = dataset[i]
        img = img.asnumpy()

        onedb = {}
        onedb["boxes"] = bbox[:,:4].astype(np.int32)
        onedb["height"] = img.shape[0]
        onedb["width"] = img.shape[1]
        onedb["image"] = img_path_ori
        onedb["flipped"] = False

        num_objs = bbox.shape[0]
        num_classes = dataset.num_classes
        overlaps = np.zeros(shape=(num_objs, num_classes), dtype=np.float32)
        for idx in range(bbox.shape[0]):
            cls = bbox[idx,4]
            overlaps[idx,cls] = 1.0
        onedb["gt_classes"] = bbox[:,4].astype(np.int32)
        onedb["gt_overlaps"] = overlaps
        onedb["max_classes"] = overlaps.argmax(axis=1)
        onedb["max_overlaps"] = overlaps.max(axis=1)
        r.append(onedb)
    bar.close()
    import pickle
    # pickle.dump(r, open("roidb.pickle", "wb"), protocol=0)
    pickle.dump(r,open(os.path.join(db_prefix,"roidb.pickle"),"wb"),protocol = 0)
def viz_roidb(db_path):
    import pickle
    from matplotlib import pyplot as plt
    from gluoncv.utils import viz
    roidb = pickle.load(open(db_path,"rb"))
    for onedb in roidb:
        bbox = onedb["boxes"]
        img_path = onedb["image"]
        img = cv2.imread(img_path)
        ax = viz.plot_bbox(img, bbox, labels=None, class_names=["###"])
        plt.show()
    return

def label_train_test_split(train_txt = "/data1/zyx/yks/baidu/round2/datasets/train.txt"):
    import pandas as pd
    image_anno_all = pd.read_csv(train_txt,header=None,names=["id","label","x1","y1","x2","y2"],sep=None)
    img_files_unique = image_anno_all.drop_duplicates(["id"])
    train_pd,val_pd = train_test_split(img_files_unique,test_size=0.1,random_state=37,shuffle=True,stratify=img_files_unique["label"])
    train_images = set(train_pd["id"])
    val_images = set(val_pd["id"])
    return train_images,val_images
if __name__ == '__main__':
    # label_train_test_split()
    # exit(-1)
    # from matplotlib import pyplot as plt
    # from gluoncv.utils import viz
    # # logging.basicConfig(level=logging.INFO)
    train_images,val_images = label_train_test_split()
    print(len(train_images),len(val_images))
    train_dataset = getDataset("/data1/zyx/yks/baidu/round2/datasets/train.txt",images_set = train_images )
    print(len(train_dataset))
    import gluoncv,matplotlib.pyplot as plt
    # train_aug = bbox_Aug_train()
    # val_aug = bbox_Aug_val()
    write_roid_db(train_dataset,db_prefix="output/dataset/train")
    # # write_roid_db(val_dataset,db_prefix="/data1/zyx/yks/ocr_formula/generated/dataset_3375_plus_4948_aug/val",aug = val_aug)
    # viz_roidb("/data1/zyx/yks/ocr_formula/generated/dataset_3375_plus_4948_aug/train/roidb.pickle")
    for i in range(len(train_dataset)):
        _,train_image, train_label = train_dataset[i]
        train_image = train_image.asnumpy()
        # train_image,train_label = train_aug(train_image,train_label)

        try:
            bboxes = train_label[:, :4]
            cids = train_label[:, 4:5]
            ax = gluoncv.utils.viz.plot_bbox(train_image, bboxes, labels=cids)
            plt.show()
        except IndexError as e:
            logging.exception(e)