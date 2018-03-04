'''
Created on Jan 16, 2018

@author: kohill
'''
from __future__ import print_function
from __future__ import division
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import cv2,os,random
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from pycocotools import mask as maskUtils
import json
class DataIter(Dataset):
    def __init__(self):
        annFile = '/home/kohill/hszc/data/coco/annotations/person_keypoints_train2014.json' # keypoint file
        self.trainimagepath = '/home/kohill/hszc/data/coco/train2014'             # train image path        
        coco = COCO(annFile)
        self.catIds = coco.getCatIds(catNms=['person'])
        self.imgIds = coco.getImgIds(catIds=self.catIds );  
        self.coco_kps = coco
        self.NUM_PARTS=18
        self.NUM_LINKS=19
        self.HEAT_RADIUS = 12
        self.PART_LINE_WIDTH=16
        self.annos_path = "anno.json"
        if not os.path.exists(self.annos_path):
            self.annos = self.convert_anno()
            json.dump(self.annos,open(self.annos_path,"wb"))
        else:
            self.annos = json.load(open(self.annos_path,"rb"))
    def __len__(self):
        return len(self.annos)
    def convert_anno(self):
        all_annos = []
        #{"img_path","img_id","person_index","keypoints"}
        for index in range(len(self.imgIds)):
            img = self.coco_kps.loadImgs(self.imgIds[index])[0]
            annIds = self.coco_kps.getAnnIds(imgIds=img['id'], catIds=self.catIds, iscrowd=None)
            anns = self.coco_kps.loadAnns(annIds)
            
            # plt.imshow(img_ori)
            # self.coco_kps.showAnns(anns)
            # plt.show()
            assert len(anns) > 0
            assert 'segmentation' in anns[0] or 'keypoints' in anns[0]
    
            polygons = []
            keypoints = [] #(part_id,x,y)
            parts = []#((partid0,x0,y0),(partid1,x1,y1))
            for ann in anns:
                one_anno = {}            
                one_anno["file_name"] = img['file_name']
                one_anno['id'] = img['id']
                one_anno['bbox'] = []                
                one_anno['keypoints'] = []                
                one_anno['limbs'] = []                

                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        polies = []
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polies.append(poly)
                            polygons.append(Polygon(poly))
                        polies = np.concatenate(polies,axis = 0)
                        x0 = np.min(polies[:,0])
                        y0 = np.min(polies[:,1])
                        x1 = np.max(polies[:,0])
                        y1 = np.max(polies[:,1])
                        one_anno['bbox'] = [x0,y0,x1,y1]                
                        if 'keypoints' in ann and (ann['num_keypoints'] < 5 or ann['area']< 32*32) :
                            continue
                            for seg in ann['segmentation']:
                                poly = np.array(seg).reshape((int(len(seg)/2), 2))
#                                 cv2.drawContours(loss_mask,[poly[np.newaxis,:].astype(np.int32)],0,(0,0,0),-1)
    
                    else:
                        # mask
                        t = self.coco_kps.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
    
#                         loss_mask *= (1.0-m[:,:,0]).astype(np.float32)
    
                COCO_to_ours_1 = [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
                COCO_to_ours_2 = [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
                mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]
                mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]   
                assert len(COCO_to_ours_1) == len(COCO_to_ours_2) == self.NUM_PARTS                 
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
    #                 sks = np.array(self.coco_kps.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
    
                    x_coco = kp[0::3]
                    y_coco = kp[1::3]
                    v_coco = kp[2::3]
                    x = []
                    y = []
                    v = []
                    for index1,index2 in zip(COCO_to_ours_1,COCO_to_ours_2):
                        index1 -= 1
                        index2 -= 1
                        x.append(0.5*(x_coco[index1] + x_coco[index2]))
                        y.append(0.5*(y_coco[index1] + y_coco[index2]))
                        v.append(min(v_coco[index1],v_coco[index2]))
                    for i in range(self.NUM_PARTS):
                        if v[i] > 0:
                            # cv2.circle(heatmaps[i],(int(round(x[i])),int(round(y[i]))),self.HEAT_RADIUS,(1,1,1),-1)
                            keypoints.append([i,x[i],y[i]])
                            one_anno['keypoints'].append([i,x[i],y[i]])
                    for i in range(self.NUM_LINKS):
                        kp0,kp1 = mid_1[i]-1,mid_2[i]-1
                        if v[kp0] > 0 and v[kp1] > 0:
                            parts.append([i,x[kp0],y[kp0],x[kp1],y[kp1]])
                            one_anno['limbs'].append([i,x[kp0],y[kp0],x[kp1],y[kp1]])    
                if len(one_anno)>0 and len(one_anno["bbox"])==4:
                    all_annos.append(one_anno)
        return all_annos
    def __getitem__(self, index):
        ishape = (96*2,96)
        anno =  self.annos[index]
        img_path = os.path.join(self.trainimagepath, anno["file_name"])
        img_ori = cv2.imread(img_path)
        x0,y0,x1,y1 =map(lambda x: int(x), anno['bbox'])
        img_cropped = img_ori[y0:y1,x0:x1,:]
        img_cropped = cv2.resize(img_cropped,(ishape[1],ishape[0]))
        fscalex = img_cropped.shape[1]/(x1-x0)
        fscaley = img_cropped.shape[0]/(y1-y0)
        keypoints = map(lambda x:(x[0],(x[1]-x0)*fscalex,(x[2]-y0)*fscaley),anno['keypoints'])
#         print(keypoints)
        stride = 8.0
        heatmaps = [np.zeros(
            shape = (int(ishape[0]//stride),int(ishape[1]//stride)),dtype=np.float32) for _ in range(self.NUM_PARTS)]
        for m in range(int(ishape[0]//stride)):
            for n in range(int(ishape[1]//stride)):
                ori_x = n *stride + stride / 2 - 0.5
                ori_y = m * stride + stride / 2 - 0.5
                for  pard_id,x,y in keypoints:
                    d2 = (ori_x-x)**2+(ori_y-y)**2
                    thgma  = 7.0
                    exponent = d2 / 2.0 / (thgma**2)
                    heatmaps[pard_id][m, n] = max(np.exp(-exponent), heatmaps[pard_id][m,n])
        
        img_cropped = np.transpose(img_cropped, (2,0,1))
        return [img_cropped,np.array(heatmaps)]
def draw_heatmap(heatmap,img=None):
    _,axes = plt.subplots(4,5,figsize=(35,28))
    plt.subplots_adjust(wspace = 0,hspace = 0.15)
    for i in range(5):
        for j in range(4):
            index = i*5+j                
            if index < heatmap.shape[0]:
#                 heatmap[index,:,:][0,0] = 1
                axes[j][i].imshow(heatmap[index,:,:])
            elif index == heatmap.shape[0]:
                axes[j][i].title.set_text("max")
                axes[j][i].imshow(np.max(heatmap,axis = 0))                    
            elif img is not None and index == heatmap.shape[0]+1:
                axes[j][i].title.set_text("img")
                axes[j][i].imshow(img)
            else:
                axes[j][i].imshow(heatmap[-1,:,:])
def collate_fn(batch):
    imgs_batch = []
    heatmaps_batch = []
#     pafmaps_batch = []
#     loss_mask_batch = []
    for sample in batch:
        img_ori,heatmaps = sample
        imgs_batch.append(img_ori[np.newaxis])
        heatmaps_batch.append(heatmaps[np.newaxis])
#         pafmaps_batch.append(pafmaps[np.newaxis])
#         loss_mask_batch.append(loss_mask[np.newaxis])
    imgs_batch =np.concatenate(imgs_batch,axis = 0)
    heatmaps_batch =np.concatenate(heatmaps_batch,axis = 0)
#     pafmaps_batch =np.concatenate(pafmaps_batch,axis = 0)
#     loss_mask_batch =np.concatenate(loss_mask_batch,axis = 0)            
    return [imgs_batch,heatmaps_batch]    
def getDataLoader(batch_size = 16):
    test_iter = DataIter()
    r = DataLoader(test_iter, batch_size=batch_size, shuffle=True, num_workers=5, collate_fn=collate_fn, pin_memory=False,drop_last = True)
    return r
if __name__ == '__main__':
    print("length",len(getDataLoader(8)))
    data_iter = DataIter()
    for i in range(len(data_iter)):
        da = data_iter[i]
        for d in da:
            print(d.shape)

        x = list(map(lambda x: np.transpose(x,(1,2,0)) if len(x.shape) > 2 else x, da))
        
        fig, axes = plt.subplots(2, len(x)//2 + len(x)%2, figsize=(45, 45),
                             subplot_kw={'xticks': [], 'yticks': []},squeeze = False)
        fig.subplots_adjust(hspace=0.3, wspace=0.05) 
 
        count = 0
       
        for j in range(len(axes)):
            for i in range(len(axes[0])):
                try:
                    img = x[count]
                    count += 1
                except IndexError:
                    break
                print(count,len(x))
                if len(img.shape)>2 and img.shape[2]==38:
                    img = np.array([np.sqrt(img[:,:,k *2] ** 2 + img[:,:,k *2+1 ] ** 2) for k in range(img.shape[2]//2)])
                    axes[j][i].imshow(np.max(img,axis = 0))
                    print("limb")
                elif len(img.shape)>2 and img.shape[2] > 3:
                    axes[j][i].imshow(np.max(img[:,:,:-1],axis = 2)) 
                elif len(img.shape)>2 and img.shape[2] == 1:
                    axes[j][i].imshow(img[:,:,0]) 
                else:
                    axes[j][i].imshow(img.astype(np.uint8))
                     
        plt.show()
    pass