import cv2
import numpy as np
import torch
import random
from torch.utils.data.dataset import Dataset

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + 1 + num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #--------------------------------------------------------------------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #   注意：这部分需要根据自定义数据集情况自定义，对于robomaster比赛现场不可能出现的情况不进行增强，以下修改思路均如此
        #--------------------------------------------------------------------------------------------------------------#
        if self.epoch_now < self.epoch_length * self.special_aug_ratio and self.mosaic and self.rand() < self.mosaic_prob:
            lines = random.sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            random.shuffle(lines)
            image, rbox  = self.get_random_data_with_Mosaic(lines, self.input_shape)

        elif self.epoch_now < self.epoch_length * self.special_aug_ratio and self.mixup and self.rand() < self.mixup_prob:
            lines            = random.sample(self.annotation_lines, 2)
            if self.train:
                image_1, rbox_1      = self.get_random_data(lines[0], self.input_shape)
                image_2, rbox_2      = self.get_random_data(lines[1], self.input_shape)
            else:
                image_1, rbox_1      = self.get_data(lines[0], self.input_shape)
                image_2, rbox_2      = self.get_data(lines[1], self.input_shape)
            image, rbox      = self.get_random_data_with_MixUp(image_1, rbox_1, image_2, rbox_2)    

        else:
            if self.train:
                image, rbox      = self.get_random_data(self.annotation_lines[index], self.input_shape)
            else:
                image, rbox      = self.get_data(self.annotation_lines[index], self.input_shape)

        image       = np.transpose(np.array(image, dtype=np.float32)/255.0, (2, 0, 1))
        rbox        = np.array(rbox, dtype=np.float32)
        
        #---------------------------------------------------#
        #   对真实框进行预处理
        #---------------------------------------------------#
        nL          = len(rbox)
        labels_out  = np.zeros((nL, 7))
        if nL:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            rbox[:, [0, 2]] = rbox[:, [0, 2]] / self.input_shape[1]
            rbox[:, [1, 3]] = rbox[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   调整顺序，符合训练的格式
            #   labels_out中序号为0的部分在collect时处理
            #---------------------------------------------------#
            labels_out[:, 1]  = rbox[:, -1]
            labels_out[:, 2:] = rbox[:, :5]
        return image, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_data(self, annotation_line, input_shape):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image_data   = cv2.imread(line[0])
        image_data   = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih, _  = image_data.shape
        h, w    = input_shape
        #------------------------------#
        #   获得标注框
        #------------------------------#
        box     = np.array([np.array(list(map(float,box.split(',')))) for box in line[1:]])
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        dx      = (w-nw)//2
        dy      = (h-nh)//2
        #---------------------------------------------------#
        #   检查分辨率，保证最后的图像符合模型输入要求
        #---------------------------------------------------#
        if (w-nw)%2 == 1:
            nw = nw + 1
        if (h-nh)%2 == 1:
            nh = nh + 1
        #---------------------------------#
        #   使用双三次插值缩放图像
        #---------------------------------#
        image_data = cv2.resize(image_data, [nw,nh], cv2.INTER_CUBIC)
        #---------------------------------#
        #   将图像多余的部分加上灰条
        #---------------------------------#
        if w != h:
            if w > h:
                new_image = np.ones((dy ,nw , 3), dtype=np.uint8) * 128
                image_data = cv2.vconcat([new_image,image_data,new_image])
            else:
                new_image = np.ones((nh, dx, 3), dtype=np.uint8) * 128
                image_data = cv2.hconcat([new_image,image_data,new_image])
        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
            box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
            box[:, [4]] = box[:, [4]] % np.pi
        return image_data.astype(np.float32), box
    
    def get_random_data(self, annotation_line, input_shape, jitter=.1, hue=.1, sat=0.7, val=0.4):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image_data   = cv2.imread(line[0])
        image_data   = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih, _   = image_data.shape
        h, w        = input_shape
        #------------------------------#
        #   获得标注框
        #------------------------------#
        box     = np.array([np.array(list(map(float,box.split(',')))) for box in line[1:]])
        #------------------------------------------#
        #   对图像进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        #------------------------------------------#
        #   对图像进行缩放
        #   在这里图像不进行放大，
        #------------------------------------------#
        scale = self.rand(0.75, 1)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        dx = (w - nw)//2
        dy = (h - nh)//2
        #---------------------------------------------------#
        #   检查分辨率，保证最后的图像符合模型输入要求
        #---------------------------------------------------#
        if (w-nw)%2:
            nw = nw + 1
        if (h-nh)%2:
            nh = nh + 1
        #---------------------------------#
        #   使用双三次插值缩放图像
        #---------------------------------#
        image_data = cv2.resize(image_data, [nw,nh], cv2.INTER_CUBIC)
        #----------------------------------------------------------#
        #   将图像多余的部分加上灰条
        #   由于rand函数不可能返回1，所以这里一定是要加灰条的
        #----------------------------------------------------------#
        new_image = np.ones((dy, nw, 3), dtype = np.uint8) * 128 
        image_data = cv2.vconcat([new_image,image_data,new_image])
        new_image = np.ones((h, dx, 3), dtype=np.uint8) * 128 
        image_data = cv2.hconcat([new_image,image_data,new_image])
        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box) > 0:
            box[:, [0]] = box[:, [0]]*nw/iw + dx
            box[:, [2]] = box[:, [2]]*nw/iw
            box[:, [1]] = box[:, [1]]*nh/ih + dy
            box[:, [3]] = box[:, [3]]*nh/ih
            box[:, [4]] = box[:, [4]] % np.pi
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip:
            flip_label = int(self.rand()<.5)
            image_data = cv2.flip(image_data, flip_label)
            if flip_label:
                box[:, [0]] = w - box[:, [0]]
                box[:, [4]] = - box[:, [4]] + np.pi
            else:
                box[:, [1]] = h - box[:, [1]]
                box[:, [4]] = - box[:, [4]] + np.pi
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype

        x       = np.arange(0, 256, dtype = r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data.astype(np.float32), box

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.1, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        rbox_datas  = []

        for line in annotation_line:
            #---------------------------------#
            #   直接使用写好的部分增强图像
            #---------------------------------#
            image_data, rbox_data = self.get_random_data(line, input_shape, jitter, hue, sat, val)
            image_datas.append(image_data)
            rbox_datas.append(rbox_data)
        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :]   = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:h, :cutx, :]  = image_datas[1][cuty:h, :cutx, :]
        new_image[cuty:h, cutx:w, :] = image_datas[2][cuty:h, cutx:w, :]
        new_image[:cuty, cutx:w, :]  = image_datas[3][:cuty, cutx:w, :]
        new_image = new_image.astype(np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        new_image       = cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV)
        hue, sat, val   = cv2.split(new_image)
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #--------------------------------------------------------------------------#
        #   检查框的大小是否超过裁减点
        #   由于本质上是旋转框，不做相关矫正，中心超过裁剪位置直接判定为不是目标
        #--------------------------------------------------------------------------#
        new_rboxes = []
        rbox_data = rbox_datas[0]
        for rbox in rbox_data:
            if rbox[0] < cutx and  rbox[1] < cuty:
                new_rboxes.append(rbox)
        rbox_data = rbox_datas[1]
        for rbox in rbox_data:
            if rbox[0] < cutx and rbox[1] > cuty:
                new_rboxes.append(rbox)
        rbox_data = rbox_datas[2]
        for rbox in rbox_data:
            if rbox[0] > cutx and rbox[1] > cuty:
                new_rboxes.append(rbox)
        rbox_data = rbox_datas[3]
        for rbox in rbox_data:
            if rbox[0] > cutx and rbox[1] < cuty:
                new_rboxes.append(rbox)  
        new_rboxes = np.array(new_rboxes)

        return new_image.astype(np.float32), new_rboxes

    def get_random_data_with_MixUp(self, image_1, rbox_1, image_2, rbox_2):
        new_image = cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)
        if len(rbox_1) == 0:
            new_rboxes = rbox_2
        elif len(rbox_2) == 0:
            new_rboxes = rbox_1
        else:
            new_rboxes = np.concatenate([rbox_1, rbox_2], axis=0)
        return new_image.astype(np.float32), new_rboxes
    
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes
