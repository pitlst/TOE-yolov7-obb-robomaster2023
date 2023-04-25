import cv2
import time
import colorsys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from nets.yolo import YoloBody
from utils.utils import get_anchors, get_classes, show_config, sigmoid, draw_contours_and_putText
from utils.utils_dbox import DecodeBox

'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, model_path, classes_path, anchors_path, phi = 'l',anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]], \
                 input_shape = [640, 640], confidence = 0.5, nms_iou = 0.3, letterbox_image = False, cuda = False):
        
        self.model_path        = model_path
        self.classes_path      = classes_path

        self.anchors_path      = anchors_path
        self.anchors_mask      = anchors_mask

        self.input_shape       = input_shape
        self.confidence        = confidence
        self.nms_iou           = nms_iou
        self.letterbox_image   = letterbox_image
        self.cuda              = cuda
        self.phi               = phi

        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   生成模型
        #---------------------------------------------------#
        self.generate()
        #---------------------------------------------------#
        #   显示参数
        #---------------------------------------------------#
        show_config(model_path = self.model_path, classes_path = self.classes_path, anchors_path = self.anchors_path, anchors_mask = self.anchors_mask,\
                    input_shape = self.input_shape, confidence = self.confidence, nms_iou = self.nms_iou, cuda = self.cuda, class_names = self.class_names,\
                    num_classes = self.num_classes, num_anchors = self.num_anchors, letterbox_image = self.letterbox_image)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect(self, image):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #------------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #------------------------------------------------------------#
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #---------------------------------------------------------#
        #   直接resize进行识别
        #---------------------------------------------------------#
        image_data  = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), cv2.INTER_CUBIC)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(np.array(image_data, dtype='float32') / 255.0, (2, 0, 1)), 0)

        with torch.no_grad():
            images_ = torch.from_numpy(image_data)
            if self.cuda:
                images_ = images_.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测
            #---------------------------------------------------------#
            outputs = self.net(images_)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)[0]                                  
        return results

    #---------------------------------------------------#
    #   检测图片并绘制结果
    #---------------------------------------------------#
    def detect_image(self, image):

        results = self.detect(image)                           
        if results is None: 
            return image

        top_label   = np.array(results[:, 7], dtype = 'int32')
        top_conf    = results[:, 5] * results[:, 6]
        top_rboxes  = results[:, :5]
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for rbi in range(len(results)):
            x, y, w, h, angle,  = top_rboxes[rbi]
            label = top_label[rbi]
            image = draw_contours_and_putText(image, x, y, w, h, angle, self.colors[label], top_conf[rbi])
        return image


    #---------------------------------------------------#
    #   统计检测帧率
    #---------------------------------------------------#
    def get_FPS(self, image, test_interval):        
        if self.cuda:
            image_ = torch.from_numpy(image).cuda()

        t1 = time.time()
        for _ in range(test_interval):
            __ = self.detect(image_)            
        t2 = time.time()

        return (t2 - t1) / test_interval

    def detect_heatmap(self, image, heatmap_save_path):
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #---------------------------------------------------------#
        #   resize进行识别
        #---------------------------------------------------------#
        image_data  = cv2.resize(image, (self.input_shape[1], self.input_shape[0]), cv2.INTER_CUBIC)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(np.array(image_data, dtype='float32') / 255.0, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测
            #---------------------------------------------------------#
            outputs = self.net(images)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score      = np.max(sigmoid(sub_output[..., 4]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    #---------------------------------------------------#
    #   生成onnx模型
    #---------------------------------------------------#
    def convert_to_onnx(self, model_path):
        self.generate()

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        print('Onnx model save as {}'.format(model_path))
