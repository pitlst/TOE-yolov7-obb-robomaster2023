import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .utils_dbox import DecodeBox

#---------------------------------------------------------#
#   计算旋转框的IOU
#   输入框结构：x,y,w,h,angle
#---------------------------------------------------------#
def iou_rotate_calculate(boxes1, boxes2):
    ious = []
    for box1 in boxes1:
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4] * 180 / np.pi)
        for box2 in boxes2:
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4] * 180 / np.pi)
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                int_area = cv2.contourArea(order_pts)
                inter = int_area * 1.0 / (box1[2] * box1[3] + box2[2] * box2[3] - int_area)
                temp_ious.append(inter)
        ious.append(temp_ious)
    return ious


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)

        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, cuda, \
                    max_boxes=100, confidence=0.05, nms_iou=0.5, map_iou=0.3, MINOVERLAP=0.5, eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.map_iou            = map_iou
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.bbox_util          = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write("# start_map")
                f.write("\n")
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            #------------------------------#
            #   更新自己的网络
            #------------------------------#
            self.net = model_eval

            temp_precision = []
            temp_recall = []
            for i in range(self.num_classes):
                temp_precision.append([])
                temp_recall.append([])

            print("Get map.")

            for annotation_line in tqdm(self.val_lines):
                ground_truth = []
                detection_results = []

                line = annotation_line.split()
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                image = cv2.imread(line[0])
                #------------------------------#
                #   获得真实框
                #------------------------------#
                rboxes = np.array(
                    [np.array(list(map(float, box.split(',')))) for box in line[1:]])
                #------------------------------#
                #   保存真实框
                #------------------------------#
                for rbox in rboxes:
                    xc, yc, w, h, angle, turth_class = rbox
                    ground_truth.append([xc, yc, w, h, angle, turth_class])
                #------------------------------#
                #   检查真实框
                #------------------------------#
                if len(ground_truth) == 0:
                    print('Error: No ground-truth files found!')
                    break
                #------------------------------#
                #   获得预测框
                #------------------------------#
                image_shape = np.array(np.shape(image)[0:2])
                #---------------------------------------------------------#
                #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
                #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
                #---------------------------------------------------------#
                image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #---------------------------------------------------------#
                #   也可以直接resize进行识别
                #---------------------------------------------------------#
                image_data = cv2.resize(image, self.input_shape, cv2.INTER_CUBIC)
                #---------------------------------------------------------#
                #   添加上batch_size维度
                #---------------------------------------------------------#
                image_data = np.expand_dims(np.transpose(np.array(image_data, dtype='float32')/255.0, (2, 0, 1)), 0)

                with torch.no_grad():
                    images = torch.from_numpy(image_data)
                    if self.cuda:
                        images = images.cuda()
                    #---------------------------------------------------------#
                    #   将图像输入网络当中进行预测
                    #---------------------------------------------------------#
                    outputs = self.net(images)
                    outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,\
                                                                image_shape, False, conf_thres=self.confidence, nms_thres=self.nms_iou)[0]

                if results is None:
                    print('Error: No detect results found!')
                    continue

                top_label = np.array(results[:, 7], dtype='int32')
                top_conf = results[:, 5] * results[:, 6]
                top_rboxes = results[:, :5]
                #------------------------------#
                #   保存预测框
                #------------------------------#
                for i, predicted_class in list(enumerate(top_label)):
                    score = top_conf[i]
                    #------------------------------#
                    #   在这里做置信度筛选
                    #------------------------------#
                    if score >= self.confidence:
                        rbox = top_rboxes[i]
                        xc, yc, w, h, angle = rbox
                        detection_results.append([xc, yc, w, h, angle, predicted_class, score])

                ground_truth = np.array(ground_truth)
                detection_results = np.array(detection_results)
                #----------------------------------------------------#
                #   对于检测框只取置信度前100，一点小小的作弊手段
                #   限制这个能提高在极端不准确情况下的map
                #   总结：屁用没有
                #----------------------------------------------------#
                if detection_results.shape[0] > self.max_boxes:
                    detection_results = detection_results[np.argsort(detection_results[:,6])[0:100]]
                for i_ in range(self.num_classes):
                    #----------------------------------------#
                    #   把对应类别的预测框和真实框提取出来
                    #----------------------------------------#
                    gt_i_ = ground_truth[np.where(ground_truth[:, 5] == i_)]
                    dr_i_ = detection_results[np.where(detection_results[:, 5] == i_)]
                    if gt_i_.shape[0] != 0 and dr_i_.shape[0] != 0:
                        #----------------------------------#
                        #   计算该类别真实框与预测框的iou
                        #----------------------------------#
                        temp_ious = iou_rotate_calculate(gt_i_[:, 0:5], dr_i_[:, 0:5])
                        #------------------------------#
                        #   统计检测正确的个数
                        #------------------------------#
                        tp = 0
                        for iou_ in temp_ious:
                            if len(iou_) == 0:
                                continue
                            if np.sum(np.array(iou_) > self.map_iou) != 0:
                                tp += 1
                        #------------------------------#
                        #   计算该类别的准确率和召回率
                        #------------------------------#
                        temp_precision[i_].append(tp / len(dr_i_))
                        temp_recall[i_].append(tp / len(gt_i_))

            print("Calculate Map.")
            #------------------------------#
            #   计算各类别map
            #------------------------------#
            t_map = []
            for i_ in range(self.num_classes):
                final_point = [[0], [1]]
                for i__ in range(len(temp_recall[i_])):
                    if temp_recall[i_][i__] not in final_point[0]:
                        final_point[0].append(temp_recall[i_][i__])
                        final_point[1].append(temp_precision[i_][i__])
                    else:
                        index_same = final_point[0].index(temp_recall[i_][i__])
                        if temp_precision[i_][i__] > final_point[1][index_same]:
                            final_point[1][index_same] = temp_precision[i_][i__]
                #--------------------------------------#
                #   这里进行离散点积分时，y在前，x在后
                #--------------------------------------#
                tm = scipy.integrate.simpson(final_point[1], final_point[0])
                t_map.append(tm)
            temp_map = np.mean(t_map)

            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")

            plt.figure()
            plt.plot(self.epoches, self.maps, 'red',
                     linewidth=2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s' % str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
