import numpy as np
import cv2

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
        
def download_weights(phi, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        "l" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_backbone_weights.pth',
        "x" : 'https://github.com/bubbliiiing/yolov7-pytorch/releases/download/v1.0/yolov7_x_backbone_weights.pth',
    }
    url = download_urls[phi]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)

def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y

#---------------------------------------------------------#
#   绘制对应的box
#---------------------------------------------------------#
def draw_contours_and_putText(img, x, y, w, h, angle, color, text):
    box = []
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)
    #ag = (180 - float(angle))/180*math.pi
    ag = np.pi - float(angle)

    x1 = x-w/2*np.cos(ag)+h/2*np.sin(ag)
    y1 = y+w/2*np.sin(ag)+h/2*np.cos(ag)
    box.append([[x1,y1]])

    x2 = x-w/2*np.cos(ag)-h/2*np.sin(ag)
    y2 = y+w/2*np.sin(ag)-h/2*np.cos(ag)
    box.append([[x2,y2]])

    x3 = x+w/2*np.cos(ag)-h/2*np.sin(ag)
    y3 = y-w/2*np.sin(ag)-h/2*np.cos(ag)
    box.append([[x3,y3]])

    x4 = x+w/2*np.cos(ag)+h/2*np.sin(ag)
    y4 = y-w/2*np.sin(ag)+h/2*np.cos(ag)
    box.append([[x4,y4]])

    box = np.array(box,dtype='int32')

    cv2.drawContours(img, [box], 0, color, 2)
    cv2.putText(img, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    return img