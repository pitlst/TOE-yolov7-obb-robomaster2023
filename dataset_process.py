import os
import random
import numpy as np
import xml.etree.ElementTree as ET

#-------------------------------------------------------#
#   指向数据集所在的文件夹
#   默认指向根目录下的data_set文件夹
#-------------------------------------------------------#
data_set_path = 'data_set'
#-------------------------------------------------------#
#   用于分割训练集和测试集的比例系数
#   表示测试集占整体数据集的比重
#-------------------------------------------------------#
rate = 0.2
#-------------------------------------------------------------------#
#   必须要修改，用于生成目标信息
#   与训练和预测所用的classes_path一致即可
#   如果生成的文件里面没有目标信息
#   那么就是因为classes没有设定正确
#-------------------------------------------------------------------#
classes_path = 'model_data/reagent_classes.txt'
#-------------------------------------------------------------------#
#   模型输入大小
#-------------------------------------------------------------------#
input_shape = [640, 640]
#-------------------------------------------------------------------#
#   聚类次数，anchor数量
#-------------------------------------------------------------------#
anchors_count = 30
anchors_num = 9
#---------------------------------------------------------------------#
#   anchors_path    代表先验框对应的txt文件。
#---------------------------------------------------------------------#
anchors_path    = 'model_data/reagent_anchors.txt'

def distance_deal(box, clusters):
    #-------------------------------------------------------------------#
    #   计算一个ground truth边界盒和k个先验框(Anchor)的长宽比值。
    #   参数box: 元组或者数据，代表ground truth的长宽。
    #   参数clusters: 形如(k,2)的numpy数组，其中k是聚类Anchor框的个数
    #   返回：ground truth和每个Anchor框的交并比。
    #-------------------------------------------------------------------#
    box_ = box[0] / box[1]
    if np.isnan(box_):
        raise ValueError()
    cluster_ = clusters[:, 0] / clusters[:, 1]
    if True in np.isnan(cluster_):
        print(clusters)
        print(cluster_)
        raise ValueError()
    iou_ = np.zeros(len(cluster_))
    for i, temp_i in enumerate(cluster_):
        iou_[i] = np.absolute(box_ - temp_i)
    return iou_

def kmeans(boxes, k):
    #-------------------------------------------------------------------#
    #   利用长宽比值进行K-means聚类
    #   参数boxes: 形状为(r, 2)的ground truth框，其中r是ground truth的个数
    #   参数k: Anchor的个数
    #   返回值：形状为(k, 2)的k个Anchor框
    #-------------------------------------------------------------------#
    # 即是上面提到的r
    rows = boxes.shape[0]
    # 距离数组，计算每个ground truth和k个Anchor的距离
    distances = np.empty((rows, k))
    # 上一次每个ground truth"距离"最近的Anchor索引
    last_clusters = np.zeros((rows,))
    # 初始化聚类中心，k个簇，从r个ground truth随机选k个
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    # 对于我们的
    last_temp_chose = []
    # 开始聚类
    while True:
        # 计算每个ground truth和k个Anchor的距离，用长宽比(box,anchor)来计算
        for row in range(rows):
            distances[row] = distance_deal(boxes[row], clusters)
        # 对每个ground truth，选取距离最小的那个Anchor，并存下索引
        nearest_clusters = np.argmin(distances, axis=1)
        # 如果当前每个ground truth"距离"最近的Anchor索引和上一次一样，聚类结束
        if (last_clusters == nearest_clusters).all():
            break
        # 更新簇中心为簇里面所有的ground truth框的均值
        for cluster in range(k):
            temp_chose = (nearest_clusters == cluster)
            # 对于一些特殊的自制数据集，anchor有可能获取不到相近的框，无法计算均值，需要排除
            if not True in temp_chose:
                if not cluster in last_temp_chose:
                    print("\033[1;33;31m注意，该anchor没有与其相近的真实框，请注意更改\033[0m", cluster)
                    last_temp_chose.append(cluster)
                continue
            clusters[cluster] = np.mean(boxes[temp_chose], axis=0)
        # 更新每个ground truth"距离"最近的Anchor索引
        last_clusters = nearest_clusters
    
    return clusters


if __name__ == '__main__':
    if " " in os.path.abspath(data_set_path):
        raise ValueError("数据集存放的文件夹路径与图片名称中不可以存在空格，否则会影响正常的模型训练，请注意修改。")
    
    # 检查数据集
    img_list = os.listdir(data_set_path + '\\img')
    xml_list = os.listdir(data_set_path + '\\xml')
    if len(img_list) == 0 or len(xml_list) == 0:
        raise ValueError("数据集不存在")
    if len(img_list) != len(xml_list):
        raise ValueError("数据集img和xml文件数量对应不上")

    # 获取当前文件的父目录
    father_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".")
    # 分割数据集
    len_list = len(img_list) if len(img_list) < len(xml_list) else len(xml_list)
    len_index = list(range(0, len_list))
    random.shuffle(len_index)
    val_img_index = len_index[:int(len_list * rate)]
    train_img_index = len_index[int(len_list * rate):]

    train_file = open(data_set_path + 'reagent_train.txt', 'w', encoding='utf-8')
    val_file = open(data_set_path + 'reagent_val.txt', 'w', encoding='utf-8')

    with open(classes_path, encoding='utf-8') as f:
        classes = f.readlines()
    classes = [c.strip() for c in classes]

    # 该函数用于读取xml生成
    def gen_txt(file, img_index):
        for i in img_index:
            img_path = father_path + '\\' + data_set_path + '\\img\\' + img_list[i]
            xml_path = father_path + '\\' + data_set_path + '\\xml\\' + xml_list[i]

            write_line = img_path

            tree = ET.parse(open(xml_path, encoding='utf-8'))
            root = tree.getroot()
            
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('robndbox')

                b =  (float(xmlbox.find('cx').text), \
                    float(xmlbox.find('cy').text), \
                    float(xmlbox.find('w').text), \
                    float(xmlbox.find('h').text), \
                    float(xmlbox.find('angle').text))
                
                write_line = write_line +  " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
            write_line = write_line + '\n'
            file.write(write_line)

    print("\033[1;33;34m正在生成数据集文件\033[0m")
    gen_txt(val_file, val_img_index)
    gen_txt(train_file, train_img_index)

    # 计算数据集框的边长
    dataset = []
    for i in len_index:
        xml_path = father_path + '\\' + data_set_path + '\\xml\\' + xml_list[i]
        tree = ET.parse(xml_path)
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        for obj in tree.iter("object"):
            dataset.append([float(obj.findtext("robndbox/w")), float(obj.findtext("robndbox/h"))])
    dataset = np.array(dataset)
    # 进行10次聚类取均值，减少随机种子产生的影响
    final_out = []
    for _ in range(anchors_count):
        out = kmeans(dataset.copy(), anchors_num)
        final_out.append(sorted(out.tolist()))
        print("\033[1;33;34m运行完一次聚类\033[0m")
    out_ = np.mean(np.array(final_out), axis=0)
    out_ = np.rint(out_).tolist()
    # 极其丑陋的写法，但能用就行，回头再改吧
    with open(anchors_path, 'w') as f:
        for i_, o_ in enumerate(out_):
            for i__, o__ in enumerate(o_):
                if i_ != len(out_)-1 or i__ != len(o_)-1:
                    print(int(o__), end=', ', file = f)
                else:
                    print(int(o__), end='', file = f)
    
    print("\033[1;33;34m已计算完anchors并保存\033[0m")
    print("\033[1;33;34m程序运行完成\033[0m")