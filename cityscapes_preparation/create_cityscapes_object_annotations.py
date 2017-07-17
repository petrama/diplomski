import os
import pickle
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.data
from tqdm import tqdm
import xml.etree.cElementTree as ET
import skimage.transform
import cv2

import argparse

id_to_label = {
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
}

ids = np.array(sorted(id_to_label.keys()))


def get_bboxes_and_labels_from_label_and_instance_ids(instance_ids_path, label_ids_path):
    inst = ski.data.load(instance_ids_path)
    label = ski.data.load(label_ids_path)
    h, w = inst.shape
    instance_ids = np.unique(inst)
    unique_ids = np.unique(label)

    bboxes = []
    llabels = []

    for instance_id in instance_ids:
        y, x = np.where(inst == instance_id)
        y_min, y_max = np.min(y), np.max(y)
        x_min, x_max = np.min(x), np.max(x)

        labels_instance = label[y, x]
        l = np.unique(labels_instance)[0]

        if l not in ids:
            continue

        bboxes.append(np.array([[y_min, x_min, y_max, x_max]]))
        llabels.append(l)

    llabels = np.array(llabels)
    if bboxes:
        bboxes = np.vstack(bboxes)
    else:
        bboxes = np.array([[]])

    return bboxes, llabels


def write_object_annotations_to_xml(file_name,
                                    instance_path,
                                    label_path,
                                    subset,
                                    output_path, img_width=2048, img_height=1024, img_depth=3,
                                    ):
    boxes, labels = get_bboxes_and_labels_from_label_and_instance_ids(instance_path, label_path)

    name_splitted = file_name.split('_')
    citystr = name_splitted[0]

    output_folder = os.path.join(output_path, citystr)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    xml_name = os.path.join(output_path, file_name + '_bboxes.xml')

    annotation = ET.Element("annotation")

    filename = ET.SubElement(annotation, "file_name").text = file_name
    subset = ET.SubElement(annotation, "subset").text = subset
    city = ET.SubElement(annotation, "city").text = citystr

    size = ET.SubElement(annotation, "size")

    width = ET.SubElement(size, "width").text = str(img_width)
    height = ET.SubElement(size, "height").text = str(img_height)
    depth = ET.SubElement(size, "depth").text = str(img_depth)

    for l, b in zip(labels, boxes):
        obj = ET.SubElement(annotation, 'object')

        name = ET.SubElement(obj, 'name').text = id_to_label[l]
        label = ET.SubElement(obj, 'label_id').text = str(l)
        trun = ET.SubElement(obj, 'truncated').text = "0"
        diff = ET.SubElement(obj, 'difficult').text = "0"
        bb = ET.SubElement(obj, 'bndbox')

        xmin = ET.SubElement(bb, 'xmin').text = str(b[1])
        ymin = ET.SubElement(bb, 'ymin').text = str(b[0])
        xmax = ET.SubElement(bb, 'xmax').text = str(b[3])
        ymax = ET.SubElement(bb, 'ymax').text = str(b[2])

    tree = ET.ElementTree(annotation)
    tree.write(xml_name)


def traverse(cityscapes_root, subset, ann_folder, output_root, rgb_folder,
             instance_filename_template="{}_gtFine_instanceIds.png",
             labelids_filename_template="{}_gtFine_labelIds.png",
             gt_imagename="{}_leftImg8bit.png"):
    ann_path = os.path.join(cityscapes_root, ann_folder, subset)

    for (dirpath, dirnames, filenames) in tqdm(os.walk(ann_path)):
        for filename in filenames:

            splitted_name = filename.split('_')
            city = splitted_name[0]
            # print('city:',city)
            output_folder = os.path.join(cityscapes_root, output_root, subset, city)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            name = '_'.join((splitted_name[0], splitted_name[1], splitted_name[2]))
            # print('name',name)
            instance_filename = instance_filename_template.format(name)
            labelids_filename = labelids_filename_template.format(name)

            instance_filepath = os.path.join(ann_path, splitted_name[0], instance_filename)
            labelids_filepath = os.path.join(ann_path, splitted_name[0], labelids_filename)

            write_object_annotations_to_xml(name, instance_filepath, labelids_filepath, subset, output_folder)


def create_file_list_p(cityscapes_root, rgb_folder, subset):
    path = os.path.join(cityscapes_root, rgb_folder, subset)
    cities=os.listdir(path)
    for city in cities:
        citypath=os.path.join(path,city)
        images=os.listdir(citypath)
        for image in images:
            p=os.path.join(city,'_'.join(image.split('_')[:-1]))
            print(p)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cityscapes_root', help="Folder with fine_annotations and leftImg8bit")
    parser.add_argument('subset', help="train or val, subfolders")
    parser.add_argument('output_folder', help="folder where generated xmls will be saved")

    parser.add_argument('--fine_annotations_folder', default='fine_annotations',
                        help="subfolder inside cityscapes_root")
    parser.add_argument('--rgb_folder', default='leftImg8bit')
    args = parser.parse_args()

    #create_file_list_p(args.cityscapes_root, args.rgb_folder, args.subset)
    traverse(args.cityscapes_root,args.subset,args.fine_annotations_folder,args.output_folder,args.rgb_folder)
