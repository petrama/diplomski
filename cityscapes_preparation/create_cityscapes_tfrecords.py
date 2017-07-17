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



id_to_label={0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25:
    12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255, 31: 16, 32: 17, 33: 18, -1: -1}
hasInst={0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False, 10: False, 11: False, 12: False, 13: False, 14: False, 15: False, 16: False, 17: False, 18: False, 19: False, 20: False, 21: False, 22: False, 23: False, 24:
    True, 25: True, 26: True, 27: True, 28: True, 29: True, 30: True, 31: True, 32: True, 33: True, -1: False}

cx_start,cx_end=0,2048
cy_start,cy_end=0,900

img_width,img_height=640,288

ids = np.array(sorted(id_to_label.keys()))


def get_instance_centers(inst, lab, instance_objects_only=True):
    y, x = np.mgrid[0:inst.shape[0], 0:inst.shape[1]]
    grid = np.stack((y, x), axis=-1).astype(float)
    instance_vectors = np.zeros_like(grid, dtype=float)
    instance_mask = np.zeros_like(inst)

    h, w = inst.shape
    instance_ids = np.unique(inst)

    centers = []

    for instance_id in instance_ids:
        y, x = np.where(inst == instance_id)
        labels_instance = lab[y, x]
        l = np.unique(labels_instance)[0]

        if instance_objects_only and hasInst[l]==False:
            continue

        y_center = np.mean(y)
        x_center = np.mean(x)
        instance_vectors[inst == instance_id] = grid[inst == instance_id] - np.array([y_center, x_center])
        instance_mask[inst == instance_id] = 1

        centers.append((y_center, x_center))

    centers = np.array(centers)
    return centers, instance_vectors, instance_mask


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_bboxes_and_labels_from_label_and_instance_ids(inst, label):
    instance_ids = np.unique(inst)

    assert inst.shape==label.shape

    h,w=inst.shape

    bboxes = []
    llabels = []

    for instance_id in instance_ids:
        y, x = np.where(inst == instance_id)
        y_min, y_max = np.min(y), np.max(y)
        x_min, x_max = np.min(x), np.max(x)

        labels_instance = label[y, x]
        l = np.unique(labels_instance)[0]

        if hasInst[l] == False:
            continue

        #[ymin, xmin, ymax, xmax].append([y_min/h, x_min/w, y_max/h, x_max/w])
        bboxes.append(np.array([[y_min/h, x_min/w, y_max/h, x_max/w]]))
        llabels.append(l)

    llabels = np.array(llabels)
    if bboxes:
        bboxes = np.vstack(bboxes)
    else:
        bboxes = np.array([[]])

    return bboxes, llabels


def create_tfrecord_old(rgb, label_map, instance_map, vector_to_centroid,instance_mask,img_name,bboxes,bbox_labels,save_dir):
  rows = rgb.shape[0]
  cols = rgb.shape[1]
  depth = rgb.shape[2]

  shape=[rows,cols,depth]

  filename = os.path.join(save_dir , img_name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  rgb_str = rgb.tostring()
  labels_str = label_map.tostring()

  instance_str=instance_map.tostring()
  vec_str=vector_to_centroid.tostring()
  mask_str=instance_mask.tostring()


  xmin = []
  ymin = []
  xmax = []
  ymax = []
  for b in bboxes:
      assert len(b) == 4
      # pylint: disable=expression-not-assigned
      [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(rows),
      'image/width': _int64_feature(cols),
      'image/channels': _int64_feature(depth),
      'image/shape':_int64_feature(shape),
      'image/object/bbox/xmin': float_feature(xmin),
      'image/object/bbox/xmax': float_feature(xmax),
      'image/object/bbox/ymin': float_feature(ymin),
      'image/object/bbox/ymax': float_feature(ymax),
      'image/object/bbox/label': _int64_feature(bbox_labels.tolist()),
      'image/object/bbox/difficult': _int64_feature(np.zeros_like(bbox_labels).tolist()),
      'image/object/bbox/truncated': _int64_feature(np.zeros_like(bbox_labels).tolist()),
      'image': _bytes_feature(rgb_str),
      'labels': _bytes_feature(labels_str),
      'instances': _bytes_feature(instance_str),
      'vector_centroid': _bytes_feature(vec_str),
      'instance_mask': _bytes_feature(mask_str),
  }))


  writer.write(example.SerializeToString())
  writer.close()



def create_tfrecord(rgb, label_map, instance_map, vector_to_centroid,instance_mask,img_name,bboxes,bbox_labels,save_dir):
  rows = rgb.shape[0]
  cols = rgb.shape[1]
  depth = rgb.shape[2]

  filename = os.path.join(save_dir , img_name + '.tfrecords')
  writer = tf.python_io.TFRecordWriter(filename)
  rgb_str = rgb.tostring()
  labels_str = label_map.tostring()

  instance_str=instance_map.tostring()
  vec_str=vector_to_centroid.tostring()
  mask_str=instance_mask.tostring()



  xmin = []
  ymin = []
  xmax = []
  ymax = []
  for b in bboxes:
      #assert len(b) == 4
      # pylint: disable=expression-not-assigned
      [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

  example = tf.train.Example(features=tf.train.Features(feature={
      'height': _int64_feature(rows),
      'width': _int64_feature(cols),
      'depth': _int64_feature(depth),
      'img_name': _bytes_feature(img_name.encode()),
      'rgb': _bytes_feature(rgb_str),
      'labels': _bytes_feature(labels_str),
      'instances':_bytes_feature(instance_str),
      'vector_centroid':_bytes_feature(vec_str),
      'instance_mask':_bytes_feature(mask_str),
      'image/object/bbox/xmin': float_feature(xmin),
      'image/object/bbox/xmax': float_feature(xmax),
      'image/object/bbox/ymin': float_feature(ymin),
      'image/object/bbox/ymax': float_feature(ymax),
      'image/object/bbox/label': _int64_feature(bbox_labels.tolist()),
      'image/object/bbox/difficult': _int64_feature(np.zeros_like(bbox_labels).tolist()),
      'image/object/bbox/truncated': _int64_feature(np.zeros_like(bbox_labels).tolist()),

                                                }))
  writer.write(example.SerializeToString())
  writer.close()




def traverse(cityscapes_root, subset, ann_folder, output_root, rgb_folder,
             instance_filename_template="{}_gtFine_instanceIds.png",
             labelids_filename_template="{}_gtFine_labelIds.png",
             gt_filename_template="{}_leftImg8bit.png"):
    ann_path = os.path.join(cityscapes_root, ann_folder, subset)
    rgb_path=os.path.join(cityscapes_root,rgb_folder,subset)

    output_folder = os.path.join(cityscapes_root, output_root, subset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for (dirpath, dirnames, filenames) in tqdm(os.walk(rgb_path)):
        for filename in filenames:



            splitted_name = filename.split('_')


            name = '_'.join((splitted_name[0], splitted_name[1], splitted_name[2]))

            tfrec_filename = os.path.join(output_folder, name + '.tfrecords')
            if os.path.isfile(tfrec_filename):
                print('skipping: ',filename)
                continue

            # print('name',name)
            instance_filename = instance_filename_template.format(name)
            labelids_filename = labelids_filename_template.format(name)
            gt_filename=gt_filename_template.format(name)

            instance_filepath = os.path.join(ann_path, splitted_name[0], instance_filename)
            labelids_filepath = os.path.join(ann_path, splitted_name[0], labelids_filename)
            gt_filepath=os.path.join(rgb_path,splitted_name[0],gt_filename)

            rgb = ski.data.load(gt_filepath)
            rgb_res = np.ascontiguousarray(rgb[cy_start:cy_end, cx_start:cx_end, :])


            labels = ski.data.load(labelids_filepath)
            labels_res = np.ascontiguousarray(labels[cy_start:cy_end, cx_start:cx_end])

            instances = ski.data.load(instance_filepath)
            inst_res = np.ascontiguousarray(instances[cy_start:cy_end, cx_start:cx_end])

            bboxes,bbox_labels=get_bboxes_and_labels_from_label_and_instance_ids(inst_res,labels_res)

            rgb_res = ski.transform.resize(
                rgb_res, (img_height, img_width), order=3, preserve_range=True).astype(np.uint8)

            labels_res = ski.transform.resize(labels_res, (img_height, img_width),
                                              order=0, preserve_range=True).astype(np.uint8)


            inst_res = ski.transform.resize(inst_res, (img_height, img_width),
                                            order=0, preserve_range=True).astype(np.uint16)

            _,centers_res,mask = get_instance_centers(inst_res, labels_res)

            centers_res=centers_res.astype(np.float32)
            mask=mask.astype(np.uint8)

            #print(rgb_res.shape)



            train_labels_res=np.vectorize(id_to_label.get)(labels_res).astype(np.uint8)
            if bbox_labels.any():
                bbox_labels_res = np.vectorize(id_to_label.get)(bbox_labels).astype(np.uint8)
            else:
                bbox_labels_res = bbox_labels

            create_tfrecord(rgb_res,train_labels_res,inst_res,centers_res,mask,name,bboxes,bbox_labels_res,save_dir=output_folder)


def generate_one(cityscapes_root,ann_folder,rgb_folder,subset,output_folder):
    names = 'lindau_000017_000019',\
            'lindau_000018_000019',\
            'lindau_000019_000019',\
            'lindau_000021_000019',\
            'lindau_000032_000019',\
            'lindau_000040_000019', \
            'lindau_000045_000019'
    for name in names:
        splitted_name=name.split('_')
        instance_filename_template = "{}_gtFine_instanceIds.png"
        labelids_filename_template = "{}_gtFine_labelIds.png"
        gt_filename_template = "{}_leftImg8bit.png"

        ann_path = os.path.join(cityscapes_root, ann_folder, subset)
        rgb_path = os.path.join(cityscapes_root, rgb_folder, subset)

        tfrec_filename = os.path.join('/home/pmarce/datasets/cityscapes/proba/', name + '.tfrecords')


        # print('name',name)
        instance_filename = instance_filename_template.format(name)
        labelids_filename = labelids_filename_template.format(name)
        gt_filename = gt_filename_template.format(name)

        instance_filepath = os.path.join(ann_path, splitted_name[0], instance_filename)
        labelids_filepath = os.path.join(ann_path, splitted_name[0], labelids_filename)
        gt_filepath = os.path.join(rgb_path, splitted_name[0], gt_filename)

        rgb = ski.data.load(gt_filepath)
        rgb_res = np.ascontiguousarray(rgb[cy_start:cy_end, cx_start:cx_end, :])

        labels = ski.data.load(labelids_filepath)
        labels_res = np.ascontiguousarray(labels[cy_start:cy_end, cx_start:cx_end])

        instances = ski.data.load(instance_filepath)
        inst_res = np.ascontiguousarray(instances[cy_start:cy_end, cx_start:cx_end])

        bboxes, bbox_labels = get_bboxes_and_labels_from_label_and_instance_ids(inst_res, labels_res)

        rgb_res = ski.transform.resize(
            rgb_res, (img_height, img_width), order=3, preserve_range=True).astype(np.uint8)

        labels_res = ski.transform.resize(labels_res, (img_height, img_width),
                                          order=0, preserve_range=True).astype(np.uint8)

        inst_res = ski.transform.resize(inst_res, (img_height, img_width),
                                        order=0, preserve_range=True).astype(np.uint16)

        _, centers_res, mask = get_instance_centers(inst_res, labels_res)

        centers_res = centers_res.astype(np.float32)
        mask = mask.astype(np.uint8)

        # print(rgb_res.shape)



        train_labels_res = np.vectorize(id_to_label.get)(labels_res).astype(np.uint8)
        if bbox_labels.any():
            bbox_labels_res = np.vectorize(id_to_label.get)(bbox_labels).astype(np.uint8)
        else:
            bbox_labels_res = bbox_labels

        create_tfrecord(rgb_res, train_labels_res, inst_res, centers_res, mask, name, bboxes, bbox_labels_res,
                        save_dir=output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cityscapes_root', help="Folder with fine_annotations and leftImg8bit",default='/home/pmarce/datasets/cityscapes')
    parser.add_argument('--subset', help="train or val, subfolders",default='train')
    parser.add_argument('--output_folder', help="folder where generated xmls will be saved",default='/home/pmarce/datasets/cityscapes/bbox_tfrecords/')

    parser.add_argument('--fine_annotations_folder', default='fine_annotations',
                        help="subfolder inside cityscapes_root")
    parser.add_argument('--rgb_folder', default='leftImg8bit')
    args = parser.parse_args()

    #create_file_list_p(args.cityscapes_root, args.rgb_folder, args.subset)
    #traverse(args.cityscapes_root,args.subset,args.fine_annotations_folder,args.output_folder,args.rgb_folder)
    generate_one(args.cityscapes_root,args.fine_annotations_folder,args.rgb_folder,'val','/home/pmarce/datasets/cityscapes/proba/')