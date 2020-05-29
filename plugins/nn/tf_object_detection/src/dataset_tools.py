# coding: utf-8
import tensorflow as tf

import numpy as np

from tensorflow.python.framework import dtypes
from object_detection.core import standard_fields as fields

from supervisely_lib.annotation.annotation import Annotation
from supervisely_lib.io.json import load_json_file
from supervisely_lib.imaging import image as sly_image


def get_bbox(mask):
    mask_points = np.where(mask == 1)
    return np.min(mask_points[1]), np.min(mask_points[0]), np.max(mask_points[1]), np.max(mask_points[0])


def load_ann(ann_fpath, classes_mapping, project_meta):
    ann_packed = load_json_file(ann_fpath)
    ann = Annotation.from_json(ann_packed, project_meta)
    # ann.normalize_figures()  # @TODO: enaaaable!
    (h, w) = ann.img_size

    gt_boxes, classes_text, classes = [], [], []
    for label in ann.labels:
        gt = np.zeros((h, w), dtype=np.uint8)  # default bkg
        gt_idx = classes_mapping.get(label.obj_class.name, None)
        if gt_idx is None:
            raise RuntimeError('Missing class mapping (title to index). Class {}.'.format(label.obj_class.name))
        label.geometry.draw(gt, 1)
        if np.sum(gt) > 0:
            xmin, ymin, xmax, ymax = get_bbox(gt)
            gt_boxes.append([ymin / h, xmin / w, ymax / h, xmax / w])
            classes_text.append(label.obj_class.name.encode('utf8'))
            # List of string class name of bounding box (1 per box)
            classes.append(gt_idx)  # List of integer class id of bounding box (1 per box)
    num_boxes = len(gt_boxes)
    gt_boxes = np.array(gt_boxes).astype(np.float32)
    classes = np.array(classes, dtype=np.int64)
    if num_boxes == 0:
        gt_boxes = np.reshape(gt_boxes, [0,4])
    return gt_boxes, classes, np.array([num_boxes]).astype(np.int32)[0]


def read_supervisely_data(sample, classes_mapping, project_meta):
    img_filepath, ann_filepath = sample[0], sample[1]

    def read_image_fn(img_path_bytes):
        return sly_image.read(img_path_bytes.decode('utf-8'))

    image = tf.py_func(read_image_fn, [img_filepath], dtypes.uint8, stateful=False)
    train_tensor = dict()

    def load_ann_fn(x):
        return load_ann(x, classes_mapping=classes_mapping, project_meta=project_meta)
    gt_boxes, classes, num_boxes = tf.py_func(load_ann_fn, [ann_filepath], (dtypes.float32, dtypes.int64, dtypes.int32),
                                              stateful=False)
    train_tensor[fields.InputDataFields.image] = image
    train_tensor[fields.InputDataFields.source_id] = img_filepath
    train_tensor[fields.InputDataFields.key] = img_filepath
    train_tensor[fields.InputDataFields.filename] = img_filepath
    train_tensor[fields.InputDataFields.groundtruth_boxes] = gt_boxes
    train_tensor[fields.InputDataFields.num_groundtruth_boxes] = num_boxes

    train_tensor[fields.InputDataFields.groundtruth_classes] = classes
    train_tensor[fields.InputDataFields.image].set_shape([None, None, 3])
    return train_tensor


def build_dataset(data_dict):
    samples_dataset = tf.data.Dataset.from_tensor_slices(data_dict['samples'])
    samples_dataset = samples_dataset.shuffle(buffer_size=data_dict['sample_cnt'], reshuffle_each_iteration=True)
    samples_dataset = samples_dataset.repeat()

    def sup_decod_fn(x):
        return read_supervisely_data(x, classes_mapping=data_dict['classes_mapping'],
                                     project_meta=data_dict['project_meta'])
    tensor_dataset = samples_dataset.map(sup_decod_fn, num_parallel_calls=1)
    return tensor_dataset.prefetch(1)
