# coding: utf-8
import os
import os.path as osp

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from supervisely_lib import logger
from supervisely_lib.nn import raw_to_labels

from object_detection import exporter
from object_detection.protos import pipeline_pb2


def create_detection_graph(model_dirpath):
    fpath = osp.join(model_dirpath, 'model.pb')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(fpath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    logger.info('Restored model weights from training.')
    return detection_graph


def freeze_graph(input_type,
                 pipeline_config_path,
                 trained_checkpoint_prefix,
                 output_directory,
                 input_shape=None):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    if input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in input_shape.split(',')
        ]
    else:
        input_shape = None
    exporter.export_inference_graph(input_type, pipeline_config,
                                    trained_checkpoint_prefix,
                                    output_directory, input_shape)


def inverse_mapping(mapping):
    return {v: k for k, v in mapping.items()}


def get_scope_vars(detection_graph):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return detection_boxes, detection_scores, detection_classes, num_detections, image_tensor


def construct_model(model_dir):
    if 'model.pb' not in os.listdir(model_dir):
        logger.info('Freezing training checkpoint!')
        freeze_graph('image_tensor',
                     model_dir + '/model.config',
                     model_dir + '/model_weights/model.ckpt',
                     model_dir)
    detection_graph = create_detection_graph(model_dir)
    session = tf.Session(graph=detection_graph)
    return detection_graph, session


def infer_rectangles(image, graph, session, out_class_mapping, confidence_threshold, confidence_tag_meta):
    image_np_expanded = np.expand_dims(image, axis=0)
    detection_boxes, detection_scores, detection_classes, num_detections, image_tensor = \
        get_scope_vars(graph)

    net_out = session.run([detection_boxes, detection_scores, detection_classes, num_detections],
                               feed_dict={image_tensor: image_np_expanded})
    network_prediction = raw_to_labels.DetectionNetworkPrediction(
        boxes=net_out[0], scores=net_out[1], classes=net_out[2])
    res_figures = raw_to_labels.detection_preds_to_sly_rects(out_class_mapping,
                                                             network_prediction,
                                                             image.shape,
                                                             confidence_threshold,
                                                             confidence_tag_meta)
    return res_figures
