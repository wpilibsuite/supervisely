# coding: utf-8

import os

import supervisely_lib as sly
from supervisely_lib.nn.hosted.inference_single_image import SingleImageInferenceBase, GPU_DEVICE
from supervisely_lib.nn.hosted.inference_batch import BatchInferenceApplier
from supervisely_lib.nn.hosted.inference_modes import InfModeFullImage, CONFIDENCE
from supervisely_lib.nn.config import JsonConfigValidator

from common import construct_model, infer_rectangles
import config as config_lib


class ObjectDetectionSingleImageApplier(SingleImageInferenceBase):
    def __init__(self, task_model_config=None):
        sly.logger.info('TF object detection inference init started.')
        super().__init__(task_model_config)
        self.confidence_thresh = self._config['min_confidence_threshold']
        sly.logger.info('TF object detection inference init done.')

    @staticmethod
    def get_default_config():
        return {
            GPU_DEVICE: 0,
            'min_confidence_threshold': 0.5,
            'confidence_tag_name': CONFIDENCE
        }

    @property
    def train_classes_key(self):
        return config_lib.train_classes_key()

    @property
    def class_title_to_idx_key(self):
        return config_lib.class_to_idx_config_key()

    def _load_train_config(self):
        self.confidence_tag_meta = sly.TagMeta(self._config['confidence_tag_name'], sly.TagValueType.ANY_NUMBER)
        super()._load_train_config()

    def _validate_model_config(self, config):
        JsonConfigValidator().validate_inference_cfg(config)

    def _model_out_tags(self):
        tag_meta_dict = sly.TagMetaCollection()
        return tag_meta_dict.add(self.confidence_tag_meta)

    def _construct_and_fill_model(self):
        super()._construct_and_fill_model()
        self.device_ids = sly.env.remap_gpu_devices([self._config[GPU_DEVICE]])
        self.detection_graph, self.session = construct_model(sly.TaskPaths.MODEL_DIR)
        sly.logger.info('Weights are loaded.')

    def inference(self, img, ann):
        labels = infer_rectangles(img,
                                  self.detection_graph,
                                  self.session,
                                  self.out_class_mapping,
                                  self.confidence_thresh,
                                  self.confidence_tag_meta)
        return sly.Annotation(ann.img_size, labels=labels)


def main():
    single_image_applier = ObjectDetectionSingleImageApplier()
    default_inference_mode_config = InfModeFullImage.make_default_config(model_result_suffix='_detection')
    dataset_applier = BatchInferenceApplier(single_image_inference=single_image_applier,
                                            default_inference_mode_config=default_inference_mode_config,
                                            config_validator=JsonConfigValidator())
    dataset_applier.run_inference()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly.add_default_logging_into_file(sly.logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('TF_OBJECT_DETECTION_INFERENCE', main)
