# coding: utf-8

import json

import os
import os.path as osp

import cv2

import supervisely_lib as sly
import supervisely_lib.nn.dataset
from supervisely_lib import logger
from supervisely_lib import sly_logger
from supervisely_lib.nn.config import JsonConfigValidator
from supervisely_lib.nn.hosted.constants import SETTINGS
from supervisely_lib.nn.hosted.class_indexing import TRANSFER_LEARNING
from supervisely_lib.nn.hosted.trainer import SuperviselyModelTrainer

from tf_config_converter import load_sample_config, save_config, determine_tf_config

import config as config_lib
import custom_train as train


class ObjectDetectionTrainer(SuperviselyModelTrainer):
    @staticmethod
    def get_default_config():
        return {
            'dataset_tags': {
                'train': 'train',
                'val': 'val',
            },
            'batch_size': {
                'train': 1,
                'val': 1,
            },
            'input_size': {
                'width': 300,
                'height': 300,
            },
            'epochs': 2,
            'val_every': 1,
            'lr': 0.001,
            'weights_init_type': TRANSFER_LEARNING,  # CONTINUE_TRAINING,
            'gpu_devices': [0],
            'validate_with_model_eval': False
        }

    def __init__(self):
        super().__init__(default_config=ObjectDetectionTrainer.get_default_config())
        logger.info('Model is ready to train.')
        # To be filled in by dump_model() callback inside train().
        self.saver = None
        self.sess = None

    @property
    def class_title_to_idx_key(self):
        return config_lib.class_to_idx_config_key()

    @property
    def train_classes_key(self):
        return config_lib.train_classes_key()

    def _validate_train_cfg(self, config):
        JsonConfigValidator().validate_train_cfg(config)

    def _determine_model_classes(self):
        super()._determine_model_classes_detection()
        self._determine_model_configuration()

    @staticmethod
    def _determine_architecture_model_configuration(model_config_fpath):

        if not sly.fs.file_exists(model_config_fpath):
            raise RuntimeError('Unable to start training, model does not contain config.')

        with open(model_config_fpath) as fin:
            model_config = json.load(fin)

        # The old version of this code stored model_configuration inside the training config. For backwards
        # compatibility we accept both locations. If both are present, make sure they are consistent.
        # still accept that field in the input, but clear it before writing out the config in a new
        # format. Also, if model_configuration is present in the training config, make sure it is consistent with the
        # values coming from the model itself.
        model_configuration_model_root = model_config.get('model_configuration', None)
        model_configuration_model_subfield = model_config.get(SETTINGS, {}).get('model_configuration', None)
        if model_configuration_model_root is None and model_configuration_model_subfield is None:
            raise RuntimeError('Plugin misconfigured. model_configuration field is missing from internal config.json')
        elif (model_configuration_model_root is not None and
              model_configuration_model_subfield is not None and
              model_configuration_model_root != model_configuration_model_subfield):
            raise RuntimeError(
                'Plugin misconfigured. Inconsistent duplicate model_configuration field in internal config.json')
        else:
            return (model_configuration_model_root
                    if model_configuration_model_root is not None
                    else model_configuration_model_subfield)

    def _determine_model_configuration(self):
        self.model_configuration = ObjectDetectionTrainer._determine_architecture_model_configuration(
            sly.TaskPaths.MODEL_CONFIG_PATH)

        # Check for possible model_configuration field in old-style config. If exists, make sure it is consistent with
        # the actual model config and clear model_configuration from training config before writing new-style model
        # config.
        training_model_configuration = self.config.get('model_configuration', None)
        if training_model_configuration is not None:
            if training_model_configuration != self.model_configuration:
                error_msg = (
                        'Unable to start training. model_confguration in the training config is not consistent with ' +
                        'selected model architecture. Make sure you have selected the right model plugin and remove ' +
                        'model_confguration from the training config as it is not required anymore.')
                logger.critical(error_msg,
                                extra={'training_model_configuration': self.config['model_configuration'],
                                       'model_configuration': self.model_configuration})
                raise RuntimeError(error_msg)
            del self.config['model_configuration']

    def _determine_out_config(self):
        super()._determine_out_config()
        self.out_config['model_configuration'] = self.model_configuration

    def _construct_data_loaders(self):
        self.tf_data_dicts = {}
        self.iters_cnt = {}
        for the_name, the_tag in self.name_to_tag.items():
            samples_lst = self._deprecated_samples_by_tag[the_tag]
            supervisely_lib.nn.dataset.ensure_samples_nonempty(samples_lst, the_tag, self.project.meta)
            dataset_dict = {
                "samples": samples_lst,
                "classes_mapping": self.class_title_to_idx,
                "project_meta": self.project.meta,
                "sample_cnt": len(samples_lst)
            }
            self.tf_data_dicts[the_name] = dataset_dict
            num_gpu_devices = len(self.config['gpu_devices'])
            single_gpu_batch_size = self.config['batch_size'][the_name]
            effective_batch_size = single_gpu_batch_size * num_gpu_devices
            if len(samples_lst) < effective_batch_size:
                raise RuntimeError(f'Not enough items in the {the_name!r} fold (tagged {the_tag!r}). There are only '
                                   f'{len(samples_lst)} items, but the effective batch size is {effective_batch_size} '
                                   f'({num_gpu_devices} GPU devices X {single_gpu_batch_size} single GPU vatch size).')

            self.iters_cnt[the_name] = len(samples_lst) // effective_batch_size
            logger.info('Prepared dataset.', extra={
                'dataset_purpose': the_name, 'dataset_tag': the_tag, 'sample_cnt': len(samples_lst)
            })

    def _construct_and_fill_model(self):
        self._make_tf_train_config()

    def _construct_loss(self):
        pass

    def _make_tf_train_config(self):
        self.train_iters = self.tf_data_dicts['train']['sample_cnt'] // self.config['batch_size']['train']
        total_steps = self.config['epochs'] * self.train_iters
        src_size = self.config['input_size']
        input_size = (src_size['height'], src_size['width'])
        base_config_path, remake_config_fn = determine_tf_config(self.model_configuration)

        tf_config = load_sample_config(base_config_path)

        weights_dir = osp.join(sly.TaskPaths.MODEL_DIR, "model_weights")
        if (not sly.fs.dir_exists(weights_dir)) or sly.fs.dir_empty(weights_dir):
            checkpoint = None
            logger.info('Weights will not be inited.')
        else:
            checkpoint = osp.join(sly.TaskPaths.MODEL_DIR, 'model_weights', 'model.ckpt')
            logger.info('Weights will be loaded from previous train.')

        self.tf_config = remake_config_fn(tf_config,
                                          'SUPERVISELY_FORMAT',
                                          total_steps,
                                          max(self.class_title_to_idx.values()),
                                          input_size,
                                          self.config['batch_size']['train'],
                                          self.config['lr'],
                                          checkpoint)

        logger.info(self.tf_config)
        logger.info('Model config created.')

    def _dump_model_weights(self, out_dir):
        save_config(osp.join(out_dir, 'model.config'), self.tf_config)
        model_fpath = os.path.join(out_dir, 'model_weights', 'model.ckpt')
        self.saver.save(self.sess, model_fpath)

    def train(self):
        device_ids = sly.env.remap_gpu_devices(self.config['gpu_devices'])

        progress_dummy = sly.Progress('Building model:', 1)
        progress_dummy.iter_done_report()

        def dump_model(saver, sess, is_best, opt_data):
            self.saver = saver
            self.sess = sess
            self._save_model_snapshot(is_best, opt_data)

        train(self.tf_data_dicts,
              self.config['epochs'],
              self.config['val_every'],
              self.iters_cnt,
              self.config['validate_with_model_eval'],
              pipeline_config=self.tf_config,
              num_clones=len(device_ids),
              save_cback=dump_model,
              is_transfer_learning=(self.config['weights_init_type'] == 'transfer_learning'))


def main():
    cv2.setNumThreads(0)
    x = ObjectDetectionTrainer()  # load model & prepare all
    x.train()


if __name__ == '__main__':
    if os.getenv('DEBUG_LOG_TO_FILE', None):
        sly_logger.add_default_logging_into_file(logger, sly.TaskPaths.DEBUG_DIR)
    sly.main_wrapper('TF_OBJECT_DETECTION_TRAIN', main)
