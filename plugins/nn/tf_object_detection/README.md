# WPILib Tensorflow Object Detection

The TensorFlow Object Detection API is an open source framework built on top of TensorFlow that makes it easy to construct, train and deploy object detection models.
### Description:
- **Paper**: [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/abs/1611.10012)
- **Framework**: [Tensorflow](https://www.tensorflow.org/)
- **Work modes**: train, inference, deploy

---

### Train configuration
_Default train configuration available in model presets._ 

Also you can read common training configurations [documentation](https://docs.supervise.ly/neural-networks/configs/inference_config/).

- `lr` - Learning rate.
- `epochs` - the count of training epochs.
- `val_every` - validation peroid by epoch (value `0.5` mean 2 validations per epoch).
- `batch_size` - batch sizes for training (`train`) and validation (`val`) stages.
- `gpu_devices` - list of selected GPU devices indexes.
- `data_workers` - how many subprocesses to use for data loading.
- `dataset_tags` - mapping for split data to train (`train`) and validation (`val`) parts by images tags. Images must be tagged by `train` or `val` tags.
- `special_classes` - objects with specified classes will be interpreted in a specific way. Default class name for `background` is `bg`, default class name for `neutral` is `neutral`. All pixels from `neutral` objects will be ignored in loss function. 
- `weights_init_type` - can be in one of 2 modes. In `transfer_learning` mode all possible weights will be transfered except last layer. In `continue_training` mode all weights will be transfered and validation for classes number and classes names order will be performed.

Full training configuration example:
```json
{
  "lr": 0.001,
  "epochs": 2,
  "val_every": 0.5,
  "batch_size": {
    "val": 6,
    "train": 12
  },
  "input_size": {
    "width": 256,
    "height": 256
  },
  "gpu_devices": [
    0
  ],
  "dataset_tags": {
    "val": "val",
    "train": "train"
  },
  "special_classes": {
    "neutral": "neutral",
    "background": "bg"
  },
  "weights_init_type": "continue_training"
}
```

### Inference configuration

For full explanation see [documentation](https://docs.supervise.ly/neural-networks/configs/inference_config).

**`model`** - group contains unique settings for each model:
 
  * `gpu_device` - device to use for inference. Right now we support only single GPU.
  
  * `confidence_tag_name` - name of confidence tag for predicted bound boxes.
 
 
**`mode`** - group contains all mode settings:

  *  `name` - mode name defines how to apply NN to image (e.g. `full_image` - apply NN to full image)
   
  *  `model_classes` - which classes will be used, e.g. NN produces 80 classes and you are going to use only few and ignore other. In that case you should set `save_classes` field with the list of interested class names. `add_suffix` string will be added to new class to prevent similar class names with exisiting classes in project. If you are going to use all model classes just set `"save_classes": "__all__"`.


Full image inference configuration example:

```json
{
  "model": {
    "gpu_device": 0,
    "confidence_tag_name": "confidence",
  },
  "mode": {
    "name": "full_image",
    "model_classes": {
      "save_classes": "__all__",
      "add_suffix": "_tf"
    }
  }
}
```