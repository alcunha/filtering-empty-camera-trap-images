# Filtering Empty Camera Trap Images in Embedded Systems

This is the origin TensorFlow implementation for: [Filtering Empty Camera Trap Images in Embedded Systems]()

### Requirements

Prepare an environment with python=3.8, tensorflow=2.3.1, and install the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md).

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

### Datasets

[Caltech](http://lila.science/datasets/caltech-camera-traps) and [Snapshot Serengeti](http://lila.science/datasets/snapshot-serengeti) Datasets can be downloaded from [Lila](http://lila.science/).

We used the recommended Lila train/val partitions for splits based on locations (site).The hyperparameters were tuned using a val_dev split from the train split of each data set.

For Caltech, we only used a [subset](https://drive.google.com/file/d/1aMcP5aDhBTBXrpkog8_TTKVxLSvW4DGt/view?usp=sharing) of images containing bounding boxes plus a similar amount of instances sampled from images labeled as empty.

For Snapshot Serengeti, we also split the dataset regarding time (seasons) and capture events (bursts). The list of images used for each partition can be found [here](https://drive.google.com/drive/folders/1yGNmigERn1N3pWQ45-jJLKE8aIkLtJaQ?usp=sharing).

We also provide scripts for resizing images and convert dataset to tfrecords format. See `dataset_tools` folder.

### Training

#### Classifiers

To train a classifier use the script `main.py`:
```bash
python main.py --training_files=PATH_TO_BE_CONFIGURED/caltech_train.record-?????-of-00068 \
    --num_training_instances=40606 \
    --validation_files=PATH_TO_BE_CONFIGURED/caltech_val_dev.record-?????-of-00012 \
    --num_validation_instances=6701 \
    --num_classes=2 \
    --model_name=mobilenetv2 \
    --input_size=224 \
    --input_scale_mode=tf_mode \
    --batch_size=128 \
    --lr=0.01 \
    --epochs=10 \
    --randaug_num_layers=2 \
    --randaug_magnitude=2 \
    --random_seed=42
```

For more parameter information, please refer to `main.py`. See `configs` folder for some training configs examples.

#### Detectors

To train a detector use the following script `object_detection/model_main_tf2.py` from TensorFlow Object Detection API:
```bash
# From the tensorflow/models/research/ directory
python object_detection/model_main_tf2.py \
    --pipeline_config_path=PATH_TO_BE_CONFIGURED/ssd_efficientdet_d0_512x512_serengeti_agnostic.config \
    --model_dir=PATH_TO_BE_CONFIGURED/efficientdet_d0_serengeti_site_agnostic_12jan \
    --alsologtostderr
```

See `configs` folder for detectors training config files.

### Evaluation

To evaluate a classifier use the script `eval_classifier_from_ckpt.py`:
```bash
python eval_classifier_from_ckpt.py --model_name=mobilenetv2 \
    --input_size=320 \
    --num_classes=2 \
    --input_scale_mode=tf_mode \
    --ckpt_dir=PATH_TO_BE_CONFIGURED/mobilenet_v2_320_caltech_agnostic_19nov \
    --validation_files=PATH_TO_BE_CONFIGURED/caltech_val_small.record-?????-of-00009
```

To evaluate a detector as a classifier use the script `eval_detector_from_saved_model.py`:
```bash
python eval_detector_from_saved_model.py \
    --exported_model_path=PATH_TO_BE_CONFIGURED/ssdlite_mobilenetv2_320_caltech_agnostic_19nov_exported \
    --model_name=ssdlite-mobilenetv2 \
    --num_classes=2 \
    --input_scale_mode=uint8 \
    --validation_files=PATH_TO_BE_CONFIGURED/caltech_val_small.record-?????-of-00009
```

For more evaluation options, refer to files starting with `eval_`.

To measure latency on Raspberry Pi, download the [TensorFlow Lite benchmark tool](https://www.tensorflow.org/lite/performance/measurement) for ARM, upload this tool and the TFLite model to Raspberry Pi, and run the command:
```bash
./linux_arm_benchmark_model --num_runs=50 --graph=model.tflite
```

### Results

All model checkpoints and TFLite binary files are available [here](https://drive.google.com/drive/folders/16vQSGJEmbHDbLerut7bRIIoGl8mL-pOZ?usp=sharing).

#### Precision-Recall Curves
![Precision-Recall Curves](data/pr_curves.png?raw=true)

#### Precision and  True Negative Rate (TNR) @ Recall = 96%

| Model name          | Input size  | Training set  | Precision | TNR    | Threshold |
|---------------------|-------------|---------------|-----------|--------|-----------|
| Efficientnet-B0     | 224x224     | Caltech       | 60.26%    | 25.50% | 0.355     |
| Efficientnet-B3     | 300x300     | Caltech       | 56.42%    | 12.75% | 0.305     |
| MobileNetV2-224     | 224x224     | Caltech       | 58.60%    | 20.18% | 0.228     |
| MobileNetV2-320     | 320x320     | Caltech       | 58.58%    | 20.13% | 0.239     |
| SSDLite+MobileNetV2 | 320x320     | Caltech       | 67.03%    | 44.42% | 0.166     |
| Efficientdet-D0     | 512x512     | Caltech       | 73.31%    | 58.86% | 0.148     |
| Efficientnet-B0     | 224x224     | SS-Site-Small | 50.32%    | 63.00% | 0.153     |
| Efficientnet-B3     | 300x300     | SS-Site-Small | 57.21%    | 71.97% | 0.155     |
| MobileNetV2-224     | 224x224     | SS-Site-Small | 57.72%    | 72.55% | 0.188     |
| MobileNetV2-320     | 320x320     | SS-Site-Small | 62.84%    | 77.84% | 0.191     |
| SSDLite+MobileNetV2 | 320x320     | SS-Site-Small | 75.32%    | 87.72% | 0.167     |
| Efficientdet-D0     | 512x512     | SS-Site-Small | 79.14%    | 90.12% | 0.150     |
| Efficientnet-B0     | 224x224     | SS-Site       | 73.92%    | 86.78% | 0.291     |
| Efficientnet-B3     | 300x300     | SS-Site       | 87.67%    | 94.73% | 0.438     |
| MobileNetV2-224     | 224x224     | SS-Site       | 75.18%    | 87.63% | 0.364     |
| MobileNetV2-320     | 320x320     | SS-Site       | 82.89%    | 92.26% | 0.420     |
| Efficientnet-B0     | 224x224     | SS-Time-Small | 33.08%    | 61.90% | 0.159     |
| Efficientnet-B3     | 300x300     | SS-Time-Small | 39.27%    | 70.87% | 0.170     |
| MobileNetV2-224     | 224x224     | SS-Time-Small | 30.51%    | 57.11% | 0.126     |
| MobileNetV2-320     | 320x320     | SS-Time-Small | 35.74%    | 66.13% | 0.147     |
| SSDLite+MobileNetV2 | 320x320     | SS-Time-Small | 47.14%    | 78.89% | 0.147     |
| Efficientdet-D0     | 512x512     | SS-Time-Small | 47.35%    | 79.06% | 0.143     |
| Efficientnet-B0     | 224x224     | SS-Time       | 48.81%    | 80.25% | 0.266     |
| Efficientnet-B3     | 300x300     | SS-Time       | 64.28%    | 89.54% | 0.403     |
| MobileNetV2-224     | 224x224     | SS-Time       | 49.12%    | 80.50% | 0.317     |
| MobileNetV2-320     | 320x320     | SS-Time       | 58.61%    | 86.70% | 0.372     |


#### Latency* on Raspberry Pi 3

| Model name           | Input size  | Latency<br>Float | Latency<br>Int8 |
|----------------------|-------------|---------|---------|
| Efficientnet-B0      | 224x224     | 800ms   | -       |
| Efficientnet-B3      | 300x300     | 3173ms  | -       |
| MobileNetV2-224      | 224x224     | 322ms   | 237ms   |
| MobileNetV2-0.50-320 | 320x320     | 259ms   | 237ms   |
| MobileNetV2-0.75-320 | 320x320     | 484ms   | 408ms   |
| MobileNetV2-320      | 320x320     | 635ms   | 485ms   |
| SSDLite+MobileNetV2  | 320x320     | 840ms   | 575ms   |
| Efficientdet-D0      | 512x512     | 4631ms  | -       |

*The entries show average latency over 50 runs. Results are from models trained on SS-Site, but the latency is similar for the other datasets.

### Contact

If you have any questions, feel free to contact Fagner Cunha (e-mail: fagner.cunha@icomp.ufam.edu.br) or Github issues. 

### License

[Apache License 2.0](LICENSE)