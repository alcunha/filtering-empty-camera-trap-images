# microdetector

### Datasets

[Caltech](http://lila.science/datasets/caltech-camera-traps) and [Snapshot Serengeti](http://lila.science/datasets/snapshot-serengeti) Datasets can be downloaded from [Lila](http://lila.science/).

We used the recommended Lila train/val partitions for splits based on locations (site).The hyperparameters were tuned using a val_dev split from the train split of each data set.

For caltech, we only used a [subset](https://drive.google.com/file/d/1aMcP5aDhBTBXrpkog8_TTKVxLSvW4DGt/view?usp=sharing) of images containing bounding boxes plus a similar amount of instances sampled from images labeled as empty.

For Snapshot Serengeti, we also split the dataset regarding time (seasons) and capture events (bursts). The list of images used for each partition can be found [here](https://drive.google.com/drive/folders/1yGNmigERn1N3pWQ45-jJLKE8aIkLtJaQ?usp=sharing).

We also provide scripts for resizing images and convert dataset to tfrecords format. See `dataset_tools` folder.

### Classifiers
Model name | Input size | Dataset | Accuracy | Precision (nonempty) | Recall (nonempty)
-----------|------------|---------|----------|----------------------|------------------
[Efficientnet-B0](https://drive.google.com/file/d/1HRfmJyC_1QkYdrRHJdrLhzAQ16NmbVlv/view?usp=sharing) | 224x224 | Caltech | 70.50% | 66.74% | 90.58%
[Efficientnet-B3](https://drive.google.com/file/d/1-30yk2IWMQqMIPbQVQPq01icUn8BmFTO/view?usp=sharing) | 300x300 | Caltech | 65.25% | 62.34% | 90.20%
[MobileNetV2](https://drive.google.com/file/d/1eyqC4kgYoXdvCGeEI4cCOTI5U7FIei5R/view?usp=sharing) | 224x224 | Caltech | 70.15% | 66.46% | 90.40%
[MobileNetV2](https://drive.google.com/file/d/16w5kz3cWhfyIooP3axfXfVXZlvyTyuFL/view?usp=sharing) | 320x320 | Caltech | 69.55% | 65.88% | 90.59%
[Efficientnet-B0](https://drive.google.com/file/d/1xbXNvgvRoSYPgv7ZC7RPmDWuz2gHzhYy/view?usp=sharing) | 224x224 | Snapshot Serengeti (Site) | 94.13% | 86.56% | 93.66%
[Efficientnet-B3](https://drive.google.com/file/d/1B44WgMgSx2dMr2qfN7Lq7vr_ll0oNMIQ/view?usp=sharing) | 300x300 | Snapshot Serengeti (Site) | 95.76% | 90.00% | 95.54%
[MobileNetV2](https://drive.google.com/file/d/1E4F6PZcuRFJ5HiQf7GKJ9TRDdpsdLyf5/view?usp=sharing) | 224x224 | Snapshot Serengeti (Site) | 93.13% | 83.20% | 94.64%
[MobileNetV2](https://drive.google.com/file/d/1mMvp_gsUd_wucg8LlzkgAYJ_Q6BV-VkV/view?usp=sharing) | 320x320 | Snapshot Serengeti (Site) | 94.59% | 86.70% | 95.36%
[Efficientnet-B0](https://drive.google.com/file/d/1T6TYGkcpKmjnG6LJtS8OCcabtDCle1Yw/view?usp=sharing) | 224x224 | Snapshot Serengeti (Time) | 90.91% | 65.78% | 92.89%
[Efficientnet-B3](https://drive.google.com/file/d/1ZU9nb_1G-gEPJwcjnm1sUa_Ik3LxD6wb/view?usp=sharing) | 300x300 | Snapshot Serengeti (Time) | 92.61% | 70.30% | 95.14%
[MobileNetV2](https://drive.google.com/file/d/1VsFMxDrhvZqBCxrdd4WMjMsiA2i20Tv2/view?usp=sharing) | 224x224 | Snapshot Serengeti (Time) | 89.75% | 62.46% | 93.99%
[MobileNetV2](https://drive.google.com/file/d/1dyOU0GnQphSq-S7_B1_d3qWwRnjOCQnF/view?usp=sharing) | 320x320 | Snapshot Serengeti (Time) | 91.70% | 67.64% | 94.79%
[Efficientnet-B0](https://drive.google.com/file/d/1zkDN1g8LeBdgqFoGEBqgbcGdoKjpLn-3/view?usp=sharing) | 224x224 | Snapshot Serengeti (Event) | 95.84% | 96.49% | 95.04%


### Detectors
Model name | Input size | Dataset | mAP | AR@1 | Accuracy* | Precision (nonempty)* | Recall (nonempty)*
-----------|------------|---------|-----|------|----------|----------------------|------------------
[Efficientdet-D0](https://drive.google.com/file/d/1PV9r3V7c1zMaiYDAjXKgqf82e8wWAgFU/view?usp=sharing) | 512x512 | Caltech | 55.4 | 59.6 | 88.39% | 97.14% | 80.90%
[SSD MobileNetV2 FPNLite](https://drive.google.com/file/d/1xpCbsFkjpDSLzcCg2vKGcHmSQsPnO-lO/view?usp=sharing) | 320x320 | Caltech | 49.2 | 54.2 | 84.70% | 93.13% | 77.41%
[Efficientdet-D0](https://drive.google.com/file/d/15W3LsJN6w9quK8Url7YJgqGcSUWT6WDv/view?usp=sharing) | 512x512 | Snapshot Serengeti (Site) | 51.60** | 41.64** | 95.43% | 96.12% | 87.27%
[SSD MobileNetV2 FPNLite](https://drive.google.com/file/d/1AlpXHB_5uvi7of55ThyhwxFze_gtmvEI/view?usp=sharing) | 320x320 | Snapshot Serengeti (Site) | 46.43** | 39.80** | 94.06% | 95.34% | 82.89%
[Efficientdet-D0](https://drive.google.com/file/d/1DSHL_o64e-bXPgFpJFq4yAQbcf69Pg-t/view?usp=sharing) | 512x512 | Snapshot Serengeti (Time) | 53.45** | 42.90** | 96.65% | 94.03% | 85.01%
[SSD MobileNetV2 FPNLite](https://drive.google.com/file/d/1pmcZIQ92dFyrSxRccynuvS-uksV5h5xV/view?usp=sharing) | 320x320 | Snapshot Serengeti (Time) | 48.07** | 40.82** | 95.70% | 92.08% | 80.77%

*Accuracy, precision and recall on detectors are calculated considering only the bounding box with highest confidence. We used the 0.5 as thrshold to consider as a nonempty class, the same used for the classifiers.

**mAP and AR@1 on val_dev split because validation split of serengeti doesn't contain bbox for all images containing animals.

### Latency* on Raspberry Pi 3

Model name | Input size | Latency (ms)** | Peak memory (MB)
-----------|------------|----------------|-----------------
Efficientnet-B0 | 224x224 | 801.205 ± 4.459 | 34.3047
Efficientnet-B0 (Quantized) | 224x224 | 558.842 ± 4.822 | 14.1758
Efficientnet-B3 | 300x300 | 3205.12 ± 27.869 | 89.3789
Efficientnet-B3 (Quantized) | 300x300 | DNR | DNR
MobileNetV2 | 224x224 | 324.017 ± 2.941 | 22.8711
MobileNetV2 (Quantized) | 224x224 | 240.637 ± 3.3 | 9.30469
MobileNetV2 | 320x320 | 638.368 ± 5.277 | 33.5195
MobileNetV2 (Quantized) | 320x320 | 488.569 ± 4.835 | 13.8516
Efficientdet-D0 | 512x512 | 4686.15 ± 28.41 | 144.816
Efficientdet-D0 (Quantized) | 512x512 | 3457.48 ± 21.352 | 92.5391
SSD MobileNetV2 FPNLite | 320x320 | 838.604 ± 10.335 | 31.6719
SSD MobileNetV2 FPNLite (Quantized) | 320x320 | 577.993 ± 4.393 | 13.6055

*Results are from models trained on Caltech, but the latency is similar for the [other datasets](https://drive.google.com/file/d/1i4ryy9ubVW6j0SxD8KF8s6LrYhVwfA0X/view?usp=sharing).

**The entries show average latency ± standard deviation over 50 runs

DNR = Did not run

## License

[Apache License 2.0](LICENSE)