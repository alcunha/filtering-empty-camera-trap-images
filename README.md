# microdetector

### Classifiers
Model name | Input size | Dataset | Accuracy | Precision (nonempty) | Recall (nonempty)
-----------|------------|---------|----------|----------------------|------------------
[Efficientnet-B0](https://drive.google.com/file/d/1HRfmJyC_1QkYdrRHJdrLhzAQ16NmbVlv/view?usp=sharing) | 224x224 | Caltech | 70.5% | 66.7% | 90.6%
[Efficientnet-B3](https://drive.google.com/file/d/1-30yk2IWMQqMIPbQVQPq01icUn8BmFTO/view?usp=sharing) | 300x300 | Caltech | 65.3% | 62.3% | 90.2%
[MobileNetV2](https://drive.google.com/file/d/1eyqC4kgYoXdvCGeEI4cCOTI5U7FIei5R/view?usp=sharing) | 224x224 | Caltech | 70.2% | 66.5% | 90.4%
[MobileNetV2](https://drive.google.com/file/d/16w5kz3cWhfyIooP3axfXfVXZlvyTyuFL/view?usp=sharing) | 320x320 | Caltech | 69.6% | 65.9% | 90.6%
[Efficientnet-B0](https://drive.google.com/file/d/1xbXNvgvRoSYPgv7ZC7RPmDWuz2gHzhYy/view?usp=sharing) | 224x224 | Snapshot Serengeti (Site) | 94.1% | 86.6% | 93.6%
[Efficientnet-B0](https://drive.google.com/file/d/1T6TYGkcpKmjnG6LJtS8OCcabtDCle1Yw/view?usp=sharing) | 224x224 | Snapshot Serengeti (Time) | 90.9% | 65.8% | 92.9%
[Efficientnet-B0](https://drive.google.com/file/d/1zkDN1g8LeBdgqFoGEBqgbcGdoKjpLn-3/view?usp=sharing) | 224x224 | Snapshot Serengeti (Event) | 95.8% | 96.5% | 95.0%


### Detectors
Model name | Input size | Dataset | mAP | AR@1 | Accuracy* | Precision (nonempty)* | Recall (nonempty)*
-----------|------------|---------|-----|------|----------|----------------------|------------------
[Efficientdet-D0](https://drive.google.com/file/d/1PV9r3V7c1zMaiYDAjXKgqf82e8wWAgFU/view?usp=sharing) | 512x512 | Caltech | 55.4 | 59.6 | 88.4% | 97.1% | 80.9%
[SSD MobileNetV2 FPNLite](https://drive.google.com/file/d/1xpCbsFkjpDSLzcCg2vKGcHmSQsPnO-lO/view?usp=sharing) | 320x320 | Caltech | 49.2 | 54.2 | 84.7% | 93.1% | 77.4%

*Accuracy, precision and recall on detectors are calculated considering only the bounding box with highest confidence. We used the 0.5 as thrshold to consider as a nonempty class, the same used for the classifiers.