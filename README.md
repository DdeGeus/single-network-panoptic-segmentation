# Single Network Panoptic Segmentation for Street Scene Understanding

Code for reproducing results presented in **Daan de Geus, Panagiotis Meletis, Gijs Dubbelman**, _Single Network Panoptic Segmentation for Street Scene Understanding_, IEEE Intelligent Vehicles Symposium 2019.

Link to paper on arXiv: [https://arxiv.org/abs/1902.02678](https://arxiv.org/abs/1902.02678)
 
If you find our work useful for your research, please cite the following paper:
```
@inproceedings{panoptic2019degeus,
  title={Single Network Panoptic Segmentation for Street Scene Understanding},
  author={Daan {de Geus} and Panagiotis Meletis and Gijs Dubbelman},
  booktitle={2019 IEEE Intelligent Vehicles Symposium (IV)},
  year={2019}
}
```

## Code usage
**THIS CODE COMES WITHOUT WARRANTY, USE THIS CODE AT YOUR OWN RISK.**

Before using the code, download the checkpoint files [here](https://www.dropbox.com/sh/iapcebdwiox40wk/AABAfTZu9ICPCbNHqLhUhEK2a?dl=0) and place them in the ```examples/checkpoint/``` directory. 

These checkpoints contain weights for a model trained on the Cityscapes dataset, and will therefore make predicitons for the _stuff_ and _things_ classes as defined for Cityscapes.

The Cityscapes images and annotations can be downloaded [here](https://www.cityscapes-dataset.com/). Conversion scripts, to convert Cityscapes annotations into panoptic segmentation annotations, can be found [here](https://github.com/cocodataset/panopticapi).
### Dependencies

Tested on Python 3.6.4, with:

```
- scipy==1.0.1
- matplotlib==2.2.2
- scikit_image==0.13.1
- numpy==1.14.2
- tensorflow==1.6.0
- opencv_python==3.4.0.12
- Pillow==5.1.0
```

It is likely that the code works on higher versions of the aforementioned packages, but this has not been tested.

For GPU inference, use ```tensorflow-gpu```.


### Inference

To run inference on demo images in the ```examples/demoimages/``` directory, run 

```shell
python predict_demo.py
```

To run inference on images from a different directory, e.g. ```~/datasets/Cityscapes/```, run

```shell
python predict_demo.py --image_dir=~/datasets/Cityscapes/
```


### Save predictions 
To save the panoptic segmentation predictions in the ```output/save_dir/``` directory, run

```shell
python predict_demo.py --save_predictions
```

Panoptic segmentation predictions will be saved as 2-channel images, where 
- Channel 1: **Class id**
- Channel 2: **Instance id**

To save the predictions, e.g. ```~/panoptic_outputs/```, run

```shell
python predict_demo.py --save_predictions --save_dir=~/panoptic_outputs/
```

### Performance
With our new, clean, implementation of the method described in the paper, we now achieve slightly better results with slightly lower prediction times:

Performance on Cityscapes:
```
          |    PQ     SQ     RQ     N
--------------------------------------
All       |  48.1   76.8   60.2    19
Things    |  40.6   75.1   53.6     8
Stuff     |  53.5   78.1   65.0    11
```

Average inference time on Cityscapes (using Nvidia Titan Xp):
```531 ms```

Evaluation scripts can be found [here](https://github.com/cocodataset/panopticapi).

## References
**Implementation references:**
- Faster-RCNN implementation by DetectionTeamUCAS: [https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow](https://github.com/DetectionTeamUCAS/Faster-RCNN_Tensorflow)
- TensorFlow Object Detection API:[https://github.com/tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

**Cityscapes dataset:**

M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016


**More information about Panoptic Segmentation:**
- Panoptic Segmentation paper (Kirillov et al.) on arXiv: [https://arxiv.org/abs/1801.00868](https://arxiv.org/abs/1801.00868)
- Panoptic API for COCO dataset: [https://github.com/cocodataset/panopticapi](https://github.com/cocodataset/panopticapi)