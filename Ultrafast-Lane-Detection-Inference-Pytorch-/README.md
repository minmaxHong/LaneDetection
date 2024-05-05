# Requirements

 * **OpenCV**, **Scikit-learn** and **pytorch**. **pafy** and **youtube-dl** are required for youtube video inference. 
 
# Installation
```
pip install -r requirements
```
**Pytorch:** Check the [Pytorch website](https://pytorch.org/) to find the best method to install Pytorch in your computer.

# Pretrained model
Download the pretrained model from the [original repository](https://github.com/cfzd/Ultra-Fast-Lane-Detection) and save it into the **[models](https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-/tree/main/models)** folder. 

# Ultra fast lane detection - TuSimple([link](https://github.com/cfzd/Ultra-Fast-Lane-Detection))

 * **Input**: RGB image of size 1280 x 720 pixels.
 * **Output**: Keypoints for a maximum of 4 lanes (left-most lane, left lane, right lane, and right-most lane).
 
# Examples

 * **Image inference**:
 
 ```
 python imageLaneDetection.py 
 ```
 
  * **Webcam inference**:
 
 ```
 python webcamLaneDetection.py
 ```
 
  * **Video inference**:
 
 ```
 python videoLaneDetection.py
 ```
 
