# Semantic-segmentation-of-tumor-using-brain-MRI-scans
The aim of this project was to draw a boundary around the tumor in Brain MRI scans. Semantic segmentation was achieved using multiple neural networks(Unet &amp; Resunet) and loss functions (Dice Loss &amp; Focal Tversky Loss) trained from scratch using Keras. ResUnet trained using Focal Tversky loss performed the best giving a dice_coeff of 0.901 and IoU of 0.82.

All the metrics are on validation data which was 20% of whole dataset

![Metrics](https://github.com/yashkhasgiwala/Semantic-segmentation-of-tumor-using-brain-MRI-scans/blob/main/Images/metrics.JPG?raw=true)

Some examples of prediction of Resunet trained using Focal Tversky Loss

![Predictions](https://github.com/yashkhasgiwala/Semantic-segmentation-of-tumor-using-brain-MRI-scans/blob/main/Images/result.JPG?raw=true)
