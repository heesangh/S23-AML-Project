# Change Detection in satellite images with Early and Late fusion 

## Term project for Virginia Tech CS5824/ECE5424 Advanced Machine Learning Spring 2023
### Heesang Han (VT ECE), Chhayly Sreng (VT ECE), Spencer Paragas (VT CS), Ranhee Yoon (VT ME)

### To execute the code, run the jupyter notebook file ```change_detection.ipynb```
If not running in Google Colab, ignore and remove the first cell. If running in Google Colab, zip the whole directory except change_detection.ipynb and name it addon.zip. Then upload to the drive and mount drive to Google Colab. This code is configured to run in Google Colab. 

### Base code, including jupyter notebook:
Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.
(We modified the code)

### Model
FC-EF, FC-Siam-Conc, FC-Siam-Diff: \\
Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch. (2018, October). Fully convolutional siamese networks for change detection. In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE. 

Fres-UNet: (Not implemented yet) \\
Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2019. Multitask learning for large-scale semantic change detection. Computer Vision and Image Understanding, 187, p.102783.


### Dataset
CDD Dataset: \\
M. A. Lebedev, Y. V. Vizilter, O. V. Vygolov, V. A. Knyaz, and A. Y. Rubis, “Change detection in remote sensing images using conditional adversarial networks,” The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, vol. XLII-2, pp. 565–571, May 2018. 

### Notes

The `requirements.txt` file should list all Python libraries that your notebooks
depend on, and they will be installed using:

```
pip install -r requirements.txt
```
The `requirements-gpu.txt` file should list all Python libraries that your notebooks
depend on for training models with cuda GPU-accelerations, and they will be installed using:

```
pip install -r requirements-gpu.txt
```


