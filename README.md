# Change Detection in satellite images with Early and Late fusion 

## Term project for Virginia Tech CS5824/ECE5424 Advanced Machine Learning Spring 2023
### Heesang Han (VT ECE), Chhayly Sreng (VT ECE), Spencer Paragas (VT CS), Ranhee Yoon (VT ME)

### How to run
1. Download or clone this repository. 
2. Install libraries and dependencies using 'requirements.txt' or 'requirements-gpu.txt' (You can skip this step if using Google Colab)

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
3. Download addon.zip from this link: https://drive.google.com/file/d/1bVkP3PNMRQeadZYJisbCMtG-s_6bHKOk/view?usp=sharing
(addon.zip contains dataset and python code for datasets and torch model)

4. Extract addon.zip right in the local repository. If you are using Google Colab, upload this zip file to your drive root directory and mount Google Drive to Google Colab.

5. Run the Jupyter notebook - change_detections.ipynb. If you are not using Google Colab, skip the first cell. 

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


