

# Private FGSM

## Introduction
This is the official repository of Private FGSM (P-FGSM), a work published as *Scene privacy protection* on Proc. of IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), Brighton, UK, May 12-17, 2019.


| ![Original Image](https://github.com/smartcameras/P-FGSM/blob/master/example/image.png) | ![Adversarial Image](https://github.com/smartcameras/P-FGSM/blob/master/example/image_adv.png) |
|--|--|
| <center>Original Image | <center>Adversarial image |
| <center>Church (confidence: 82.6%) | <center>Zen Garden (confidence: 99.2%) |

## Requirements
* Conda
* Python 2.7
* Numpy
* PyTorch
* Torchvision
* Opencv-python
* Tqdm

The code has been tested on Ubuntu 18.04 and MacOs 10.14.4.

## Setup

Install miniconda: https://docs.conda.io/en/latest/miniconda.html
Create conda environment for Python 2.7
```
conda create -n pfgsm python=2.7
```
Activate conda environment:
```
source activate pfgsm
```
Install requirements
```
pip install -r requirements.txt
```
**Only if using MacOs**:
```
export PATH="<pathToMiniconda>/bin:$PATH"
brew install wget
```

## Generate adversarial images

Create a folder with the images that you want to create their adversarials in ```<pathToImages>```
Generate adversarial images executing:
```
python p-fgsm.py --model=<model> --path=<pathToImages>
```
For example:
```
python p-fgsm.py --model=resnet50 --path=./images
```
In the publication [ResNet50](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar) is used. However, other models can be also be used: [ResNet18](http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar),  and [AlexNet](http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar). The pre-trained models will download automatically on the first execution of the code.

## Output and format

1. New adversarial folder inside the location of the input images with the generated adversarial images.
```
<pathToImages>/adversialimages/<imageName>.png
```
2. *log.txt* file in the following order of columns (format): 
 - image name
 - number of iterations to converge
 - original class
 - original class probability
 - final class
 - final class probability
 - target class
 - target class probability

## Authors
* [Chau Yi Li](mailto:chauyi.li@qmul.ac.uk), 
* [Ali Shahin Shamsabadi](mailto:a.shahinshamsabadi@qmul.ac.uk),
* [Ricardo Sanchez-Matilla](mailto:ricardo.sanchezmatilla@qmul.ac.uk),
* [Riccardo Mazzon](mailto:r.mazzon@qmul.ac.uk), and
* [Andrea Cavallaro](mailto:a.cavallaro@qmul.ac.uk).

## References
If you use our code, please cite the following publication:

    @InProceedings{Li2019,
      Title = {Scene Privacy Protection},
      Author = {C. Y. Li and A. S.  Shamsabadi and R. Sanchez-Matilla and R. Mazzon and A. Cavallaro},
      Booktitle = {Proc. IEEE Int. Conf. on Acoustics, Speech and Signal Processing},
      Year = {2019},
      Address  = {Brighton, UK},
      Month = May
    }

## License
The content of this project itself is licensed under the [Creative Commons Non-Commercial (CC BY-NC)](https://creativecommons.org/licenses/by-nc/2.0/uk/legalcode).
