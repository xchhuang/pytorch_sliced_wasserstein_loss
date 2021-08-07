### A Sliced Wasserstein Loss for Neural Textyre Sythesis - PyTorch version

This is an unofficial, refactored PyTorch implementation of ["A Sliced Wasserstein Loss for Neural Texture Synthesis" paper](https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis) [CVPR 2021].

### Notice:
* The customized VGG-19 architecture might be different from the original Tensorflow implementation. Thus, some results might be inconsistent to the paper. Feel free to give advice.
* The spatial tag part is not included in this implementation.

### Prerequisites
* Python 3.7.10
* PyTorch 1.9.0

### Data
I have collected data in the `data` folder from the official repository and from ["Deep Correlations for Texture Synthesis
"](https://github.com/omrysendik/DCor) [Siggraph 2017].

### Run
First `cd pytorch` and then run some random examples:
```
python texturegen.py --data_folder=SlicedW --img_nam=input.jpg
python texturegen.py --data_folder=SlicedW --img_nam=2.png
python texturegen.py --data_folder=SlicedW --img_nam=berry.png
python texturegen.py --data_folder=SlicedW --img_nam=64.png
python texturegen.py --data_folder=DCor --img_nam=Texture13.png
python texturegen.py --data_folder=DCor --img_nam=Texture32.jpg
python texturegen.py --data_folder=DCor --img_nam=Texture19.png
```
After slgihtly more than 1 minute for each scene, you can find intermediate outputs in `outputs` folder, and final results in `results` folder.


### Results

Input                      |  Synthesized
:-------------------------:|:-------------------------:
![](results/input.jpg/resized-input.jpg)  |  ![](results/input.jpg/output.jpg)
![](results/berry.png/resized-input.jpg)  |  ![](results/berry.png/output.jpg)
![](results/Texture32.jpg/resized-input.jpg)  |  ![](results/Texture32.jpg/output.jpg)
![](results/64.png/resized-input.jpg)  |  ![](results//64.png/output.jpg)
![](results/Texture19.png/resized-input.jpg)  |  ![](results/Texture19.png/output.jpg)
![](results/2.png/resized-input.jpg)  |  ![](results/2.png/output.jpg)
![](results/Texture13.png/resized-input.jpg)  |  ![](results/Texture13.png/output.jpg)

### References
* [Official Tensorflow version](https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis)