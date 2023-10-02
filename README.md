## BAAF

[BAAF: A Benchmark Attention Adaptive Framework for Medical Ultrasound Image Segmentation Tasks](https://doi.org/10.1016/j.eswa.2023.119939) (Expert Systems with Applications)


## Network
![BAAF](https://github.com/CGPxy/BAAF/assets/52651150/ca0eeba4-8960-441d-9f27-8b51e3ac0a36)
The proposed benchmark attention adaptive framework.

![U-net with BAAF](https://github.com/CGPxy/BAAF/assets/52651150/64b318d0-7708-473a-b4ba-1cb9d16eff96)
The U-shaped network we constructed using the BAAF block.


## 3. Datasets：
### Breast ultrasound dataset:
(1)[BUSI:](https://doi.org/10.1016/j.dib.2019.104863) W. Al-Dhabyani., Dataset of breast ultrasound images, Data Br. 28 (2020) 104863.  
(2)[Dataset B:](https://doi.org/10.1016/j.artmed.2020.101880) M. H. Yap et al., Breast ultrasound region of interest detection and lesion localisation, Artif. Intell. Med., vol. 107, no. August 2019, p. 101880, 2020.  
(3)[STU:](https://doi.org/10.1371/journal.pone.0221535) Z. Zhuang, N. Li, A. N. Joseph Raj, V. G. V Mahesh, and S. Qiu, “An RDAU-NET model for lesion segmentation in breast ultrasound images,” PLoS One, vol. 14, no. 8, p. e0221535, 2019.  
### Kidney ultrasound dataset:
   we collected contains 300 clinical kidney ultrasound images from 300 patients in the Fourth Medical Center of the PLA General Hospital and the Civil Aviation General Hospital.

## Experimental Setting
The development environment of our network is Ubuntu 20.04, python 3.6 and TensorFlow 2.6.0. We train our network with two NVIDIA RTX 3090 GPUs.


## Experiment Results

![2](https://user-images.githubusercontent.com/52651150/227098643-07f60237-2185-4106-a8ae-cbe8a7d909c6.png)
