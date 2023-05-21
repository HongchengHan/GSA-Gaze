# <center>GSA-Gaze: Generative Self-adversarial Learning for Domain Generalized Driver Gaze Estimation</center>

### <center>*Hongcheng Han* (hanhc@stu.xjtu.edu.cn)</center>

## Purpose of Generative Self-adversarial Learning
Purpose of generative self-adversarial learning. (a) Cross-domain general gaze feature. $S_{src}$, $S_{tar1}$, $S_{tar2}$ are the feature spaces of the source domain and two target domains, $G$ is the general gaze feature for all domains. (b) Relationship of the general gaze feature and the extracted feature when training on the source domain. $S$ is the whole feature space of the source domain, $E$ indicates the extracted feature, $G$ refers to the general gaze feature. (c) Principle of generative self-adversarial learning. To enhance the cross-domain generalization capability of the model, $E$ is expected to match $G$. The extracted feature $E$ is involved to two tasks, the gaze regression task encourages precise gaze estimation to guide the model to learn more gaze-relevant features, while the adversarial image reconstruction task promotes an imprecise reconstruction of the input image, prompting the model to eliminate domain-specific features, finally, $E$ is matched to $G$.
<center>
<img src='.\figures\purpose.png', width='300'>
</center>

## Framework of GSA-Gaze
Framework of generative self-adversarial learning for gaze estimation. (a) Feature encoder. (b) Gaze regression module. GP is global pooling layer. (c) Adversarial reconstruction module. GRL refers to gradient reversal layer. (d) Loss function.In the gaze regression task, gaze regression module performs cooperative optimization with the feature encoder, encouraging to extract more gaze-relevant features. In the image reconstruction task, the generative reconstruction module performs adversarial optimization with the feature extractor, encouraging to extract fewer features from the input image. As a result, the model is guided to learn only the general gaze features, the domain generalization capability is enhanced.
<center>
<img src='.\figures\framework.png', width='600'>
</center>

## Reconstructed images
Visualized reconstruction results. The top row shows the original images, the bottom row shows the reconstructed images. 
<center>
<img src='.\figures\reconstruction.png', width='400'>
</center>

## Subjective Evaluation of Driver Gaze Estimation
Subjective evaluation results on driver gaze estimation. (a) Results on XJTU-DA dataset. (b) Results on AUC dataset. The orange, blue and purple arrows respectively refer to the estimation of the drivers' gaze directions of GSA-Gaze, CA-Net, and Dilated-Net.
<center>
<img src='.\figures\drivers.png', width='300'>
</center>

