This repo contains code to evaluate explainability models in terms of their generated explanations. It implements a suite of multifaceted metrics to objectively compare explainers based on correctness, consistency, as well as the confidence of the generated explanations. These metrics are computationally inexpensive, do not require model-retraining and can be used across different data modalities. 

Currently, the explainers considered are Grad-CAM, SmoothGrad, LIME and Integrated Gradients, and the focus is on image data.

<img width="482" alt="explainers" src="https://user-images.githubusercontent.com/93252225/139057999-feb87082-05dd-419a-a552-648c8e47cabc.png">

Results of the metrics analytis can be found here: [explainable-metrics.pdf](https://github.com/amarogayo/xai-metrics/files/7425773/explainable-metrics.pdf)


