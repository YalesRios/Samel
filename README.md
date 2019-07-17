# Samel

## Confidence Assessments
These are the methods that Samel might be compared to

- Softmax Threshold (singular network)

- MC Dropout (Droupout as a Bayesian Approximation) (maybe singular network)

### Saliency Variance Calculation Methods

- Standard Deviation

- Interquartile range

### How will the saliency play a part in influencing the prediction?

- Threshold

- Learned Threshold

### Datasets/Problems

- Endangered Animals

### Ensemble types
Ways to organise/generate classifiers that are different from each other

- Dropout Ensembles

- Bagging

- Boosting

- Snapshot Ensembles

- AdaNet

### Fusion types
Ways to make a prediction using the predictions of the individual classifiers

- Majority Voting

- Unweighted Average

- Super Learner

- Stacked Generalisation (should be more specific)

- Bayes Optimal Classifier

### Architectures
Neural Network architectures that could be used

- NIN

- GoogLeNet

- VGG net

- Res Net

### Saliency Methods
Saliency map generation methods that could be used

- Gradient Input

- Competitive Gradient Input

- Integrated Gradients

- Layerwise Relevance Propagation

- Taylor Decomposition

- DeepLIFT

> Kindermans et al. [6] and Shrikumar et al. [3] showed that if modifications for numerical stability are not taken into account, the LRP rules are equivalent within a scaling factor to Gradient ⊙ Input. Ancona et al. [7] showed that for ReLU networks (with zero baseline and no biases) the ε-LRP and DeepLIFT (Rescale) explanation methods are equivalent to the Gradient ⊙ Input.

[Gupta, Arushi, and Sanjeev Arora. "A Simple Saliency Method That Passes the Sanity Checks." arXiv preprint arXiv:1905.12152 (2019).](https://arxiv.org/abs/1905.12152)
