# Samel

## Confidence Assessments
These are the methods that Samel might be compared to

- Softmax Threshold (singular network)

- MC Dropout ([Droupout as a Bayesian Approximation](https://arxiv.org/pdf/1506.02142.pdf)) (maybe singular network)

- Bayesian NNs

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

- [Dropout](https://arxiv.org/pdf/1506.02142.pdf) Consensus (Majority Voting) Ensembles

- [Bagging](https://link.springer.com/content/pdf/10.1007%2FBF00058655.pdf)

- [Boosting](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.51.6252&rep=rep1&type=pdf)

- [Snapshot Ensembles](https://openreview.net/pdf?id=BJYwwY9ll)

- [AdaNet](https://arxiv.org/pdf/1607.01097.pdf)

### Fusion types
Ways to make a prediction using the predictions of the individual classifiers

- Majority Voting

- Unweighted Average

- [Super Learner](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.211.6393&rep=rep1&type=pdf)

- [Stacked Generalisationa](https://www.sciencedirect.com/science/article/pii/S0893608005800231)

### Architectures
Neural Network architectures that could be used

- [NIN](https://arxiv.org/pdf/1312.4400.pdf)

- [GoogLeNet](https://arxiv.org/pdf/1409.4842.pdf)

- [VGG net](https://arxiv.org/pdf/1409.4842.pdf)

- [Res Net](https://arxiv.org/pdf/1512.03385.pdf)

### Saliency Methods
Saliency map generation methods that could be used

- [Gradient × Input](https://arxiv.org/pdf/1312.6034.pdf)

- [Competitive Gradient Input](https://arxiv.org/pdf/1905.12152)

- [Integrated Gradients](https://arxiv.org/pdf/1703.01365.pdf)

- [Layerwise Relevance Propagation](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable)

- [Taylor Decomposition](https://arxiv.org/pdf/1706.07979.pdf)

- [DeepLIFT](https://arxiv.org/pdf/1704.02685.pdf)

> Kindermans et al. [6] and Shrikumar et al. [3] showed that if modifications for numerical stability are not taken into account, the LRP rules are equivalent within a scaling factor to Gradient ⊙ Input. Ancona et al. [7] showed that for ReLU networks (with zero baseline and no biases) the ε-LRP and DeepLIFT (Rescale) explanation methods are equivalent to the Gradient ⊙ Input.

[Gupta, Arushi, and Sanjeev Arora. "A Simple Saliency Method That Passes the Sanity Checks." arXiv preprint arXiv:1905.12152 (2019).](https://arxiv.org/abs/1905.12152)
