# Learning-latent-representations-for-nitrogen-response-rate-prediction

Appeared at [AI for Earth and Space Science workshop, ICLR 2022](https://ai4earthscience.github.io/iclr-2022-workshop/). In this paper we examined if our [previously proposed method](https://www.sciencedirect.com/science/article/pii/S1364815221003169) to overcome data quantity and resolution issues is able to work independently of the machine learning algorithm used to make predictions. We compared three neural network architectures with a reference Random Forest model in a case study of nitrogen response rate prediction. The architectures were:

- a Multilayer Perceptron
<p align = "center">
<img src="/imgs/mlp_architecture.png" width="320" height="200" />
</p>

- an Autoencoder where we replace the decoder with a regression head after training
<p align = "center">
<img src="/imgs/autoencoder_architecture.png" width="450" height="250" />
</p>

- a dual-head Autoencoder which optimizes the reconstruction and prediction losses simultaneously
<p align = "center">
<img src="/imgs/dual_head_autoencoder_architecture.png" width="450" height="250"/>
</p>

On the repository there is also the LSTM version of the dual-head Autoencoder which was not included in the paper:

<p align = "center">
<img src="/imgs/dual_head_lstm_autoencoder_architecture.png" width="500" height="250"/>
</p>
