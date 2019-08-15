# Significance-Offset Convolutional Neural Network (SOCNN)
PyTorch implementation of SOCNN preseneted in 2018 ICML paper, [Autoregressive Convolutional Neural Networks for Asynchronous Time Series](http://proceedings.mlr.press/v80/binkowski18a/binkowski18a.pdf), Mikołaj Bińkowski, Gautier Marti, Philippe Donnat. This model proposes a autoregressive model for asynchronous time series data with an ability to adjust its regression weights based on input time-series. 

**Model architecture**
![Figure](https://www.researchgate.net/profile/Gautier_Marti/publication/314943008/figure/fig3/AS:669710655959046@1536682871421/A-scheme-of-proposed-SOCNN-architecture-The-network-preserves-the-time-dimension-up-to.ppm)

### Some minor notices
- [UCI dataset] In the paper, author generated asynchronously sampled the observations in UCI dataset, however to fully consider the asynchronous dynamics of the data, I randomly sampled the timestep.
- [UCI dataset] Unlike the paper, which suggests the probabilities of observing the attributes at each timestep, I just sampled the attributes from uniform distribution.
- [UCI dataset] Attributes in UCI household dataset have different scales so it seems that rescaling it during data preprocessing is mandatory. Also, results given in the paper cannot be achieved without rescaling or normalizing process. However, as such processing procedure is not explained in the paper, I did not implement it in my codes. So, if you are considering to further apply the model, you should implement **normalizing or rescaling** process.

## Prepare dataset
In the paper, experiments are conducted on several time series data, which includes artifical dataset, UCI household electricity dataset and Hedge fund credit quotes data. Following codes will download and preprocess the required dataset.(So far, preprocessing stage has been implemented for UCI household electricity dataset.)

### UCI household electricity dataset
```
# download dataset
curl -L -O -J https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip \
    -o data/household_power_consumption.zip
cd data
unzip household_power_consumption.zip
rm household_power_consumption.zip

# generate train/test dataset
python generate_data.py --fpath data/household_power_consumption.txt
```

## Run codes
First, Configure settings for the training with yaml files. Template for yaml file is given in [template config](configs/default.yaml). Example template for UCI household dataset is presented in [uci config](configs/elec_config_template.yaml). 

Run the code with following codes
```
python run.py --config_file configs/default.yaml 
```
