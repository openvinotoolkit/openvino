Sample
===========

In this sample we will go through typical steps required to evaluate DL topologies. 

We will try to evaluate **SampLeNet** topology as an example

### 1. Extract dataset

In this sample we will use toy dataset which we refer to as *sample dataset*, which contains 10k images 
of 10 different classes (classification problem), which is actually CIFAR10 dataset converted to png.

```bash
tar xvf sample/sample_dataset.tar.gz -C sample
```

### 2. Evaluate sample topology

Typically you need to write configuration file, describing evaluation process of your topology. 
There is already config file for evaluating SampLeNet using OpenVINO framework, read it carefully.

```bash
accuracy_check -c sample/sample_config.yml -m data/test_models -s sample
```

Used options: `-c` path to evaluation config, `-m` directory where models are stored, `-s` directory where source data (datasets).

If everything worked correctly, you should be able to get `75.02%` accuracy. 

Now try edit config, to run SampLeNet on other plugin of Inference Engine, or go directly to your topology!
