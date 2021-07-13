# A real example to calibrate mobilenet model

Tested on OpenVINO 2021, Ubuntu 18.04

Please check the full video
XXXXXX

install
1. Run the command below:
```
export OV=/opt/intel/openvino_2021/
```

```
cd $OV/deployment_tools/model_optimizer/install_prerequisites
```

```
sudo ./install_prerequisites.sh
```

```
cd $OV/deployment_tools/open_model_zoo/tools/accuracy_checker
sudo python3 setup.py install

cd $OV/deployment_tools/tools/post_training_optimization_toolkit
sudo python3 setup.py install
