This document provides the instructions and examples on how to use Model Optimizer to convert [GluonCV SSD and YOLO-v3 models](https://gluon-cv.mxnet.io/model_zoo/detection.html) to IR.

1. Choose the topology available from the [GluonCV Moodel Zoo](https://gluon-cv.mxnet.io/model_zoo/detection.html) and export to the MXNet format using the GlounCV API. For example, for the `ssd_512_mobilenet1.0` topology: 
```python
from gluoncv import model_zoo, data, utils
from gluoncv.utils import export_block
net = model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
export_block('ssd_512_mobilenet1.0_voc', net, preprocess=True, layout='HWC')
```
As a result, you will get an MXNet model representation in `ssd_512_mobilenet1.0.param` and `ssd_512_mobilenet1.0.json` files. 
2. Convert the MXNet model using the Model Optimizer tool:
* **For GluonCV SSD topologies:**
```sh
python3 mo_mxnet.py --input_model ~/models/ssd_300_t_mobile-0000.params --enable_ssd_gluoncv --input_shape [1,480,640,3] --input data
```
* **For YOLO-v3 topology** you can use the following convertion options:
   * To convert the model as is:
   ```sh
   python3 mo_mxnet.py --input_model ~/models/yolo3_mobilenet1.0_voc-0000.params  --input_shape [1,255,255,3]
   ```
   * To convert the model with replacing the subgraph with RegionYolo layers:
   ```sh
   python3 mo_mxnet.py --input_model ~/models/yolo3_mobilenet1.0_voc-0000.params  --input_shape [1,255,255,3] --transformations_config "mo/extensions/front/mxnet/yolo_v3_mobilenet1_voc.json"
   ```
