You can convert [SSD and YOLO-v3 models from GluonCV](https://gluon-cv.mxnet.io/model_zoo/detection.html) to IR using the following instructions:

1 Choose topology and export it to MXNet format:
```python
from gluoncv import model_zoo, data, utils
from gluoncv.utils import export_block
net = model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
export_block('ssd_512_mobilenet1.0_voc', net, preprocess=True, layout='HWC')
```

2 Convert model using Model Optimizer tool:
* For GluonCV SSD topologies:
```sh
python3 mo_mxnet.py --input_model ~/models/ssd_300_t_mobile-0000.params --enable_ssd_gluoncv --input_shape [1,480,640,3] --input data
```

* For YOLO-v3 topology you have two option: convert as this and replace subgraph with RegionYolo layers
For convert as this:
```sh
python3 mo_mxnet.py --input_model ~/models/yolo3_mobilenet1.0_voc-0000.params  --input_shape [1,255,255,3]
```
For replase subgraph with RegionYolo layers:
```sh
python3 mo_mxnet.py --input_model ~/models/yolo3_mobilenet1.0_voc-0000.params  --input_shape [1,255,255,3] --transformations_config "mo/extensions/front/mxnet/yolo_v3_mobilenet1_voc.json"
```