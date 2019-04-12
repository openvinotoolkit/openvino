# OpenVINO™ Calibration Python* package
The Inference Engine `openvino.tools.calibration` Python\* package includes types to calibrate a given FP32 model so that you can run it in low-precision 8-bit integer mode while keeping the input data of this model in the original precision.  
The package has the following dependencies:
* `openvino.tools.accuracy_checker` package
* `openvino.tools.benchmark` package.  

Please, refer to https://docs.openvinotoolkit.org for details.

## Usage
You can use the `openvino.tools.calibration` package in a simple way:
```Python
import openvino.tools.calibration as calibration

with calibration.CommandLineProcessor.process() as config:
    network = calibration.Calibrator(config).run()
    if network:
        network.serialize(config.output_model)
```
### Explanation
1. Import openvino.tools.calibration types:
```Python
import openvino.tools.calibration as calibration
```

2. Read configuration and process the model:
```Python
config = calibration.CommandLineProcessor.process()
```

3. Serialize result model:
```Python
network.serialize(config.output_model)
```