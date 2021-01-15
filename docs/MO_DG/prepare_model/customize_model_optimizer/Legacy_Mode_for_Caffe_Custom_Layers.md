# Legacy Mode for Caffe* Custom Layers  {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Legacy_Mode_for_Caffe_Custom_Layers}

> **NOTE**: This functionality is deprecated and will be removed in the future releases.

Model Optimizer can register custom layers in a way that the output shape is calculated by the Caffe\* framework
installed on your system. This approach has several limitations:

* If your layer output shape depends on dynamic parameters, input data or previous layers parameters, calculation of
output shape of the layer via Caffe can be incorrect. For example, `SimplerNMS` is filtering out bounding boxes that do
not satisfy the condition. Internally, Caffe fallback forwards the whole net without any meaningful data - just some
noise. It is natural to get only one bounding box (0,0,0,0) instead of expected number (for example, 15). There is an
option to patch Caffe accordingly, however, it makes success of Intermediate Representation generation on the patched
Caffe on the particular machine. To keep the solution independent from Caffe, we recommend to use extensions mechanism
for such layers described in the [Model Optimizer Extensibility](Customize_Model_Optimizer.md).
* It is not possible to produce Intermediate Representation on a machine that does not have Caffe installed.

> **NOTE**: Caffe Python\* API has an issue when layer name does not correspond to the name of its top. The fix was
> implemented on [BVLC Caffe\*](https://github.com/BVLC/caffe/commit/35a7b87ad87457291dfc79bf8a7e7cf7ef278cbb). The
> Caffe framework on your computer must contain this fix. Otherwise, Caffe framework can unexpectedly fail during the
> fallback procedure.

> **NOTE**: The Caffe fallback feature was validated against [this GitHub revision](https://github.com/BVLC/caffe/tree/99466224dac86ddb86296b1e727794fb836bd80f). You may have issues with forks or later Caffe framework versions.

1.  Create a file `CustomLayersMapping.xml`:
```shell
mv extensions/front/caffe/CustomLayersMapping.xml.example extensions/front/caffe/CustomLayersMapping.xml
```
2.  Add (register) custom layers to `CustomLayersMapping.xml`:
```
\<CustomLayer NativeType="${Type}" hasParam="${has_params}" protoParamName="${layer_param}"/\>
```

Where:

*   `${Type}` is a type of the layer in the Caffe
*   `${has_params}` is "true" if the layer has parameters, and is "false" otherwise
*   `${layer_param}` is a name of the layer parameters in `caffe.proto` if the layer has it

**Example**:

1.  `Proposal` layer has parameters, and they appear in the Intermediate Representation. The parameters are stored in
the `proposal_param` property of the layer:
```shell
\<CustomLayer NativeType="Proposal" hasParam ="true" protoParamName = "proposal_param"/\> 
```
2.  CustomLayer layer has no parameters: 
```shell 
\<CustomLayer NativeType="CustomLayer" hasParam ="false"/\>
```

## Building Caffe\*

1.  Build Caffe\* with Python\* 3.5:
```shell
export CAFFE_HOME=PATH_TO_CAFFE
cd $CAFFE_HOME
rm -rf  ./build
mkdir ./build
cd ./build
cmake -DCPU_ONLY=ON -DOpenCV_DIR=<your opencv install dir> -DPYTHON_EXECUTABLE=/usr/bin/python3.5 ..
make all # also builds pycaffe
make install
make runtest # optional
```
2.  Add Caffe Python directory to `PYTHONPATH` to let it be imported from the Python program:
```shell
export PYTHONPATH=$CAFFE_HOME/python;$PYTHONPATH
```
3.  Check the Caffe installation:
```shell
python3
import caffe
```

If Caffe was installed correctly, the `caffe` module is imported without errors.
