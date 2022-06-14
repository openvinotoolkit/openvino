# Converting an MXNet* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet}

## Convert an MXNet* Model <a name="ConvertMxNet"></a>
To convert an MXNet\* model, run Model Optimizer with a path to the input model `.params` file:

```sh
 mo --input_model model-file-0000.params
```

### Using MXNet\*-Specific Conversion Parameters <a name="mxnet_specific_conversion_params"></a>
The following list provides the MXNet\*-specific parameters.

```
MXNet-specific parameters:
  --input_symbol <SYMBOL_FILE_NAME>
            Symbol file (for example, "model-symbol.json") that contains a topology structure and layer attributes
  --nd_prefix_name <ND_PREFIX_NAME>
            Prefix name for args.nd and argx.nd files
  --pretrained_model_name <PRETRAINED_MODEL_NAME>
            Name of a pre-trained MXNet model without extension and epoch
            number. This model will be merged with args.nd and argx.nd
            files
  --save_params_from_nd
            Enable saving built parameters file from .nd files
  --legacy_mxnet_model
            Enable Apache MXNet loader to make a model compatible with the latest Apache MXNet version.
            Use only if your model was trained with Apache MXNet version lower than 1.0.0
  --enable_ssd_gluoncv
            Enable transformation for converting the gluoncv ssd topologies.
            Use only if your topology is one of ssd gluoncv topologies
```

> **NOTE**: By default, Model Optimizer does not use the Apache MXNet loader. It transforms the topology to another format which is compatible with the latest
> version of Apache MXNet. However, the Apache MXNet loader is required for models trained with lower version of Apache MXNet. If your model was trained with an Apache MXNet version lower than 1.0.0, specify the
> `--legacy_mxnet_model` key to enable the Apache MXNet loader. Note that the loader does not support models with custom layers. In this case, you must manually
> recompile Apache MXNet with custom layers and install it in your environment.

## Custom Layer Definition

Internally, when you run the Model Optimizer, it loads the model, goes through the topology, and tries to find each layer type in a list of known layers. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in this list of known layers, the Model Optimizer classifies them as custom.

## Supported MXNet\* Layers
Refer to [Supported Framework Layers](../Supported_Frameworks_Layers.md) for the list of supported standard layers.

## Frequently Asked Questions (FAQ)

The Model Optimizer provides explanatory messages if it is unable to run to completion due to issues like typographical errors, incorrectly used options, or other issues. The message describes the potential cause of the problem and gives a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md). The FAQ has instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## Summary

In this document, you learned:

* Basic information about how the Model Optimizer works with MXNet\* models
* Which MXNet\* models are supported
* How to convert a trained MXNet\* model using the Model Optimizer with both framework-agnostic and MXNet-specific command-line options

## See Also
[Model Conversion Tutorials](Convert_Model_Tutorials.md)
