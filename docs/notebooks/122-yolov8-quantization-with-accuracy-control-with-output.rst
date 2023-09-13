Convert and Optimize YOLOv8 with OpenVINO™
==========================================



The YOLOv8 algorithm developed by Ultralytics is a cutting-edge,
state-of-the-art (SOTA) model that is designed to be fast, accurate, and
easy to use, making it an excellent choice for a wide range of object
detection, image segmentation, and image classification tasks. More
details about its realization can be found in the original model
`repository <https://github.com/ultralytics/ultralytics>`__.

This tutorial demonstrates step-by-step instructions on how to run apply
quantization with accuracy control to PyTorch YOLOv8. The advanced
quantization flow allows to apply 8-bit quantization to the model with
control of accuracy metric. This is achieved by keeping the most
impactful operations within the model in the original precision. The
flow is based on the `Basic 8-bit
quantization <https://docs.openvino.ai/2023.0/basic_quantization_flow.html>`__
and has the following differences:

-  Besides the calibration dataset, a validation dataset is required to
   compute the accuracy metric. Both datasets can refer to the same data
   in the simplest case.
-  Validation function, used to compute accuracy metric is required. It
   can be a function that is already available in the source framework
   or a custom function.
-  Since accuracy validation is run several times during the
   quantization process, quantization with accuracy control can take
   more time than the Basic 8-bit quantization flow.
-  The resulted model can provide smaller performance improvement than
   the Basic 8-bit quantization flow because some of the operations are
   kept in the original precision.

.. note::

   Currently, 8-bit quantization with accuracy control in NNCF
   is available only for models in OpenVINO representation.

The steps for the quantization with accuracy control are described
below.

The tutorial consists of the following steps:



-  `Prerequisites <#prerequisites>`__
-  `Get Pytorch model and OpenVINO IR model <#get-pytorch-model-and-openvino-ir-model>`__
-  `Define validator and data loader <#define-validator-and-data-loader>`__
-  `Prepare calibration and validation datasets <#prepare-calibration-and-validation-datasets>`__
-  `Prepare validation function <#prepare-validation-function>`__
-  `Run quantization with accuracy control <#run-quantization-with-accuracy-control>`__
-  `Compare Accuracy and Performance of the Original and Quantized Models <#compare-accuracy-and-performance-of-the-original-and-quantized-models>`__

Prerequisites `⇑ <#top>`__
###############################################################################################################################


Install necessary packages.

.. code:: ipython2

    !pip install -q "openvino==2023.1.0.dev20230811"
    !pip install git+https://github.com/openvinotoolkit/nncf.git@develop
    !pip install -q "ultralytics==8.0.43"

Get Pytorch model and OpenVINO IR model `⇑ <#top>`__
###############################################################################################################################

Generally, PyTorch models represent an instance of the
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the YOLOv8 nano model (also known as ``yolov8n``) pre-trained on a COCO
dataset, which is available in this
`repo <https://github.com/ultralytics/ultralytics>`__. Similar steps are
also applicable to other YOLOv8 models. Typical steps to obtain a
pre-trained model:

1. Create an instance of a model class.
2. Load a checkpoint state dict, which contains the pre-trained model
   weights.

In this case, the creators of the model provide an API that enables
converting the YOLOv8 model to ONNX and then to OpenVINO IR. Therefore,
we do not need to do these steps manually.

.. code:: ipython2

    import os
    from pathlib import Path

    from ultralytics import YOLO
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.data.utils import check_det_dataset
    from ultralytics.yolo.engine.validator import BaseValidator as Validator
    from ultralytics.yolo.utils import DATASETS_DIR
    from ultralytics.yolo.utils import DEFAULT_CFG
    from ultralytics.yolo.utils import ops
    from ultralytics.yolo.utils.metrics import ConfusionMatrix

    ROOT = os.path.abspath('')

    MODEL_NAME = "yolov8n-seg"

    model = YOLO(f"{ROOT}/{MODEL_NAME}.pt")
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128-seg.yaml"

Load model.

.. code:: ipython2

    import openvino


    model_path = Path(f"{ROOT}/{MODEL_NAME}_openvino_model/{MODEL_NAME}.xml")
    if not model_path.exists():
        model.export(format="openvino", dynamic=True, half=False)

    ov_model = openvino.Core().read_model(model_path)

Define validator and data loader `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The original model
repository uses a ``Validator`` wrapper, which represents the accuracy
validation pipeline. It creates dataloader and evaluation metrics and
updates metrics on each data batch produced by the dataloader. Besides
that, it is responsible for data preprocessing and results
postprocessing. For class initialization, the configuration should be
provided. We will use the default setup, but it can be replaced with
some parameters overriding to test on custom data. The model has
connected the ``ValidatorClass`` method, which creates a validator class
instance.

.. code:: ipython2

    validator = model.ValidatorClass(args)
    validator.data = check_det_dataset(args.data)
    data_loader = validator.get_dataloader(f"{DATASETS_DIR}/coco128-seg", 1)

    validator.is_coco = True
    validator.class_map = ops.coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc
    validator.nm = 32
    validator.process = ops.process_mask
    validator.plot_masks = []

Prepare calibration and validation datasets `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We can use one dataset as calibration and validation datasets. Name it
``quantization_dataset``.

.. code:: ipython2

    from typing import Dict

    import nncf


    def transform_fn(data_item: Dict):
        input_tensor = validator.preprocess(data_item)["img"].numpy()
        return input_tensor


    quantization_dataset = nncf.Dataset(data_loader, transform_fn)

Prepare validation function `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython2

    from functools import partial

    import torch
    from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters


    def validation_ac(
        compiled_model: openvino.CompiledModel,
        validation_loader: torch.utils.data.DataLoader,
        validator: Validator,
        num_samples: int = None,
    ) -> float:
        validator.seen = 0
        validator.jdict = []
        validator.stats = []
        validator.batch_i = 1
        validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
        num_outputs = len(compiled_model.outputs)

        counter = 0
        for batch_i, batch in enumerate(validation_loader):
            if num_samples is not None and batch_i == num_samples:
                break
            batch = validator.preprocess(batch)
            results = compiled_model(batch["img"])
            if num_outputs == 1:
                preds = torch.from_numpy(results[compiled_model.output(0)])
            else:
                preds = [
                    torch.from_numpy(results[compiled_model.output(0)]),
                    torch.from_numpy(results[compiled_model.output(1)]),
                ]
            preds = validator.postprocess(preds)
            validator.update_metrics(preds, batch)
            counter += 1
        stats = validator.get_stats()
        if num_outputs == 1:
            stats_metrics = stats["metrics/mAP50-95(B)"]
        else:
            stats_metrics = stats["metrics/mAP50-95(M)"]
        print(f"Validate: dataset length = {counter}, metric value = {stats_metrics:.3f}")

        return stats_metrics


    validation_fn = partial(validation_ac, validator=validator)

Run quantization with accuracy control `⇑ <#top>`__
###############################################################################################################################

You should provide
the calibration dataset and the validation dataset. It can be the same
dataset. - parameter ``max_drop`` defines the accuracy drop threshold.
The quantization process stops when the degradation of accuracy metric
on the validation dataset is less than the ``max_drop``. The default
value is 0.01. NNCF will stop the quantization and report an error if
the ``max_drop`` value can’t be reached. - ``drop_type`` defines how the
accuracy drop will be calculated: ABSOLUTE (used by default) or
RELATIVE. - ``ranking_subset_size`` - size of a subset that is used to
rank layers by their contribution to the accuracy drop. Default value is
300, and the more samples it has the better ranking, potentially. Here
we use the value 25 to speed up the execution.

.. note::

   Execution can take tens of minutes and requires up to 15 GB
   of free memory

.. code:: ipython2

    quantized_model = nncf.quantize_with_accuracy_control(
        ov_model,
        quantization_dataset,
        quantization_dataset,
        validation_fn=validation_fn,
        max_drop=0.01,
        preset=nncf.QuantizationPreset.MIXED,
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(
            ranking_subset_size=25,
            num_ranking_processes=1
        ),
    )

Compare Accuracy and Performance of the Original and Quantized Models `⇑ <#top>`__
###############################################################################################################################


Now we can compare metrics of the Original non-quantized
OpenVINO IR model and Quantized OpenVINO IR model to make sure that the
``max_drop`` is not exceeded.

.. code:: ipython2

    import openvino

    core = openvino.Core()
    quantized_compiled_model = core.compile_model(model=quantized_model, device_name='CPU')
    compiled_ov_model = core.compile_model(model=ov_model, device_name='CPU')

    pt_result = validation_ac(compiled_ov_model, data_loader, validator)
    quantized_result = validation_ac(quantized_compiled_model, data_loader, validator)


    print(f'[Original OpenVino]: {pt_result:.4f}')
    print(f'[Quantized OpenVino]: {quantized_result:.4f}')

And compare performance.

.. code:: ipython2

    from pathlib import Path
    # Set model directory
    MODEL_DIR = Path("model")
    MODEL_DIR.mkdir(exist_ok=True)

    ir_model_path = MODEL_DIR / 'ir_model.xml'
    quantized_model_path = MODEL_DIR / 'quantized_model.xml'

    # Save models to use them in the commandline banchmark app
    openvino.save_model(ov_model, ir_model_path, compress_to_fp16=False)
    openvino.save_model(quantized_model, quantized_model_path, compress_to_fp16=False)

.. code:: ipython2

    # Inference Original model (OpenVINO IR)
    ! benchmark_app -m $ir_model_path -shape "[1,3,640,640]" -d CPU -api async

.. code:: ipython2

    # Inference Quantized model (OpenVINO IR)
    ! benchmark_app -m $quantized_model_path -shape "[1,3,640,640]" -d CPU -api async
