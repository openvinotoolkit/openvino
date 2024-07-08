AI Models that run with OpenVINO
=====================================

This list of 500+ computer vision, LLM, and GenAI models includes pre-optimized OpenVINO
models/notebooks plus models from public model zoos (ONNX Model Zoo, Pytorch Model Zoo, and
HuggingFace). This list is not comprehensive and only includes models tested by Intel.

**AI Models that run on Intel® Core Ultra ™ Processors with OpenVINO™ toolkit:**

.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/openVinoDataTables.css">


.. csv-table::
   :class: modeldata stripe
   :name: supportedModelsTable
   :header-rows: 1
   :file:  ../../_static/download/supported_models.csv


| Note:
| The results as of June 17 2024, for OpenVINO version 2024.2.

| The validation process involved using the OpenVINO toolkit (natively or as a backend) to load
  each model onto the designated hardware and successfully execute inference without encountering
  any errors or failures. These successfully run models are indicated with a check mark below and
  blanks are not tested. In the precision column where it is listed as optimum-intel default,
  that corresponds to FP32 for small models and INT8 for models greater than 1B parameters.
