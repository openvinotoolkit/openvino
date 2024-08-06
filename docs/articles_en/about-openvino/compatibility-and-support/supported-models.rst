AI Models verified for OpenVINO™
=====================================

The following is a list of models that have been verified to work with OpenVINO. Note that other
models from OpenVINO-supported frameworks may also work properly but have not been tested.

**AI Models that run on Intel® Core Ultra™ Processors with OpenVINO™ toolkit:**

.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/openVinoDataTables.css">


.. csv-table::
   :class: modeldata stripe
   :name: supportedModelsTable
   :header-rows: 1
   :file:  ../../_static/download/supported_models.csv


| Marked cells indicate models that passed inference with no errors. Empty cells indicate
  models that were not tested. No failing runs producing an error have been recorded.
|
| In the precision column, the "optimum-intel default" label corresponds to FP32 for small models
  and INT8 for models greater than 1B parameters.
|
| The results as of June 17 2024, for OpenVINO version 2024.2.
| The models come from different public model repositories, such as Pytorch Model Zoo and
  HuggingFace; they were executed on the designated hardware with OpenVINO either natively or
  as a backend.

