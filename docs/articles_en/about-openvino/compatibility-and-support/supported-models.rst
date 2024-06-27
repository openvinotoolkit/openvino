AI Models that run with OpenVINO
=====================================

This list of 500+ computer vision, LLM, and GenAI models includes pre-optimized OpenVINO
models/notebooks plus models from public model zoos (ONNX Model Zoo, Pytorch Model Zoo, and
HuggingFace). This list is not comprehensive and only includes models tested by Intel.

**AI Models that run on Intel® Core Ultra ™ Processors with OpenVINO™ toolkit:**

.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/openVinoDataTables.css">
   <label>hide/reveal additional columns:</label><br/>
   <label class="column-container">
      CPU
      <input type="checkbox" id="AI PC CPU" name="AI PC CPU" value="AI PC CPU" data-column="3" class="toggle-vis"/>
      <label for="AI PC CPU" class="checkmark"></label>
   </label>
   <label class="column-container">
      GPU
      <input type="checkbox" id="AI PC GPU" name="AI PC GPU" value="AI PC GPU" data-column="4" class="toggle-vis"/>
      <label for="AI PC GPU" class="checkmark"></label>
   </label>
   <label class="column-container">
      NPU
      <input type="checkbox" id="AI PC NPU" name="AI PC NPU" value="AI PC NPU" data-column="5" class="toggle-vis"/>
      <label for="AI PC NPU" class="checkmark"></label>
   </label>


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
