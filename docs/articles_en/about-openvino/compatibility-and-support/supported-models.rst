Supported Models
========================

The following table lists a selection of models that are validated against various hardware
devices. The list includes only models used in validation, other models from frameworks supported
by OpenVINO may also work properly.


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
      <input type="checkbox" checked id="AI PC GPU" name="AI PC GPU" value="AI PC GPU" data-column="4" class="toggle-vis"/>
      <label for="AI PC GPU" class="checkmark"></label>
   </label>
   <label class="column-container">
      NPU
      <input type="checkbox" checked id="AI PC NPU" name="AI PC NPU" value="AI PC NPU" data-column="5" class="toggle-vis"/>
      <label for="AI PC NPU" class="checkmark"></label>
   </label>


.. csv-table::
   :class: modeldata stripe
   :name: supportedModelsTable
   :header-rows: 1
   :file:  ../../_static/download/supported_models.csv


| Note:
| The validation process involves using OpenVINO, natively or as a backend, to load each model
  onto the designated hardware and execute inference. If no errors are reported and inference
  finishes, the model receives the **passed** status (indicated by a check mark in the table).
  The models that are not tested are indicated by **empty** status cells.

| The models come from different public model repositories, such as, OpenVINO Model Zoo,
  ONNX Model Zoo, Pytorch Model Zoo, and HuggingFace.

| In the precision column, optimum-intel default corresponds to FP32 for small models and INT8
  for models greater than 1B parameters.