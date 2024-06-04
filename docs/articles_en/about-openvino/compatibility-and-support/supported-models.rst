Supported Models
==============

.. toctree::
   :maxdepth: 1
   :hidden:

   about-openvino/performance-benchmarks
   about-openvino/compatibility-and-support
   Release Notes <about-openvino/release-notes-openvino>
   Additional Resources <about-openvino/additional-resources>

.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/openVinoDataTables.css">
   <h4>Here is just sample data.</h4>
   <label class="column-container">
       Hide topology
      <input type="checkbox" id="topology" name="topology" value="topology" data-column="0" class="toggle-vis"/>
      <label for="topology" class="checkmark"></label>
   </label>
   <label class="column-container">
      Hide source framework
      <input type="checkbox" id="source" name="source" value="source" data-column="1" class="toggle-vis"/>
      <label for="source" class="checkmark"></label>
   </label>
   <label class="column-container">
      Hide precision
      <input type="checkbox" id="precision" name="precision" value="precision" data-column="2" class="toggle-vis"/>
      <label for="precision" class="checkmark"></label>
   </label>
   <label class="column-container">
      Hide arc
      <input type="checkbox" id="arc" name="arc" value="arc" data-column="6" class="toggle-vis"/>
      <label for="arc" class="checkmark"></label>
   </label>

.. csv-table::
   :class: modeldata stripe
   :name: supportedModelsTable
   :header-rows: 1
   :file:  ../../_static/models.csv


