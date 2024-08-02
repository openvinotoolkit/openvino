Supported Models
========================

.. toctree::
   :maxdepth: 1
   :hidden:

   about-openvino/performance-benchmarks
   about-openvino/compatibility-and-support
   Release Notes <about-openvino/release-notes-openvino>
   Additional Resources <about-openvino/additional-resources>

.. raw:: html
      
   <link rel="stylesheet" type="text/css" href="../../_static/css/openVinoDataTables.css">
   <label>hide/reveal additional columns:</label><br/>
   <label class="column-container">
       Topology
      <input type="checkbox" id="topology" name="topology" value="topology" data-column="0" class="toggle-vis"/>
      <label for="topology" class="checkmark"></label>
   </label>
   <label class="column-container">
      Source framework
      <input type="checkbox" checked id="source" name="source" value="source" data-column="1" class="toggle-vis"/>
      <label for="source" class="checkmark"></label>
   </label>
   <label class="column-container">
      Precision
      <input type="checkbox" checked id="precision" name="precision" value="precision" data-column="2" class="toggle-vis"/>
      <label for="precision" class="checkmark"></label>
   </label>
   <label class="column-container">
      Arc
      <input type="checkbox" checked id="arc" name="arc" value="arc" data-column="6" class="toggle-vis"/>
      <label for="arc" class="checkmark"></label>
   </label>

.. csv-table::
   :class: modeldata stripe
   :name: supportedModelsTable
   :header-rows: 1
   :file:  ../../_static/models.csv


