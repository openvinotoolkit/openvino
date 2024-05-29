.. {#supported_models}
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

   <link rel="stylesheet" type="text/css" href="_static/css/openVinoDataTables.css">
   <div>
   <input type="checkbox" id="topology" name="interest" value="topology" data-column="0" class="toggle-vis"/>
   <label for="topology" >Hide topology</label>
   </div>
   <div>
   <input type="checkbox" id="precision" name="interest" value="precision" data-column="2" class="toggle-vis"/>
   <label for="precision" >Hide precision</label>
   </div>
   <div>
   <input type="checkbox" disabled id="source" name="source" value="source" data-column="1" class="toggle-vis"/>
   <label for="source" >Hide source framework</label>
   </div>


.. csv-table::
   :class: modeldata
   :name: id-of-table
   :header-rows: 1
   :file: ../../../_static/models.csv


