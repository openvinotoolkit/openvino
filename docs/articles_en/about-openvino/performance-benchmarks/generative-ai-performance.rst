Most Efficient Large Language Models for AI PC
==============================================

The table below lists key performance indicators for a selection of Large Language Models
running on an Intel® Core™ Ultra 7-165H based system.

.. raw:: html

   <label><link rel="stylesheet" type="text/css" href="../../_static/css/openVinoDataTables.css"></label>
   <br/><label>hide/reveal additional columns:</label><br/>
   <label class="column-container">
       Token latency
       <input type="checkbox" checked id="1st" name="1st" value="1st" data-column="2" class="toggle-vis"/>
       <label for="1st" class="checkmark"></label>
    </label>
    <label class="column-container">
       Memory used
       <input type="checkbox" checked id="maxrss" name="maxrss" value="maxrss" data-column="3" class="toggle-vis"/>
       <label for="maxrss" class="checkmark"></label>
    </label>
    <label class="column-container">
       Input tokens
       <input type="checkbox" checked id="input" name="input" value="input" data-column="4" class="toggle-vis"/>
       <label for="input" class="checkmark"></label>
    </label>
    <label class="column-container">
       Output tokens
       <input type="checkbox" checked id="output" name="output" value="output" data-column="5" class="toggle-vis"/>
       <label for="output" class="checkmark"></label>
    </label>
    <label class="column-container">
       Model precision
       <input type="checkbox" checked id="precision" name="precision" value="precision" data-column="6" class="toggle-vis"/>
       <label for="precision" class="checkmark"></label>
    </label>
    <label class="column-container">
       Beam
       <input type="checkbox" checked id="beam" name="beam" value="beam" data-column="7" class="toggle-vis"/>
       <label for="beam" class="checkmark"></label>
    </label>
    <label class="column-container">
       Batch size
       <input type="checkbox" checked id="batch" name="batch" value="batch" data-column="8" class="toggle-vis"/>
       <label for="batch" class="checkmark"></label>
    </label>
    <label class="column-container">
       Framework
       <input type="checkbox" checked id="framework" name="framework" value="framework" data-column="9" class="toggle-vis"/>
       <label for="framework" class="checkmark"></label>
    </label>


.. csv-table::
   :class: modeldata stripe
   :name: supportedModelsTable
   :header-rows: 1
   :file:  ../../_static/llm_models.csv


This page is regularly updated to help you identify the best-performing LLMs on the
Intel® Core™ Ultra processor family and AI PCs.

For complete information on the system config, see:
`Hardware Platforms [PDF] <https://docs.openvino.ai/2024/_static/benchmarks_files/OV-2024.2-platform_list.pdf>`__