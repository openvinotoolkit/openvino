Generative AI Benchmark Results
===================================

This page will give you a detailed information on how OpenVINO performs running a selection of
Generative AI models.


.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/openVinoDataTables.css">

   <label class="column-container">
        Hide 1st token latency
       <input type="checkbox" id="1st" name="1st" value="1st" data-column="2" class="toggle-vis"/>
       <label for="1st" class="checkmark"></label>
    </label>
    <label class="column-container">
       Hide Max_RSS_memory used
       <input type="checkbox" id="maxrss" name="maxrss" value="maxrss" data-column="3" class="toggle-vis"/>
       <label for="maxrss" class="checkmark"></label>
    </label>
    <label class="column-container">
       Hide Input tokens
       <input type="checkbox" id="input" name="input" value="input" data-column="4" class="toggle-vis"/>
       <label for="input" class="checkmark"></label>
    </label>
    <label class="column-container">
       Hide Output tokens
       <input type="checkbox" id="output" name="output" value="output" data-column="5" class="toggle-vis"/>
       <label for="output" class="checkmark"></label>
    </label>
    <label class="column-container">
       Hide Model Precision
       <input type="checkbox" id="precision" name="precision" value="precision" data-column="6" class="toggle-vis"/>
       <label for="precision" class="checkmark"></label>
    </label>
    <label class="column-container">
       Hide Beam
       <input type="checkbox" id="beam" name="beam" value="beam" data-column="7" class="toggle-vis"/>
       <label for="beam" class="checkmark"></label>
    </label>
    <label class="column-container">
       Hide Batch size
       <input type="checkbox" id="batch" name="batch" value="batch" data-column="8" class="toggle-vis"/>
       <label for="batch" class="checkmark"></label>
    </label>
    <label class="column-container">
       Hide Framework
       <input type="checkbox" id="framework" name="framework" value="framework" data-column="9" class="toggle-vis"/>
       <label for="framework" class="checkmark"></label>
    </label>

.. csv-table::
   :class: modeldata stripe
   :name: supportedModelsTable
   :header-rows: 1
   :file:  ../../_static/llm_models.csv




