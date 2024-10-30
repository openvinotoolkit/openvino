Table Question Answering using TAPAS and OpenVINO™
==================================================

Table Question Answering (Table QA) is the answering a question about an
information on a given table. You can use the Table Question Answering
models to simulate SQL execution by inputting a table.

In this tutorial we demonstrate how to perform table question answering
using OpenVINO. This example based on `TAPAS base model fine-tuned on
WikiTable Questions
(WTQ) <https://huggingface.co/google/tapas-base-finetuned-wtq>`__ that
is based on the paper `TAPAS: Weakly Supervised Table Parsing via
Pre-training <https://arxiv.org/abs/2004.02349#:~:text=Answering%20natural%20language%20questions%20over,denotations%20instead%20of%20logical%20forms>`__.

Answering natural language questions over tables is usually seen as a
semantic parsing task. To alleviate the collection cost of full logical
forms, one popular approach focuses on weak supervision consisting of
denotations instead of logical forms. However, training semantic parsers
from weak supervision poses difficulties, and in addition, the generated
logical forms are only used as an intermediate step prior to retrieving
the denotation. In `this
paper <https://arxiv.org/pdf/2004.02349.pdf>`__, it is presented TAPAS,
an approach to question answering over tables without generating logical
forms. TAPAS trains from weak supervision, and predicts the denotation
by selecting table cells and optionally applying a corresponding
aggregation operator to such selection. TAPAS extends BERT’s
architecture to encode tables as input, initializes from an effective
joint pre-training of text segments and tables crawled from Wikipedia,
and is trained end-to-end.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Use the original model to run an
   inference <#use-the-original-model-to-run-an-inference>`__
-  `Convert the original model to OpenVINO Intermediate Representation
   (IR)
   format <#convert-the-original-model-to-openvino-intermediate-representation-ir-format>`__
-  `Run the OpenVINO model <#run-the-openvino-model>`__
-  `Interactive inference <#interactive-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
~~~~~~~~~~~~~



.. code:: ipython3

    
    %pip install -q torch "transformers>=4.31.0" "torch>=2.1" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2023.2.0" "gradio>=4.0.2"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import torch
    from transformers import TapasForQuestionAnswering
    from transformers import TapasTokenizer
    from transformers import pipeline
    import pandas as pd


.. parsed-literal::

    2024-10-23 05:11:15.997414: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 05:11:16.031769: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 05:11:16.587915: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Use ``TapasForQuestionAnswering.from_pretrained`` to download a
pretrained model and ``TapasTokenizer.from_pretrained`` to get a
tokenizer.

.. code:: ipython3

    model = TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
    
    data = {
        "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
        "Number of movies": ["87", "53", "69"],
    }
    table = pd.DataFrame.from_dict(data)
    question = "how many movies does Leonardo Di Caprio have?"
    table




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Actors</th>
          <th>Number of movies</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Brad Pitt</td>
          <td>87</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Leonardo Di Caprio</td>
          <td>53</td>
        </tr>
        <tr>
          <th>2</th>
          <td>George Clooney</td>
          <td>69</td>
        </tr>
      </tbody>
    </table>
    </div>



Use the original model to run an inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



We use `this
example <https://huggingface.co/tasks/table-question-answering>`__ to
demonstrate how to make an inference. You can use ``pipeline`` from
``transformer`` library for this purpose.

.. code:: ipython3

    tqa = pipeline(task="table-question-answering", model=model, tokenizer=tokenizer)
    result = tqa(table=table, query=question)
    print(f"The answer is {result['cells'][0]}")


.. parsed-literal::

    The answer is 53


You can read more about the inference output structure in `this
documentation <https://huggingface.co/docs/transformers/model_doc/tapas>`__.

Convert the original model to OpenVINO Intermediate Representation (IR) format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The original model is a PyTorch module, that can be converted with
``ov.convert_model`` function directly. We also use ``ov.save_model``
function to serialize the result of conversion.

.. code:: ipython3

    import openvino as ov
    from pathlib import Path
    
    
    # Define the input shape
    batch_size = 1
    sequence_length = 29
    
    # Modify the input shape of the dummy_input dictionary
    dummy_input = {
        "input_ids": torch.zeros((batch_size, sequence_length), dtype=torch.long),
        "attention_mask": torch.zeros((batch_size, sequence_length), dtype=torch.long),
        "token_type_ids": torch.zeros((batch_size, sequence_length, 7), dtype=torch.long),
    }
    
    
    ov_model_xml_path = Path("models/ov_model.xml")
    
    if not ov_model_xml_path.exists():
        ov_model = ov.convert_model(model, example_input=dummy_input)
        ov.save_model(ov_model, ov_model_xml_path)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4779: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1571: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      self.indices = torch.as_tensor(indices)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1572: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      self.num_segments = torch.as_tensor(num_segments, device=indices.device)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1674: TracerWarning: torch.tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      batch_size = torch.prod(torch.tensor(list(index.batch_shape())))
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1750: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      [torch.as_tensor([-1], dtype=torch.long), torch.as_tensor(vector_shape, dtype=torch.long)], dim=0
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1753: TracerWarning: Converting a tensor to a Python list might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      flat_values = values.reshape(flattened_shape.tolist())
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1755: TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      out = torch.zeros(int(flat_index.num_segments), dtype=torch.float, device=flat_values.device)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1763: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      torch.as_tensor(index.batch_shape(), dtype=torch.long),
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1764: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      torch.as_tensor([index.num_segments], dtype=torch.long),
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1765: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      torch.as_tensor(vector_shape, dtype=torch.long),
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1770: TracerWarning: Converting a tensor to a Python list might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      output_values = segment_means.clone().view(new_shape.tolist()).to(values.dtype)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1701: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      batch_shape = torch.as_tensor(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1705: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      num_segments = torch.as_tensor(num_segments)  # create a rank 0 tensor (scalar) containing num_segments (e.g. 64)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1716: TracerWarning: Converting a tensor to a Python list might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      new_shape = [int(x) for x in new_tensor.tolist()]
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1719: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      multiples = torch.cat([batch_shape, torch.as_tensor([1])], dim=0)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1720: TracerWarning: Converting a tensor to a Python list might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      indices = indices.repeat(multiples.tolist())
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:282: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      torch.as_tensor(self.config.max_position_embeddings - 1, device=device), position - first_position
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1231: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      indices=torch.min(row_ids, torch.as_tensor(self.config.max_num_rows - 1, device=row_ids.device)),
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1236: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      indices=torch.min(column_ids, torch.as_tensor(self.config.max_num_columns - 1, device=column_ids.device)),
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1928: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1933: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      column_logits += CLOSE_ENOUGH_TO_LOG_ZERO * torch.as_tensor(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1969: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      labels_per_column, _ = reduce_sum(torch.as_tensor(labels, dtype=torch.float32, device=labels.device), col_index)
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1992: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      torch.as_tensor(labels, dtype=torch.long, device=labels.device), cell_index
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:1999: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      column_mask = torch.as_tensor(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:2024: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      selected_column_id = torch.as_tensor(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/tapas/modeling_tapas.py:2029: TracerWarning: torch.as_tensor results are registered as constants in the trace. You can safely ignore this warning if you use this function to create tensors out of constant variables that would be the same every time you call this function. In any other case, this might cause the trace to be incorrect.
      selected_column_mask = torch.as_tensor(


Run the OpenVINO model
~~~~~~~~~~~~~~~~~~~~~~



Select a device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



We use ``ov.compile_model`` to make it ready to use for loading on a
device. To prepare inputs use the original ``tokenizer``.

.. code:: ipython3

    core = ov.Core()
    
    inputs = tokenizer(table=table, queries=question, padding="max_length", return_tensors="pt")
    
    compiled_model = core.compile_model(ov_model_xml_path, device.value)
    result = compiled_model((inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]))

Now we should postprocess results. For this, we can use the appropriate
part of the code from
`postprocess <https://github.com/huggingface/transformers/blob/fe2877ce21eb75d34d30664757e2727d7eab817e/src/transformers/pipelines/table_question_answering.py#L393>`__
method of ``TableQuestionAnsweringPipeline``.

.. code:: ipython3

    logits = result[0]
    logits_aggregation = result[1]
    
    
    predictions = tokenizer.convert_logits_to_predictions(inputs, torch.from_numpy(result[0]))
    answer_coordinates_batch = predictions[0]
    aggregators = {}
    aggregators_prefix = {}
    answers = []
    for index, coordinates in enumerate(answer_coordinates_batch):
        cells = [table.iat[coordinate] for coordinate in coordinates]
        aggregator = aggregators.get(index, "")
        aggregator_prefix = aggregators_prefix.get(index, "")
        answer = {
            "answer": aggregator_prefix + ", ".join(cells),
            "coordinates": coordinates,
            "cells": [table.iat[coordinate] for coordinate in coordinates],
        }
        if aggregator:
            answer["aggregator"] = aggregator
    
        answers.append(answer)
    
    print(answers[0]["cells"][0])


.. parsed-literal::

    53


Also, we can use the original pipeline. For this, we should create a
wrapper for ``TapasForQuestionAnswering`` class replacing ``forward``
method to use the OpenVINO model for inference and methods and
attributes of original model class to be integrated into the pipeline.

.. code:: ipython3

    from transformers import TapasConfig
    
    
    # get config for pretrained model
    config = TapasConfig.from_pretrained("google/tapas-large-finetuned-wtq")
    
    
    class TapasForQuestionAnswering(TapasForQuestionAnswering):  # it is better to keep the class name to avoid warnings
        def __init__(self, ov_model_path):
            super().__init__(config)  # pass config from the pretrained model
            self.tqa_model = core.compile_model(ov_model_path, device.value)
    
        def forward(self, input_ids, *, attention_mask, token_type_ids):
            results = self.tqa_model((input_ids, attention_mask, token_type_ids))
    
            return torch.from_numpy(results[0]), torch.from_numpy(results[1])
    
    
    compiled_model = TapasForQuestionAnswering(ov_model_xml_path)
    tqa = pipeline(task="table-question-answering", model=compiled_model, tokenizer=tokenizer)
    print(tqa(table=table, query=question)["cells"][0])


.. parsed-literal::

    53


Interactive inference
~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import pandas as pd
    
    
    def highlight_answers(x, coordinates):
        highlighted_table = pd.DataFrame("", index=x.index, columns=x.columns)
        for coordinates_i in coordinates:
            highlighted_table.iloc[coordinates_i[0], coordinates_i[1]] = "background-color: lightgreen"
    
        return highlighted_table
    
    
    def infer(query, csv_file_name):
        table = pd.read_csv(csv_file_name.name, delimiter=",")
        table = table.astype(str)
    
        result = tqa(table=table, query=query)
        table = table.style.apply(highlight_answers, axis=None, coordinates=result["coordinates"])
    
        return result["answer"], table

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/table-question-answering/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(fn=infer)
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.








.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
