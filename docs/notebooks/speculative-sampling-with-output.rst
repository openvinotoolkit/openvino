Text Generation via Speculative Decoding using FastDraft and OpenVINO™
======================================================================

As model sizes grow, Generative AI implementations require significant
inference resources. This not only increases the cost per generation
from a prompt, but also increases the power consumption used to serve
such requests.

Inference optimizations for text generation are essential for reducing
costs and power consumption. When optimizing the inference process, the
amount of time and energy required to generate text can be significantly
reduced. This can lead to cost savings in terms of hardware and
software, as well as reduced power consumption. Additionally, inference
optimizations can help improve the accuracy of text generation as well
as the speed at which it can be generated. This can lead to an improved
user experience and increased efficiency in text-generation tasks. In
summary, inference optimizations for text generation are essential to
reduce costs and power consumption, while also improving the accuracy
and speed of text generation.

Speculative decoding (or
`assisted-generation <https://huggingface.co/blog/assisted-generation#understanding-text-generation-latency>`__)
is a recent technique, that allows to speed up token generation when an
additional smaller draft model is used alongside with the main model.

Speculative decoding works the following way. The draft model predicts
the next K tokens one by one in an autoregressive manner, while the main
model validates these predictions and corrects them if necessary. We go
through each predicted token, and if a difference is detected between
the draft and main model, we stop and keep the last token predicted by
the main model. Then the draft model gets the latest main prediction and
again tries to predict the next K tokens, repeating the cycle.

This approach reduces the need for multiple infer requests to the main
model, enhancing performance. For instance, in more predictable parts of
text generation, the draft model can, in best-case scenarios, generate
the next K tokens that exactly match the target. In that case they are
validated in a single inference request to the main model (which is
bigger, more accurate but slower) instead of running K subsequent
requests. More details can be found in the original
`paper <https://arxiv.org/pdf/2211.17192.pdf>`__.

|image0|

Possibility to achieve significant speedup with Speculative Decoding is
highly depends on selection of a high-quality draft model that is both
efficient and well-aligned with the target. FastDraft is a novel and
efficient approach for pre-training and aligning a draft model to any
LLM to be used with speculative decoding, by incorporating efficient
pre-training followed by fine-tuning over synthetic datasets generated
by the target model. FastDraft was presented in the
`paper <https://arxiv.org/abs/2411.11055>`__ at
`ENLSP@NeurIPS24 <https://neurips2024-enlsp.github.io/accepted_papers.html>`__
by Intel Labs.

FastDraft pre-trained draft models achieve impressive results in key
metrics of acceptance rate, block efficiency and up to 3x memory bound
speed up when evaluated on code completion and up to 2x in
summarization, text completion and instruction tasks and unlock large
language models inference on AI-PC and other edge-devices.

In this tutorial we consider how to apply Speculative decoding using
FastDraft and OpenVINO GenAI.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Prepare models <#prepare-models>`__

   -  `Select inference device <#select-inference-device>`__

-  `Run target model without speculative
   decoding <#run-target-model-without-speculative-decoding>`__
-  `Run Speculative decoding
   pipeline <#run-speculative-decoding-pipeline>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |image0| image:: https://github.com/user-attachments/assets/eb999dea-d98b-42bb-835e-28d3054e1a84

Prerequisites
-------------



First, we should install the `OpenVINO
GenAI <https://github.com/openvinotoolkit/openvino.genai>`__ for running
model inference.

|image01|

`OpenVINO™ GenAI <https://github.com/openvinotoolkit/openvino.genai>`__
is a library of the most popular Generative AI model pipelines,
optimized execution methods, and samples that run on top of highly
performant `OpenVINO
Runtime <https://github.com/openvinotoolkit/openvino>`__.

This library is friendly to PC and laptop execution, and optimized for
resource consumption. It requires no external dependencies to run
generative models as it already includes all the core functionality
(e.g. tokenization via openvino-tokenizers).

.. |image01| image:: https://media.githubusercontent.com/media/openvinotoolkit/openvino.genai/refs/heads/master/src/docs/openvino_genai.svg

.. code:: ipython3

    %pip install -Uq "openvino>=2024.5.0" "openvino-tokenizers>=2024.5.0" "openvino-genai>=2024.5.0" huggingface_hub

Prepare models
--------------



As example, we will use already converted LLMs from `OpenVINO
collection <https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd>`__.
You can find OpenVINO optimized FastDraft models can be found in this
`collection <https://huggingface.co/collections/OpenVINO/speculative-decoding-draft-models-673f5d944d58b29ba6e94161>`__.
As example we will use
`Phi-3-mini-4k-instruct-int4-ov <https://huggingface.co/OpenVINO/Phi-3-mini-4k-instruct-int4-ov>`__
as target model and
`Phi-3-mini-FastDraft-50M-int8-ov <https://huggingface.co/OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov>`__
as draft.

In case, if you want run own models, you should convert them using
`Hugging Face
Optimum <https://huggingface.co/docs/optimum/intel/openvino/export>`__
library accelerated by OpenVINO integration. More details about model
preparation can be found in `OpenVINO LLM inference
guide <https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide/llm-inference-native-ov.html#convert-hugging-face-tokenizer-and-model-to-openvino-ir-format>`__

.. code:: ipython3

    from pathlib import Path
    import huggingface_hub as hf_hub

    draft_model_id = "OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov"
    target_model_id = "OpenVINO/Phi-3-mini-4k-instruct-int4-ov"

    draft_model_path = Path(draft_model_id.split("/")[-1])
    target_model_path = Path(target_model_id.split("/")[-1])

    if not draft_model_path.exists():
        hf_hub.snapshot_download(draft_model_id, local_dir=draft_model_path)
    if not target_model_path.exists():
        hf_hub.snapshot_download(target_model_id, local_dir=target_model_path)

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



Select the device from dropdown list for running inference using
OpenVINO. > **Note**: For achieving maximal performance, we recommend to
use GPU as target device if it is available.

.. code:: ipython3

    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

    from notebook_utils import device_widget

    device = device_widget(default="CPU", exclude=["NPU", "AUTO"])

    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU',), value='CPU')



Run target model without speculative decoding
---------------------------------------------



OpenVINO GenAI provides easy-to-use API for running text generation.
Firstly we will create pipeline with ``LLMPipeline``. ``LLMPipeline`` is
the main object used for decoding. You can construct it straight away
from the folder with the converted model. It will automatically load the
``main model``, ``tokenizer``, ``detokenizer`` and default
``generation configuration``. After that we will configure parameters
for decoding. Then we just run ``generate`` method and get the output in
text format. We do not need to encode input prompt according to model
expected template or write post-processing code for logits decoder, it
will be done easily with LLMPipeline.

To obtain intermediate generation results without waiting until when
generation is finished, we will write streamer function.

.. code:: ipython3

    import openvino_genai as ov_genai
    import time

    pipe = ov_genai.LLMPipeline(target_model_path, device.value)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100


    def streamer(subword):
        print(subword, end="", flush=True)
        # Return flag corresponds whether generation should be stopped.
        # False means continue generation.
        return False


    start_time = time.perf_counter()
    pipe.generate(["Sun is yellow because"], config, streamer=streamer)
    end_time = time.perf_counter()


.. parsed-literal::

     it is made of gas. The gas is heated to a high temperature and then cooled. The gas is yellow because it has a band of light called the "Bondeson Pendulum Effect." The Bondeson Pendulum Effect is caused by the light waves bouncing off of the gas molecules. The light waves bounce off of the gas molecules in different ways, some of the light waves get scattered, and some of the light waves get reflected. The light waves that get scattered and reflected combine to

.. code:: ipython3

    import gc

    print(f"Generation time: {end_time - start_time:.2f}s")
    del pipe
    gc.collect();


.. parsed-literal::

    Generation time: 18.44s


Run Speculative decoding pipeline
---------------------------------



To enable Speculative decoding in ``LLMPipeline,`` we should
additionally provide the ``draft_model`` structure and
``SchedulerConfig`` for resource management.

|image02|

As shown in the figure above, speculative decoding works by splitting
the generative process into two stages. In the first stage, a fast, but
less accurate draft model (AKA assistant) autoregressively generates a
sequence of tokens. In the second stage, a large, but more accurate
target model conducts parallelized verification over the generated draft
tokens. This process allows the target model to produce multiple tokens
in a single forward pass and thus accelerate autoregressive decoding.
The success of speculative decoding largely hinges on the speculation
lookahead (SL), i.e. the number of tokens produced by the draft model in
each iteration. The straightforward method, based on `Leviathan et
al. <https://arxiv.org/pdf/2211.17192>`__, uses a static value of the
speculation lookahead and involves generating a constant number of
candidate tokens at each speculative iteration. You can adjust the
number of candidates using ``num_assistant_tokens`` parameter in
generation config. If the assistant model’s confidence in its prediction
for the current token is lower than this threshold, the assistant model
stops the current token generation iteration is not yet reached.

.. |image02| image:: https://github.com/user-attachments/assets/69f5c096-abca-4f97-952b-291c52eb3444

.. code:: ipython3

    scheduler_config = ov_genai.SchedulerConfig()
    # cache params
    scheduler_config.cache_size = 2

    draft_model = ov_genai.draft_model(draft_model_path, device.value)

    pipe = ov_genai.LLMPipeline(target_model_path, device.value, draft_model=draft_model, scheduler_config=scheduler_config)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100
    config.num_assistant_tokens = 3
    start_time = time.perf_counter()
    result = pipe.generate(["Sun is yellow because"], config, streamer=streamer)
    end_time = time.perf_counter()


.. parsed-literal::

     it is made of gas. The gas is heated to a high temperature and then cooled. The gas changes from a hot gas to a cold gas and then from a cold gas to a hot gas. The gas is very hot when it changes from a hot gas to a cold gas and very cold when it changes from a cold gas to a hot gas. When the gas changes from a hot gas to a cold gas it becomes yellow. When the gas changes from a cold gas to a hot gas it

.. code:: ipython3

    print(f"Generation time: {end_time - start_time:.2f}s")


.. parsed-literal::

    Generation time: 15.62s


Alternative approach, Dynamic Speculative Decoding, described in the
`paper <https://arxiv.org/abs/2405.04304>`__ is based on heuristics and
adjusts the number of candidate tokens for the next iteration based on
the acceptance rate of the current iteration. If all speculative tokens
are correct, the number of candidate tokens increases; otherwise, it
decreases. For adjusting number of tokens
``assistant_confidence_threshold`` parameters should be used. If the
assistant model’s confidence in its prediction for the current token is
lower than this threshold, the assistant model stops the current token
generation iteration, even if the number of ``num_assistant_tokens`` is
not yet reached. You can find more details in this `blog
post <https://huggingface.co/blog/dynamic_speculation_lookahead>`__.
This approach has advantages for cases, when optimal number of tokens
for draft model is unknown and draft model has low acceptance rate.

   *Note*: For small and fast draft models like FastDraft, you may not
   see benefit for dynamic speculative decoding.

.. code:: ipython3

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 100
    config.assistant_confidence_threshold = 0.05
    start_time = time.perf_counter()
    result = pipe.generate(["Sun is yellow because"], config, streamer)
    end_time = time.perf_counter()


.. parsed-literal::

     it is made of gas. The gas is heated to a high temperature and then cooled. The gas changes from a hot gas to a cold gas and then from a cold gas to a hot gas. The gas is very hot when it changes from a hot gas to a cold gas and very cold when it changes from a cold gas to a hot gas. The gas is very light and can float in the air. When the gas cools it becomes a liquid. The Sun is a huge sphere of

.. code:: ipython3

    print(f"Generation time: {end_time - start_time:.2f}s")


.. parsed-literal::

    Generation time: 17.97s

