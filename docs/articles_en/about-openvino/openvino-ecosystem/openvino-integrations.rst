OpenVINO™ Integrations
==============================


.. meta::
   :description: Check a list of integrations between OpenVINO and other Deep Learning solutions.


OpenVINO has been adopted by multiple AI projects in various areas. For an extensive list of
community-based projects involving OpenVINO, see the
`Awesome OpenVINO repository <https://github.com/openvinotoolkit/awesome-openvino>`__.


.. = 1 ========================================================================================


**Hugging Face Optimum-Intel**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Grab and use models leveraging OpenVINO within the Hugging Face API.
        The repository hosts pre-optimized OpenVINO IR models, so that you can use
        them in your projects without the need for any adjustments.
      | Benefits:
      | - Minimize complex coding for Generative AI.

   .. grid-item::

      * :doc:`Run inference with HuggingFace and Optimum Intel <../../openvino-workflow-generative/inference-with-optimum-intel>`
      * `A notebook example: llm-chatbot <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/llm-chatbot>`__
      * `Hugging Face Inference documentation <https://huggingface.co/docs/optimum/main/intel/openvino/inference>`__
      * `Hugging Face Compression documentation <https://huggingface.co/docs/optimum/main/intel/openvino/optimization>`__
      * `Hugging Face Reference Documentation <https://huggingface.co/docs/optimum/main/intel/openvino/reference>`__

.. dropdown:: Check example code
   :animate: fade-in-slide-down
   :color: secondary

   .. code-block:: py

      -from transformers import AutoModelForCausalLM
      +from optimum.intel.openvino import OVModelForCausalLM

      from transformers import AutoTokenizer, pipeline
      model_id = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"

      -model = AutoModelForCausalLM.from_pretrained(model_id)
      +model = OVModelForCausalLM.from_pretrained(model_id, export=True)


.. = 2 ========================================================================================

**OpenVINO Execution Provider for ONNX Runtime**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Utilize OpenVINO as a backend with your existing ONNX Runtime code.
      | Benefits:
      | - Enhanced inference performance on Intel hardware with minimal code modifications.

   .. grid-item::

      * A notebook example: YOLOv8 object detection
      * `ONNX User documentation <https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html>`__
      * `Build ONNX RT with OV EP <https://oliviajain.github.io/onnxruntime/docs/build/eps.html#openvino>`__
      * `ONNX Examples <https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html#openvino-execution-provider-samples-tutorials>`__


.. dropdown:: Check example code
   :animate: fade-in-slide-down
   :color: secondary

   .. code-block:: cpp

      device = `CPU_FP32`
      # Set OpenVINO as the Execution provider to infer this model
      sess.set_providers([`OpenVINOExecutionProvider`], [{device_type` : device}])


.. = 3 ========================================================================================

**Torch.compile with OpenVINO**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Use OpenVINO for Python-native applications by JIT-compiling code into optimized kernels.
      | Benefits:
      | - Enhanced inference performance on Intel hardware with minimal code modifications.

   .. grid-item::

      * :doc:`PyTorch Deployment via torch.compile <../../openvino-workflow/torch-compile>`
      * `torch.compiler documentation <https://pytorch.org/docs/stable/torch.compiler.html>`__
      * `torch.compiler API reference <https://pytorch.org/docs/stable/torch.compiler_api.html>`__

.. dropdown:: Check example code
   :animate: fade-in-slide-down
   :color: secondary

   .. code-block:: python

      import openvino.torch

      ...
      model = torch.compile(model, backend='openvino')
      ...



.. = 4 ========================================================================================

**OpenVINO LLMs with LlamaIndex**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Build context-augmented GenAI applications with the LlamaIndex framework and enhance
        runtime performance with OpenVINO.
      | Benefits:
      | - Minimize complex coding for Generative AI.

   .. grid-item::

      * :doc:`LLM inference with Optimum-intel <../../openvino-workflow-generative/inference-with-optimum-intel>`
      * `A notebook example: llm-agent-rag <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-agent-react/llm-agent-rag-llamaindex.ipynb>`__
      * `Inference documentation <https://docs.llamaindex.ai/en/stable/examples/llm/openvino/>`__
      * `Rerank documentation <https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/openvino_rerank/>`__
      * `Embeddings documentation <https://docs.llamaindex.ai/en/stable/examples/embeddings/openvino/>`__
      * `API Reference <https://docs.llamaindex.ai/en/stable/api_reference/llms/openvino/>`__

.. dropdown:: Check example code
   :animate: fade-in-slide-down
   :color: secondary

   .. code-block:: python

      ov_config = {
          "PERFORMANCE_HINT": "LATENCY",
          "NUM_STREAMS": "1",
          "CACHE_DIR": "",
      }

      ov_llm = OpenVINOLLM(
          model_id_or_path="HuggingFaceH4/zephyr-7b-beta",
          context_window=3900,
          max_new_tokens=256,
          model_kwargs={"ov_config": ov_config},
          generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
          messages_to_prompt=messages_to_prompt,
          completion_to_prompt=completion_to_prompt,
          device_map="cpu",
      )

.. = 5 ========================================================================================

**OpenVINO Backend for ExecuTorch**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Export and run AI models using OpenVINO with ExecuTorch to optimize performance on 
        Intel hardware. 
      | Benefits:
      | - Accelerate inference, reduce latency, and simplify deployment for efficient AI applications.

   .. grid-item::

      * `OpenVINO Backend for ExecuTorch <https://github.com/pytorch/executorch/blob/main/backends/openvino/README.md>`__
      * `OpenVINO Backend Examples <https://github.com/pytorch/executorch/blob/main/examples/openvino/README.md>`__
      * `Building and Running ExecuTorch with OpenVINO Backend <https://github.com/pytorch/executorch/blob/main/docs/source/build-run-openvino.md>`__

.. dropdown:: Check example code
   :animate: fade-in-slide-down
   :color: secondary

   .. code-block:: python

      python aot_optimize_and_infer.py --export --suite timm --model vgg16 --input_shape "[1, 3, 224, 224]" --device CPU

.. = 6 ========================================================================================

**OpenVINO Integration for LangChain**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Integrate OpenVINO with the LangChain framework to enhance runtime performance for GenAI applications. 
      | Benefits:
      | - Streamline the integration and chaining of language models for efficient AI workflows.

   .. grid-item::

      * `LangChain Integration Guide <https://python.langchain.com/docs/integrations/llms/openvino/>`__

.. dropdown:: Check example code
   :animate: fade-in-slide-down
   :color: secondary

   .. code-block:: python

      ov_llm = HuggingFacePipeline.from_model_id(
         model_id="ov_model_dir",
         task="text-generation",
         backend="openvino",
         model_kwargs={"device": "CPU", "ov_config": ov_config},
         pipeline_kwargs={"max_new_tokens": 10},
      )

      chain = prompt | ov_llm

      question = "What is electroencephalography?"

      print(chain.invoke({"question": question}))

.. = 7 ========================================================================================

**Intel® Geti™**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Build computer vision models faster with less data using Intel® Geti™. It streamlines labeling, training, and deployment, exporting models optimized for OpenVINO.
      | Benefits:
      | - Train with less data and deploy models faster.
   .. grid-item::

      * `Intel® Geti Overview <https://docs.geti.intel.com/>`__
      * `Documentation <https://docs.geti.intel.com/docs/user-guide/getting-started/introduction>`__
      * `Tutorials <https://docs.geti.intel.com/docs/user-guide/getting-started/use-geti/tutorials>`__
.. = 8 ========================================================================================

**AI Playground™**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Use Intel® OpenVINO™ in AI Playground to optimize and run AI models efficiently on Intel 
        CPUs and Arc GPUs, enabling local image generation, editing, and video processing. It 
       supports OpenVINO-optimized models like TinyLlama, Mistral 7B, and Phi-3 mini, no conversion 
       needed.
      | Benefits:
      | - Easily set up pre-optimized models.
      | - Run faster, hardware-accelerated inference with OpenVINO.
   .. grid-item::

      * `AI Playground Overview <https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/ai-playground.html>`__
      * `GitHub Repository <https://github.com/intel/AI-Playground>`__
      * `User Guide <https://github.com/intel/ai-playground/blob/main/AI%20Playground%20Users%20Guide.pdf>`__
.. = 9 ========================================================================================

**Intel® AI Assistant Builder**

|hr|

.. grid:: 1 1 2 2
   :gutter: 4

   .. grid-item::

      | Run local AI assistants with Intel® AI Assistant Builder using OpenVINO-optimized models 
        like Phi-3 and Qwen2.5. Build secure, efficient assistants customized for your data 
        and workflows.
      | Benefits:
      | - Build custom assistants with agentic workflows and knowledge bases.
      | - Keep data secure by running fully local.
   .. grid-item::

      * `GitHub Repository <https://github.com/intel/intel-ai-assistant-builder>`__



.. ============================================================================================

.. |hr| raw:: html

   <hr style="margin-top:-12px!important;border-top:1px solid #383838;">