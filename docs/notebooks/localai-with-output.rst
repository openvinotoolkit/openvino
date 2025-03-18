LocalAI and OpenVINO
====================

`LocalAI <https://localai.io/>`__ is the free, Open Source OpenAI
alternative. LocalAI act as a drop-in replacement REST API that’s
compatible with OpenAI API specifications for local inferencing. It
allows you to run LLMs, generate images, audio (and not only) locally or
on-prem with consumer grade hardware, supporting multiple model families
and architectures. Does not require GPU. It is created and maintained by
``Ettore Di Giacinto``.

In this tutorial we show how to prepare a model config and launch an
OpenVINO LLM model with LocalAI in docker container.


**Table of contents:**


-  `Prepare Docker <#prepare-docker>`__
-  `Prepare a model <#prepare-a-model>`__
-  `Run the server <#run-the-server>`__
-  `Send a client request <#send-a-client-request>`__
-  `Stop the server <#stop-the-server>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    import requests
    from pathlib import Path
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)
    
    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry
    
    collect_telemetry("localai.ipynb")

Prepare Docker
--------------

Install `Docker
Engine <https://docs.docker.com/engine/install/>`__, including its
`post-installation <https://docs.docker.com/engine/install/linux-postinstall/>`__
steps, on your development system. To verify installation, test it,
using the following command. When it is ready, it will display a test
image and a message.

.. code:: ipython3

    !docker run hello-world


.. parsed-literal::

    
    Hello from Docker!
    This message shows that your installation appears to be working correctly.
    
    To generate this message, Docker took the following steps:
     1. The Docker client contacted the Docker daemon.
     2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
        (amd64)
     3. The Docker daemon created a new container from that image which runs the
        executable that produces the output you are currently reading.
     4. The Docker daemon streamed that output to the Docker client, which sent it
        to your terminal.
    
    To try something more ambitious, you can run an Ubuntu container with:
     $ docker run -it ubuntu bash
    
    Share images, automate workflows, and more with a free Docker ID:
     https://hub.docker.com/
    
    For more examples and ideas, visit:
     https://docs.docker.com/get-started/
    


Prepare a model
~~~~~~~~~~~~~~~



LocalAI allows to use customized models. For more details you can read
the
`instruction <https://localai.io/docs/getting-started/customize-model/>`__
where you can also find the detailed documentation. We will use one of
the OpenVINO optimized LLMs in the collection on the `collection on
🤗Hugging
Face <https://huggingface.co/collections/OpenVINO/llm-6687aaa2abca3bbcec71a9bd>`__.
In this example we will use
`TinyLlama-1.1B-Chat-v1.0-fp16-ov <https://huggingface.co/OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov>`__.
First of all we should create a model configuration file:

.. code:: yaml

   name: TinyLlama-1.1B-Chat-v1.0-fp16-ov
   backend: transformers
   parameters:
     model: OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov
     temperature: 0.2
     top_k: 40
     top_p: 0.95
     max_new_tokens: 32
     
   type: OVModelForCausalLM

   template:
     chat_message: |
       <|im_start|>{{if eq .RoleName "assistant"}}assistant{{else if eq .RoleName "system"}}system{{else if eq .RoleName "user"}}user{{end}}
       {{if .Content}}{{.Content}}{{end}}<|im_end|>
     chat: |
       {{.Input}}
       <|im_start|>assistant
       
     completion: |
       {{.Input}}

   stopwords:
   - <|im_end|>

The fields ``backend``, ``model``, ``type`` you can find in the code
example on the model page (we added the corresponding comments):

.. code:: python

   from transformers import AutoTokenizer   # backend
   from optimum.intel.openvino import OVModelForCausalLM  # type

   model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-fp16-ov"  # parameters.model
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = OVModelForCausalLM.from_pretrained(model_id)

The name you can choose by yourself. By this name you will specify what
model to use on the client side.

You can create a GitHub gist and modify fields:
`ov.yaml <https://gist.githubusercontent.com/aleksandr-mokrov/f007c8fa6036760a856ddc60f605a0b0/raw/9d24ceeb487f9c058a943113bd0290e8ae565b3e/ov.yaml>`__

Description of the parameters used in config YAML file can be found
`here <https://localai.io/advanced/#advanced-configuration-with-yaml-files>`__.

The most important:

-  ``name`` - model name, used to identify the model in API calls.
-  ``backend`` - backend to use for computation (like llama-cpp,
   diffusers, whisper, transformers).
-  ``parameters.model`` - relative to the models path.
-  ``temperature``, ``top_k``, ``top_p``, ``max_new_tokens`` -
   parameters for the model.
-  ``type`` - type of configuration, often related to the type of task
   or model architecture.
-  ``template`` - templates for various types of model interactions.
-  ``stopwords`` - Words or phrases that halts processing.

Run the server
~~~~~~~~~~~~~~



Everything is ready for launch. Use
``quay.io/go-skynet/local-ai:v2.23.0-ffmpeg`` image that contains all
required dependencies. For more details read `Run with container
images <https://localai.io/basics/container/#standard-container-images>`__.
If you want to see the output remove the ``-d`` flag and send a client
request from a separate notebook.

.. code:: ipython3

    !docker run -d --rm --name="localai" -p 8080:8080 quay.io/go-skynet/local-ai:master-sycl-f16-ffmpeg https://gist.githubusercontent.com/aleksandr-mokrov/f007c8fa6036760a856ddc60f605a0b0/raw/9d24ceeb487f9c058a943113bd0290e8ae565b3e/ov.yaml


.. parsed-literal::

    471a3e7e745080f9d62b3b4791103efedec7dabb43f2d6c7ca57b1b6023aa823
    docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #1: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
    nvidia-container-cli: requirement error: invalid expression: unknown.


Check whether the ``localai`` container is running normally:

.. code:: ipython3

    !docker ps | grep localai

Send a client request
~~~~~~~~~~~~~~~~~~~~~



Now you can send HTTP requests using the model name
``TinyLlama-1.1B-Chat-v1.0-fp16-ov``. More details how to use `OpenAI
API <https://platform.openai.com/docs/api-reference/chat>`__.

.. code:: ipython3

    !curl http://localhost:8080/v1/completions -H "Content-Type: application/json" -d '{"model": "TinyLlama-1.1B-Chat-v1.0-fp16-ov", "prompt": "What is OpenVINO?"}'


.. parsed-literal::

    curl: (7) Failed to connect to localhost port 8080: Connection refused


Stop the server
~~~~~~~~~~~~~~~



.. code:: ipython3

    !docker stop localai


.. parsed-literal::

    Error response from daemon: No such container: localai

