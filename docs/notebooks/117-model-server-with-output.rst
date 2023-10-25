Hello Model Server
==================

Introduction to OpenVINO™ Model Server (OVMS).

What is Model Serving?
###############################################################################################################################

A model server hosts models and makes them accessible to software
components over standard network protocols. A client sends a request to
the model server, which performs inference and sends a response back to
the client. Model serving offers many advantages for efficient model
deployment:

-  Remote inference enables using lightweight clients with only the
   necessary functions to perform API calls to edge or cloud
   deployments.
-  Applications are independent of the model framework, hardware device,
   and infrastructure.
-  Client applications in any programming language that supports REST or
   gRPC calls can be used to run inference remotely on the model server.
-  Clients require fewer updates since client libraries change very
   rarely.
-  Model topology and weights are not exposed directly to client
   applications, making it easier to control access to the model.
-  Ideal architecture for microservices-based applications and
   deployments in cloud environments – including Kubernetes and
   OpenShift clusters.
-  Efficient resource utilization with horizontal and vertical inference
   scaling.

|ovms_diagram| 

**Table of contents:**

- `Serving with OpenVINO Model Server <#serving-with-openvino-model-server>`__ 
- `Step 1: Prepare Docker <#step-1-prepare-docker>`__ 
- `Step 2: Preparing a Model Repository <#step-2-preparing-a-model-repository>`__ 
- `Step 3: Start the Model Server Container <#step-3-start-the-model-server-container>`__ 
- `Step 4: Prepare the Example Client Components <#step-4-prepare-the-example-client-components>`__ 

  - `Prerequisites <#prerequisites>`__ 
  - `Imports <#imports>`__ 
  - `Request Model Status <#request-model-status>`__ 
  - `Request Model Metadata <#request-model-metadata>`__ 
  - `Load input image <#load-input-image>`__ 
  - `Request Prediction on a Numpy Array <#request-prediction-on-a-numpy-array>`__ 
  - `Visualization <#visualization>`__ 

- `References <#references>`__

.. |ovms_diagram| image:: https://user-images.githubusercontent.com/91237924/215658773-4720df00-3b95-4a84-85a2-40f06138e914.png

Serving with OpenVINO Model Server
###############################################################################################################################

OpenVINO Model Server (OVMS) is a high-performance system for serving
models. Implemented in C++ for scalability and optimized for deployment
on Intel architectures, the model server uses the same architecture and
API as TensorFlow Serving and KServe while applying OpenVINO for
inference execution. Inference service is provided via gRPC or REST API,
making deploying new algorithms and AI experiments easy.

.. figure:: https://user-images.githubusercontent.com/91237924/215658767-0e0fc221-aed0-4db1-9a82-6be55f244dba.png
   :alt: ovms_high_level

   ovms_high_level

To quickly start using OpenVINO™ Model Server, follow these steps:

Step 1: Prepare Docker
###############################################################################################################################

Install `Docker Engine <https://docs.docker.com/engine/install/>`__,
including its
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
    


Step 2: Preparing a Model Repository
###############################################################################################################################

The models need to be placed and mounted in a particular directory
structure and according to the following rules:

::

   tree models/
   models/
   ├── model1
   │   ├── 1
   │   │   ├── ir_model.bin
   │   │   └── ir_model.xml
   │   └── 2
   │       ├── ir_model.bin
   │       └── ir_model.xml
   ├── model2
   │   └── 1
   │       ├── ir_model.bin
   │       ├── ir_model.xml
   │       └── mapping_config.json
   ├── model3
   │    └── 1
   │        └── model.onnx
   ├── model4
   │      └── 1
   │        ├── model.pdiparams
   │        └── model.pdmodel
   └── model5
          └── 1
            └── TF_fronzen_model.pb

-  Each model should be stored in a dedicated directory, for example,
   model1 and model2.

-  Each model directory should include a sub-folder for each of its
   versions (1,2, etc). The versions and their folder names should be
   positive integer values.

-  Note that in execution, the versions are enabled according to a
   pre-defined version policy. If the client does not specify the
   version number in parameters, by default, the latest version is
   served.

-  Every version folder must include model files, that is, ``.bin`` and
   ``.xml`` for OpenVINO IR, ``.onnx`` for ONNX, ``.pdiparams`` and
   ``.pdmodel`` for Paddle Paddle, and ``.pb`` for TensorFlow. The file
   name can be arbitrary.

.. code:: ipython3

    import os
    import sys
    
    sys.path.append("../utils")
    from notebook_utils import download_file
    
    dedicated_dir = "models"
    model_name = "detection"
    model_version = "1"
    
    MODEL_DIR = f"{dedicated_dir}/{model_name}/{model_version}"
    XML_PATH = "horizontal-text-detection-0001.xml"
    BIN_PATH = "horizontal-text-detection-0001.bin"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.xml"
    model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.bin"
    
    download_file(model_xml_url, XML_PATH, MODEL_DIR)
    download_file(model_bin_url, BIN_PATH_name, MODEL_DIR)
    
    model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.xml"
    model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/horizontal-text-detection-0001/FP32/horizontal-text-detection-0001.bin"
    
    download_file(model_xml_url, model_xml_name, base_model_dir)
    download_file(model_bin_url, model_bin_name, base_model_dir)


.. parsed-literal::

    Model Copied to "./models/detection/1".


Step 3: Start the Model Server Container
###############################################################################################################################

Pull and start the container:

.. code:: ipython3

    !docker run -d --rm --name="ovms" -v $(pwd)/models:/models -p 9000:9000 openvino/model_server:latest --model_path /models/detection/ --model_name detection --port 9000


.. parsed-literal::

    7bf50596c18d5ad93d131eb9e435439dfb3cedf994518c5e89cc7727f5d3530e


Check whether the OVMS container is running normally:

.. code:: ipython3

    !docker ps | grep ovms


.. parsed-literal::

    7bf50596c18d   openvino/model_server:latest   "/ovms/bin/ovms --mo…"   Less than a second ago   Up Less than a second   0.0.0.0:9000->9000/tcp, :::9000->9000/tcp   ovms


The required Model Server parameters are listed below. For additional
configuration options, see the `Model Server Parameters
section <https://docs.openvino.ai/2023.0/ovms_docs_parameters.html#doxid-ovms-docs-parameters>`__.

.. raw:: html

   <table class="table">

.. raw:: html

   <colgroup>

.. raw:: html

   <col style="width: 20%" />

.. raw:: html

   <col style="width: 80%" />

.. raw:: html

   </colgroup>

.. raw:: html

   <tbody>

.. raw:: html

   <tr class="row-odd">

.. raw:: html

   <td>

.. raw:: html

   <p>

–rm

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      remove the container when exiting the Docker container

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="row-even">

.. raw:: html

   <td>

.. raw:: html

   <p>

-d

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      runs the container in the background

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="row-odd">

.. raw:: html

   <td>

.. raw:: html

   <p>

-v

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      defines how to mount the model folder in the Docker container

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="row-even">

.. raw:: html

   <td>

.. raw:: html

   <p>

-p

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      exposes the model serving port outside the Docker container

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="row-odd">

.. raw:: html

   <td>

.. raw:: html

   <p>

openvino/model_server:latest

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      represents the image name; the OVMS binary is the Docker entry
      point

   .. container:: line

      varies by tag and build process - see tags:
      https://hub.docker.com/r/openvino/model_server/tags/ for a full
      tag list.

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="row-even">

.. raw:: html

   <td>

.. raw:: html

   <p>

–model_path

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      model location, which can be:

   .. container:: line

      a Docker container path that is mounted during start-up

   .. container:: line

      a Google Cloud Storage path gs://<bucket>/<model_path>

   .. container:: line

      an AWS S3 path s3://<bucket>/<model_path>

   .. container:: line

      an Azure blob path az://<container>/<model_path>

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="row-odd">

.. raw:: html

   <td>

.. raw:: html

   <p>

–model_name

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      the name of the model in the model_path

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="row-even">

.. raw:: html

   <td>

.. raw:: html

   <p>

–port

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      the gRPC server port

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr class="row-odd">

.. raw:: html

   <td>

.. raw:: html

   <p>

–rest_port

.. raw:: html

   </p>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. container:: line-block

   .. container:: line

      the REST server port

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </tbody>

.. raw:: html

   </table>

If the serving port ``9000`` is already in use, please switch it to
another available port on your system. For example:\ ``-p 9020:9000``

Step 4: Prepare the Example Client Components
###############################################################################################################################

OpenVINO Model Server exposes two sets of APIs: one compatible with
``TensorFlow Serving`` and another one, with ``KServe API``, for
inference. Both APIs work on ``gRPC`` and ``REST``\ interfaces.
Supporting two sets of APIs makes OpenVINO Model Server easier to plug
into existing systems the already leverage one of these APIs for
inference. This example will demonstrate how to write a TensorFlow
Serving API client for object detection.

Prerequisites
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Install necessary packages.

.. code:: ipython3

    !pip install -q ovmsclient


.. parsed-literal::

    Collecting ovmsclient
      Downloading ovmsclient-2022.3-py3-none-any.whl (163 kB)
    [2K     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 164.0/164.0 KB 2.1 MB/s eta 0:00:00a 0:00:01
    Requirement already satisfied: numpy>=1.16.6 in /home/adrian/repos/openvino_notebooks_adrian/venv/lib/python3.9/site-packages (from ovmsclient) (1.23.4)
    Requirement already satisfied: requests>=2.27.1 in /home/adrian/repos/openvino_notebooks_adrian/venv/lib/python3.9/site-packages (from ovmsclient) (2.27.1)
    Collecting grpcio>=1.47.0
      Downloading grpcio-1.51.3-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.8 MB)
    [2K     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.8/4.8 MB 5.6 MB/s eta 0:00:0000:0100:01
    Requirement already satisfied: protobuf>=3.19.4 in /home/adrian/repos/openvino_notebooks_adrian/venv/lib/python3.9/site-packages (from ovmsclient) (3.19.6)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/adrian/repos/openvino_notebooks_adrian/venv/lib/python3.9/site-packages (from requests>=2.27.1->ovmsclient) (1.26.9)
    Requirement already satisfied: idna<4,>=2.5 in /home/adrian/repos/openvino_notebooks_adrian/venv/lib/python3.9/site-packages (from requests>=2.27.1->ovmsclient) (3.3)
    Requirement already satisfied: certifi>=2017.4.17 in /home/adrian/repos/openvino_notebooks_adrian/venv/lib/python3.9/site-packages (from requests>=2.27.1->ovmsclient) (2021.10.8)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /home/adrian/repos/openvino_notebooks_adrian/venv/lib/python3.9/site-packages (from requests>=2.27.1->ovmsclient) (2.0.12)
    Installing collected packages: grpcio, ovmsclient
      Attempting uninstall: grpcio
        Found existing installation: grpcio 1.34.1
        Uninstalling grpcio-1.34.1:
          Successfully uninstalled grpcio-1.34.1
    Successfully installed grpcio-1.51.3 ovmsclient-2022.3
    WARNING: You are using pip version 22.0.4; however, version 23.0.1 is available.
    You should consider upgrading via the '/home/adrian/repos/openvino_notebooks_adrian/venv/bin/python -m pip install --upgrade pip' command.
    

Imports
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from ovmsclient import make_grpc_client

Request Model Status
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    address = "localhost:9000"
    
    # Bind the grpc address to the client object
    client = make_grpc_client(address)
    model_status = client.get_model_status(model_name=model_name)
    print(model_status)


.. parsed-literal::

    {1: {'state': 'AVAILABLE', 'error_code': 0, 'error_message': 'OK'}}


Request Model Metadata
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    model_metadata = client.get_model_metadata(model_name=model_name)
    print(model_metadata)


.. parsed-literal::

    {'model_version': 1, 'inputs': {'image': {'shape': [1, 3, 704, 704], 'dtype': 'DT_FLOAT'}}, 'outputs': {'1469_1470.0': {'shape': [-1], 'dtype': 'DT_FLOAT'}, '1078_1079.0': {'shape': [1000], 'dtype': 'DT_FLOAT'}, '1330_1331.0': {'shape': [36], 'dtype': 'DT_FLOAT'}, 'labels': {'shape': [-1], 'dtype': 'DT_INT32'}, '1267_1268.0': {'shape': [121], 'dtype': 'DT_FLOAT'}, '1141_1142.0': {'shape': [1000], 'dtype': 'DT_FLOAT'}, '1204_1205.0': {'shape': [484], 'dtype': 'DT_FLOAT'}, 'boxes': {'shape': [-1, 5], 'dtype': 'DT_FLOAT'}}}


Load input image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    # Text detection models expect an image in BGR format.
    image = cv2.imread("../data/image/intel_rnb.jpg")
    fp_image = image.astype("float32")
    
    # Resize the image to meet network expected input sizes.
    input_shape = model_metadata['inputs']['image']['shape']
    height, width = input_shape[2], input_shape[3]
    resized_image = cv2.resize(fp_image, (height, width))
    
    # Reshape to the network input shape.
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fee22d6ecd0>




.. image:: 117-model-server-with-output_files/117-model-server-with-output_20_1.png


Request Prediction on a Numpy Array
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    inputs = {"image": input_image}
    
    # Run inference on model server and receive the result data
    boxes = client.predict(inputs=inputs, model_name=model_name)['boxes']
    
    # Remove zero only boxes.
    boxes = boxes[~np.all(boxes == 0, axis=1)]
    print(boxes)


.. parsed-literal::

    [[3.9992419e+02 8.1032524e+01 5.6187299e+02 1.3619952e+02 5.3706491e-01]
     [2.6189725e+02 6.8310547e+01 3.8541251e+02 1.2095630e+02 4.7559953e-01]
     [6.1644586e+02 2.8008759e+02 6.6627545e+02 3.1178854e+02 4.4982004e-01]
     [2.0762042e+02 6.2798470e+01 2.3444728e+02 1.0706525e+02 3.7216505e-01]
     [5.1742780e+02 5.5603595e+02 5.4927539e+02 5.8736023e+02 3.2588077e-01]
     [2.2261986e+01 4.5406548e+01 1.8868817e+02 1.0225631e+02 3.0407205e-01]]


Visualization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    # For each detection, the description is in the [x_min, y_min, x_max, y_max, conf] format:
    # The image passed here is in BGR format with changed width and height. To display it in colors expected by matplotlib, use cvtColor function
    def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3, conf_labels=True):
        # Define colors for boxes and descriptions.
        colors = {"red": (255, 0, 0), "green": (0, 255, 0)}
    
        # Fetch the image shapes to calculate a ratio.
        (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
    
        # Convert the base image from BGR to RGB format.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
        # Iterate through non-zero boxes.
        for box in boxes:
            # Pick a confidence factor from the last place in an array.
            conf = box[-1]
            if conf > threshold:
                # Convert float to int and multiply corner position of each box by x and y ratio.
                # If the bounding box is found at the top of the image, 
                # position the upper box bar little lower to make it visible on the image. 
                (x_min, y_min, x_max, y_max) = [
                    int(max(corner_position * ratio_y, 10)) if idx % 2 
                    else int(corner_position * ratio_x)
                    for idx, corner_position in enumerate(box[:-1])
                ]
    
                # Draw a box based on the position, parameters in rectangle function are: image, start_point, end_point, color, thickness.
                rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)
    
                # Add text to the image based on position and confidence.
                # Parameters in text function are: image, text, bottom-left_corner_textfield, font, font_scale, color, thickness, line_type.
                if conf_labels:
                    rgb_image = cv2.putText(
                        rgb_image,
                        f"{conf:.2f}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        colors["red"],
                        1,
                        cv2.LINE_AA,
                    )
    
        return rgb_image

.. code:: ipython3

    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(convert_result_to_image(image, resized_image, boxes, conf_labels=False))




.. parsed-literal::

    <matplotlib.image.AxesImage at 0x7fee219e4df0>




.. image:: 117-model-server-with-output_files/117-model-server-with-output_25_1.png


To stop and remove the model server container, you can use the following
command:

.. code:: ipython3

    !docker stop ovms


.. parsed-literal::

    ovms


References
###############################################################################################################################

1. `OpenVINO™ Model Server
   documentation <https://docs.openvino.ai/2023.0/ovms_what_is_openvino_model_server.html>`__
2. `OpenVINO™ Model Server GitHub
   repository <https://github.com/openvinotoolkit/model_server/>`__
