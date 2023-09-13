Convert and Optimize YOLOv8 with OpenVINO™
==========================================

The YOLOv8 algorithm developed by Ultralytics is a cutting-edge,
state-of-the-art (SOTA) model that is designed to be fast, accurate, and
easy to use, making it an excellent choice for a wide range of object
detection, image segmentation, and image classification tasks. More
details about its realization can be found in the original model
`repository <https://github.com/ultralytics/ultralytics>`__.

This tutorial demonstrates step-by-step instructions on how to run apply
quantization with accuracy control to PyTorch YOLOv8. The advanced
quantization flow allows to apply 8-bit quantization to the model with
control of accuracy metric. This is achieved by keeping the most
impactful operations within the model in the original precision. The
flow is based on the `Basic 8-bit
quantization <https://docs.openvino.ai/2023.0/basic_quantization_flow.html>`__
and has the following differences:

-  Besides the calibration dataset, a validation dataset is required to
   compute the accuracy metric. Both datasets can refer to the same data
   in the simplest case.
-  Validation function, used to compute accuracy metric is required. It
   can be a function that is already available in the source framework
   or a custom function.
-  Since accuracy validation is run several times during the
   quantization process, quantization with accuracy control can take
   more time than the Basic 8-bit quantization flow.
-  The resulted model can provide smaller performance improvement than
   the Basic 8-bit quantization flow because some of the operations are
   kept in the original precision.

..note::

   Currently, 8-bit quantization with accuracy control in NNCF
   is available only for models in OpenVINO representation.

The steps for the quantization with accuracy control are described
below.

The tutorial consists of the following steps:

- `Prerequisites <#prerequisites>`__
- `Get Pytorch model and OpenVINO IR model <#get-pytorch-model-and-openvino-ir-model>`__
- `Define validator and data loader <#define-validator-and-data-loader>`__
- `Prepare calibration and validation datasets <#prepare-calibration-and-validation-datasets>`__
- `Prepare validation function <#prepare-validation-function>`__
- `Run quantization with accuracy control <#run-quantization-with-accuracy-control>`__
- `Compare Accuracy and Performance of the Original and Quantized Models <#compare-accuracy-and-performance-of-the-original-and-quantized-models>`__

Prerequisites 
###############################################################################################################################

Install necessary packages.

.. code:: ipython3

    !pip install -q "openvino==2023.1.0.dev20230811"
    !pip install git+https://github.com/openvinotoolkit/nncf.git@develop
    !pip install -q "ultralytics==8.0.43"


.. parsed-literal::

    Collecting git+https://github.com/openvinotoolkit/nncf.git@develop
      Cloning https://github.com/openvinotoolkit/nncf.git (to revision develop) to /tmp/pip-req-build-q26q169c
      Running command git clone --filter=blob:none --quiet https://github.com/openvinotoolkit/nncf.git /tmp/pip-req-build-q26q169c
      Filtering content:   1% (2/142)
      Filtering content:   2% (3/142)
      Filtering content:   3% (5/142)
      Filtering content:   4% (6/142)
      Filtering content:   5% (8/142)
      Filtering content:   6% (9/142), 11.23 MiB | 16.49 MiB/s
      Filtering content:   7% (10/142), 11.23 MiB | 16.49 MiB/s
      Filtering content:   7% (10/142), 12.61 MiB | 10.32 MiB/s
      Filtering content:   8% (12/142), 12.61 MiB | 10.32 MiB/s
      Filtering content:   9% (13/142), 13.81 MiB | 7.30 MiB/s
      Filtering content:  10% (15/142), 13.81 MiB | 7.30 MiB/s
      Filtering content:  11% (16/142), 13.81 MiB | 7.30 MiB/s
      Filtering content:  11% (17/142), 13.81 MiB | 7.30 MiB/s
      Filtering content:  12% (18/142), 13.81 MiB | 7.30 MiB/s
      Filtering content:  13% (19/142), 13.81 MiB | 7.30 MiB/s
      Filtering content:  14% (20/142), 13.81 MiB | 7.30 MiB/s
      Filtering content:  15% (22/142), 18.00 MiB | 7.01 MiB/s
      Filtering content:  16% (23/142), 18.00 MiB | 7.01 MiB/s
      Filtering content:  17% (25/142), 18.00 MiB | 7.01 MiB/s
      Filtering content:  17% (25/142), 20.21 MiB | 6.50 MiB/s
      Filtering content:  18% (26/142), 20.21 MiB | 6.50 MiB/s
      Filtering content:  19% (27/142), 20.21 MiB | 6.50 MiB/s
      Filtering content:  20% (29/142), 20.21 MiB | 6.50 MiB/s
      Filtering content:  21% (30/142), 20.21 MiB | 6.50 MiB/s
      Filtering content:  22% (32/142), 20.21 MiB | 6.50 MiB/s
      Filtering content:  23% (33/142), 23.21 MiB | 6.41 MiB/s
      Filtering content:  24% (35/142), 23.21 MiB | 6.41 MiB/s
      Filtering content:  25% (36/142), 23.21 MiB | 6.41 MiB/s
      Filtering content:  26% (37/142), 23.21 MiB | 6.41 MiB/s
      Filtering content:  26% (38/142), 23.21 MiB | 6.41 MiB/s
      Filtering content:  27% (39/142), 25.49 MiB | 6.14 MiB/s
      Filtering content:  28% (40/142), 25.49 MiB | 6.14 MiB/s
      Filtering content:  29% (42/142), 25.49 MiB | 6.14 MiB/s
      Filtering content:  30% (43/142), 25.49 MiB | 6.14 MiB/s
      Filtering content:  31% (45/142), 25.49 MiB | 6.14 MiB/s
      Filtering content:  32% (46/142), 25.49 MiB | 6.14 MiB/s
      Filtering content:  33% (47/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  34% (49/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  35% (50/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  36% (52/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  37% (53/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  38% (54/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  38% (55/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  39% (56/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  40% (57/142), 27.56 MiB | 5.89 MiB/s
      Filtering content:  41% (59/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  42% (60/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  43% (62/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  44% (63/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  45% (64/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  46% (66/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  47% (67/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  48% (69/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  49% (70/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  50% (71/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  51% (73/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  52% (74/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  53% (76/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  54% (77/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  55% (79/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  56% (80/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  57% (81/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  58% (83/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  59% (84/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  60% (86/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  61% (87/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  62% (89/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  63% (90/142), 29.59 MiB | 5.66 MiB/s
      Filtering content:  64% (91/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  65% (93/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  66% (94/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  67% (96/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  68% (97/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  69% (98/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  70% (100/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  71% (101/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  72% (103/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  73% (104/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  74% (106/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  75% (107/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  76% (108/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  77% (110/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  78% (111/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  79% (113/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  80% (114/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  81% (116/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  82% (117/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  83% (118/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  84% (120/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  85% (121/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  86% (123/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  87% (124/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  88% (125/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  89% (127/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  90% (128/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  91% (130/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  92% (131/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  93% (133/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  94% (134/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  95% (135/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  96% (137/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  97% (138/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  98% (140/142), 31.76 MiB | 4.16 MiB/s
      Filtering content:  99% (141/142), 31.76 MiB | 4.16 MiB/s
      Filtering content: 100% (142/142), 31.76 MiB | 4.16 MiB/s
      Filtering content: 100% (142/142), 32.00 MiB | 3.58 MiB/s, done.
      Resolved https://github.com/openvinotoolkit/nncf.git to commit 90a1e860c93b553fa9684113e02d41d622235c55
      Preparing metadata (setup.py) ... - done
    Collecting pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd (from nncf==2.5.0.dev0+90a1e860)
      Using cached pymoo-0.6.0.1-py3-none-any.whl
    Requirement already satisfied: jsonschema>=3.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (4.19.0)
    Requirement already satisfied: jstyleson>=0.0.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (0.0.2)
    Requirement already satisfied: natsort>=7.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (8.4.0)
    Requirement already satisfied: networkx<=2.8.2,>=2.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (2.8.2)
    Requirement already satisfied: ninja<1.11,>=1.10.0.post2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (1.10.2.4)
    Requirement already satisfied: numpy<1.25,>=1.19.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.1.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (2023.1.1)
    Requirement already satisfied: packaging>=20.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (23.1)
    Requirement already satisfied: pandas<2.1,>=1.1.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (2.0.3)
    Requirement already satisfied: psutil in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (5.9.5)
    Requirement already satisfied: pydot>=1.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (1.4.2)
    Requirement already satisfied: pyparsing<3.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (2.4.7)
    Requirement already satisfied: scikit-learn>=0.24.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (1.3.0)
    Requirement already satisfied: scipy<1.11,>=1.3.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (1.10.1)
    Requirement already satisfied: texttable>=1.6.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (1.6.7)
    Requirement already satisfied: tqdm>=4.54.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from nncf==2.5.0.dev0+90a1e860) (4.66.1)
    Requirement already satisfied: attrs>=22.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jsonschema>=3.2.0->nncf==2.5.0.dev0+90a1e860) (23.1.0)
    Requirement already satisfied: importlib-resources>=1.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jsonschema>=3.2.0->nncf==2.5.0.dev0+90a1e860) (6.0.1)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jsonschema>=3.2.0->nncf==2.5.0.dev0+90a1e860) (2023.7.1)
    Requirement already satisfied: pkgutil-resolve-name>=1.3.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jsonschema>=3.2.0->nncf==2.5.0.dev0+90a1e860) (1.3.10)
    Requirement already satisfied: referencing>=0.28.4 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jsonschema>=3.2.0->nncf==2.5.0.dev0+90a1e860) (0.30.2)
    Requirement already satisfied: rpds-py>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jsonschema>=3.2.0->nncf==2.5.0.dev0+90a1e860) (0.10.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pandas<2.1,>=1.1.5->nncf==2.5.0.dev0+90a1e860) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pandas<2.1,>=1.1.5->nncf==2.5.0.dev0+90a1e860) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pandas<2.1,>=1.1.5->nncf==2.5.0.dev0+90a1e860) (2023.3)
    Requirement already satisfied: joblib>=1.1.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from scikit-learn>=0.24.0->nncf==2.5.0.dev0+90a1e860) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from scikit-learn>=0.24.0->nncf==2.5.0.dev0+90a1e860) (3.2.0)
    Requirement already satisfied: matplotlib>=3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (3.5.2)
    Requirement already satisfied: autograd>=1.4 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (1.6.2)
    Requirement already satisfied: cma==3.2.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (3.2.2)
    Requirement already satisfied: alive-progress in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (3.1.4)
    Requirement already satisfied: dill in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (0.3.7)
    Requirement already satisfied: Deprecated in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (1.2.14)
    Requirement already satisfied: future>=0.15.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from autograd>=1.4->pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (0.18.3)
    Requirement already satisfied: zipp>=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from importlib-resources>=1.4.0->jsonschema>=3.2.0->nncf==2.5.0.dev0+90a1e860) (3.16.2)
    Requirement already satisfied: cycler>=0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib>=3->pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib>=3->pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (4.42.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib>=3->pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (1.4.5)
    Requirement already satisfied: pillow>=6.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib>=3->pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (10.0.0)
    Requirement already satisfied: six>=1.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from python-dateutil>=2.8.2->pandas<2.1,>=1.1.5->nncf==2.5.0.dev0+90a1e860) (1.16.0)
    Requirement already satisfied: about-time==4.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from alive-progress->pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (4.2.1)
    Requirement already satisfied: grapheme==0.6.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from alive-progress->pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (0.6.0)
    Requirement already satisfied: wrapt<2,>=1.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from Deprecated->pymoo@ git+https://github.com/anyoptimization/pymoo.git@695cb26923903f872c7256a9013609769f3cc2bd->nncf==2.5.0.dev0+90a1e860) (1.14.1)


Get Pytorch model and OpenVINO IR model
###############################################################################################################################

Generally, PyTorch models represent an instance of the
```torch.nn.Module`` <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the YOLOv8 nano model (also known as ``yolov8n``) pre-trained on a COCO
dataset, which is available in this
`repo <https://github.com/ultralytics/ultralytics>`__. Similar steps are
also applicable to other YOLOv8 models. Typical steps to obtain a
pre-trained model:

1. Create an instance of a model class.
2. Load a checkpoint state dict, which contains the pre-trained model
   weights.

In this case, the creators of the model provide an API that enables
converting the YOLOv8 model to ONNX and then to OpenVINO IR. Therefore,
we do not need to do these steps manually.

.. code:: ipython3

    import os
    from pathlib import Path
    
    from ultralytics import YOLO
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.data.utils import check_det_dataset
    from ultralytics.yolo.engine.validator import BaseValidator as Validator
    from ultralytics.yolo.utils import DATASETS_DIR
    from ultralytics.yolo.utils import DEFAULT_CFG
    from ultralytics.yolo.utils import ops
    from ultralytics.yolo.utils.metrics import ConfusionMatrix
    
    ROOT = os.path.abspath('')
    
    MODEL_NAME = "yolov8n-seg"
    
    model = YOLO(f"{ROOT}/{MODEL_NAME}.pt")
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128-seg.yaml"


.. parsed-literal::

    Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt to /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/122-quantizing-model-with-accuracy-control/yolov8n-seg.pt...



.. parsed-literal::

      0%|          | 0.00/6.73M [00:00<?, ?B/s]


Load model.

.. code:: ipython3

    import openvino as ov
    
    
    model_path = Path(f"{ROOT}/{MODEL_NAME}_openvino_model/{MODEL_NAME}.xml")
    if not model_path.exists():
        model.export(format="openvino", dynamic=True, half=False)
    
    ov_model = ov.Core().read_model(model_path)


.. parsed-literal::

    Ultralytics YOLOv8.0.43 🚀 Python-3.8.10 torch-1.13.1+cpu CPU
    YOLOv8n-seg summary (fused): 195 layers, 3404320 parameters, 0 gradients, 12.6 GFLOPs
    
    PyTorch: starting from /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/122-quantizing-model-with-accuracy-control/yolov8n-seg.pt with input shape (1, 3, 640, 640) BCHW and output shape(s) ((1, 116, 8400), (1, 32, 160, 160)) (6.7 MB)
    
    ONNX: starting export with onnx 1.14.1...
    ONNX: export success ✅ 0.6s, saved as /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/122-quantizing-model-with-accuracy-control/yolov8n-seg.onnx (13.1 MB)
    
    OpenVINO: starting export with openvino 2023.1.0-12050-e33de350633...
    OpenVINO: export success ✅ 0.7s, saved as /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/122-quantizing-model-with-accuracy-control/yolov8n-seg_openvino_model/ (13.3 MB)
    
    Export complete (1.5s)
    Results saved to /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/122-quantizing-model-with-accuracy-control
    Predict:         yolo predict task=segment model=/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/122-quantizing-model-with-accuracy-control/yolov8n-seg_openvino_model imgsz=640 
    Validate:        yolo val task=segment model=/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/122-quantizing-model-with-accuracy-control/yolov8n-seg_openvino_model imgsz=640 data=coco.yaml 
    Visualize:       https://netron.app


Define validator and data loader
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The original model
repository uses a ``Validator`` wrapper, which represents the accuracy
validation pipeline. It creates dataloader and evaluation metrics and
updates metrics on each data batch produced by the dataloader. Besides
that, it is responsible for data preprocessing and results
postprocessing. For class initialization, the configuration should be
provided. We will use the default setup, but it can be replaced with
some parameters overriding to test on custom data. The model has
connected the ``ValidatorClass`` method, which creates a validator class
instance.

.. code:: ipython3

    validator = model.ValidatorClass(args)
    validator.data = check_det_dataset(args.data)
    data_loader = validator.get_dataloader(f"{DATASETS_DIR}/coco128-seg", 1)
    
    validator.is_coco = True
    validator.class_map = ops.coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc
    validator.nm = 32
    validator.process = ops.process_mask
    validator.plot_masks = []


.. parsed-literal::

    val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-491/.workspace/scm/datasets/coco128-seg/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 [00:00<?, ?it/s]


Prepare calibration and validation datasets 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

We can use one dataset as calibration and validation datasets. Name it
``quantization_dataset``.

.. code:: ipython3

    from typing import Dict
    
    import nncf
    
    
    def transform_fn(data_item: Dict):
        input_tensor = validator.preprocess(data_item)["img"].numpy()
        return input_tensor
    
    
    quantization_dataset = nncf.Dataset(data_loader, transform_fn)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Prepare validation function
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    from functools import partial
    
    import torch
    from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
    
    
    def validation_ac(
        compiled_model: ov.CompiledModel,
        validation_loader: torch.utils.data.DataLoader,
        validator: Validator,
        num_samples: int = None,
    ) -> float:
        validator.seen = 0
        validator.jdict = []
        validator.stats = []
        validator.batch_i = 1
        validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
        num_outputs = len(compiled_model.outputs)
    
        counter = 0
        for batch_i, batch in enumerate(validation_loader):
            if num_samples is not None and batch_i == num_samples:
                break
            batch = validator.preprocess(batch)
            results = compiled_model(batch["img"])
            if num_outputs == 1:
                preds = torch.from_numpy(results[compiled_model.output(0)])
            else:
                preds = [
                    torch.from_numpy(results[compiled_model.output(0)]),
                    torch.from_numpy(results[compiled_model.output(1)]),
                ]
            preds = validator.postprocess(preds)
            validator.update_metrics(preds, batch)
            counter += 1
        stats = validator.get_stats()
        if num_outputs == 1:
            stats_metrics = stats["metrics/mAP50-95(B)"]
        else:
            stats_metrics = stats["metrics/mAP50-95(M)"]
        print(f"Validate: dataset length = {counter}, metric value = {stats_metrics:.3f}")
        
        return stats_metrics
    
    
    validation_fn = partial(validation_ac, validator=validator)

Run quantization with accuracy control
###############################################################################################################################

You should provide
the calibration dataset and the validation dataset. It can be the same
dataset. - parameter ``max_drop`` defines the accuracy drop threshold.
The quantization process stops when the degradation of accuracy metric
on the validation dataset is less than the ``max_drop``. The default
value is 0.01. NNCF will stop the quantization and report an error if
the ``max_drop`` value can’t be reached. - ``drop_type`` defines how the
accuracy drop will be calculated: ABSOLUTE (used by default) or
RELATIVE. - ``ranking_subset_size`` - size of a subset that is used to
rank layers by their contribution to the accuracy drop. Default value is
300, and the more samples it has the better ranking, potentially. Here
we use the value 25 to speed up the execution.

.. note::

   Execution can take tens of minutes and requires up to 15 GB
   of free memory

.. code:: ipython3

    quantized_model = nncf.quantize_with_accuracy_control(
        ov_model,
        quantization_dataset,
        quantization_dataset,
        validation_fn=validation_fn,
        max_drop=0.01,
        preset=nncf.QuantizationPreset.MIXED,
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(
            ranking_subset_size=25,
            num_ranking_processes=1
        ),
    )


.. parsed-literal::

    2023-09-08 23:17:54.173599: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-09-08 23:17:54.207357: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-09-08 23:17:54.764356: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Statistics collection:  43%|████▎     | 128/300 [00:16<00:22,  7.55it/s]
    Applying Fast Bias correction: 100%|██████████| 75/75 [00:04<00:00, 17.89it/s]

.. parsed-literal::

    INFO:nncf:Validation of initial model was started


.. parsed-literal::

    INFO:nncf:Elapsed Time: 00:00:00
    Validate: dataset length = 1, metric value = 0.589
    Validate: dataset length = 128, metric value = 0.366
    INFO:nncf:Elapsed Time: 00:00:04
    INFO:nncf:Metric of initial model: 0.36611468358574506
    INFO:nncf:Collecting values for each data item using the initial model
    Validate: dataset length = 1, metric value = 0.589
    Validate: dataset length = 1, metric value = 0.622
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.895
    Validate: dataset length = 1, metric value = 0.846
    Validate: dataset length = 1, metric value = 0.365
    Validate: dataset length = 1, metric value = 0.432
    Validate: dataset length = 1, metric value = 0.172
    Validate: dataset length = 1, metric value = 0.771
    Validate: dataset length = 1, metric value = 0.255
    Validate: dataset length = 1, metric value = 0.431
    Validate: dataset length = 1, metric value = 0.399
    Validate: dataset length = 1, metric value = 0.671
    Validate: dataset length = 1, metric value = 0.315
    Validate: dataset length = 1, metric value = 0.995
    Validate: dataset length = 1, metric value = 0.895
    Validate: dataset length = 1, metric value = 0.497
    Validate: dataset length = 1, metric value = 0.594
    Validate: dataset length = 1, metric value = 0.746
    Validate: dataset length = 1, metric value = 0.597
    Validate: dataset length = 1, metric value = 0.074
    Validate: dataset length = 1, metric value = 0.231
    Validate: dataset length = 1, metric value = 0.502
    Validate: dataset length = 1, metric value = 0.347
    Validate: dataset length = 1, metric value = 0.398
    Validate: dataset length = 1, metric value = 0.477
    Validate: dataset length = 1, metric value = 0.537
    Validate: dataset length = 1, metric value = 0.344
    Validate: dataset length = 1, metric value = 0.544
    Validate: dataset length = 1, metric value = 0.237
    Validate: dataset length = 1, metric value = 0.109
    Validate: dataset length = 1, metric value = 0.564
    Validate: dataset length = 1, metric value = 0.853
    Validate: dataset length = 1, metric value = 0.306
    Validate: dataset length = 1, metric value = 0.416
    Validate: dataset length = 1, metric value = 0.388
    Validate: dataset length = 1, metric value = 0.746
    Validate: dataset length = 1, metric value = 0.199
    Validate: dataset length = 1, metric value = 0.323
    Validate: dataset length = 1, metric value = 0.305
    Validate: dataset length = 1, metric value = 0.506
    Validate: dataset length = 1, metric value = 0.319
    Validate: dataset length = 1, metric value = 0.319
    Validate: dataset length = 1, metric value = 0.255
    Validate: dataset length = 1, metric value = 0.487
    Validate: dataset length = 1, metric value = 0.697
    Validate: dataset length = 1, metric value = 0.654
    Validate: dataset length = 1, metric value = 0.368
    Validate: dataset length = 1, metric value = 0.730
    Validate: dataset length = 1, metric value = 0.374
    Validate: dataset length = 1, metric value = 0.227
    Validate: dataset length = 1, metric value = 0.500
    Validate: dataset length = 1, metric value = 0.101
    Validate: dataset length = 1, metric value = 0.855
    Validate: dataset length = 1, metric value = 0.430
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.358
    Validate: dataset length = 1, metric value = 0.373
    Validate: dataset length = 1, metric value = 0.692
    Validate: dataset length = 1, metric value = 0.556
    Validate: dataset length = 1, metric value = 0.274
    Validate: dataset length = 1, metric value = 0.670
    Validate: dataset length = 1, metric value = 0.044
    Validate: dataset length = 1, metric value = 0.627
    Validate: dataset length = 1, metric value = 0.945
    Validate: dataset length = 1, metric value = 0.267
    Validate: dataset length = 1, metric value = 0.354
    Validate: dataset length = 1, metric value = 0.265
    Validate: dataset length = 1, metric value = 0.522
    Validate: dataset length = 1, metric value = 0.945
    Validate: dataset length = 1, metric value = 0.394
    Validate: dataset length = 1, metric value = 0.349
    Validate: dataset length = 1, metric value = 0.564
    Validate: dataset length = 1, metric value = 0.094
    Validate: dataset length = 1, metric value = 0.763
    Validate: dataset length = 1, metric value = 0.157
    Validate: dataset length = 1, metric value = 0.531
    Validate: dataset length = 1, metric value = 0.597
    Validate: dataset length = 1, metric value = 0.746
    Validate: dataset length = 1, metric value = 0.781
    Validate: dataset length = 1, metric value = 0.447
    Validate: dataset length = 1, metric value = 0.562
    Validate: dataset length = 1, metric value = 0.697
    Validate: dataset length = 1, metric value = 0.746
    Validate: dataset length = 1, metric value = 0.461
    Validate: dataset length = 1, metric value = 0.697
    Validate: dataset length = 1, metric value = 0.696
    Validate: dataset length = 1, metric value = 0.378
    Validate: dataset length = 1, metric value = 0.246
    Validate: dataset length = 1, metric value = 0.647
    Validate: dataset length = 1, metric value = 0.367
    Validate: dataset length = 1, metric value = 0.995
    Validate: dataset length = 1, metric value = 0.995
    Validate: dataset length = 1, metric value = 0.597
    Validate: dataset length = 1, metric value = 0.398
    Validate: dataset length = 1, metric value = 0.359
    Validate: dataset length = 1, metric value = 0.407
    Validate: dataset length = 1, metric value = 0.191
    Validate: dataset length = 1, metric value = 0.549
    Validate: dataset length = 1, metric value = 0.290
    Validate: dataset length = 1, metric value = 0.166
    Validate: dataset length = 1, metric value = 0.131
    Validate: dataset length = 1, metric value = 0.745
    Validate: dataset length = 1, metric value = 0.336
    Validate: dataset length = 1, metric value = 0.248
    Validate: dataset length = 1, metric value = 0.290
    Validate: dataset length = 1, metric value = 0.413
    Validate: dataset length = 1, metric value = 0.790
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.265
    Validate: dataset length = 1, metric value = 0.423
    Validate: dataset length = 1, metric value = 0.398
    Validate: dataset length = 1, metric value = 0.039
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.685
    Validate: dataset length = 1, metric value = 0.635
    Validate: dataset length = 1, metric value = 0.829
    Validate: dataset length = 1, metric value = 0.525
    Validate: dataset length = 1, metric value = 0.315
    Validate: dataset length = 1, metric value = 0.348
    Validate: dataset length = 1, metric value = 0.567
    Validate: dataset length = 1, metric value = 0.751
    Validate: dataset length = 1, metric value = 0.597
    Validate: dataset length = 1, metric value = 0.557
    Validate: dataset length = 1, metric value = 0.995
    Validate: dataset length = 1, metric value = 0.341
    Validate: dataset length = 1, metric value = 0.427
    Validate: dataset length = 1, metric value = 0.846
    INFO:nncf:Elapsed Time: 00:00:05
    INFO:nncf:Validation of quantized model was started
    INFO:nncf:Elapsed Time: 00:00:01
    Validate: dataset length = 128, metric value = 0.342
    INFO:nncf:Elapsed Time: 00:00:04
    INFO:nncf:Metric of quantized model: 0.3419095833156649
    INFO:nncf:Collecting values for each data item using the quantized model
    Validate: dataset length = 1, metric value = 0.513
    Validate: dataset length = 1, metric value = 0.647
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.895
    Validate: dataset length = 1, metric value = 0.846
    Validate: dataset length = 1, metric value = 0.448
    Validate: dataset length = 1, metric value = 0.426
    Validate: dataset length = 1, metric value = 0.165
    Validate: dataset length = 1, metric value = 0.697
    Validate: dataset length = 1, metric value = 0.255
    Validate: dataset length = 1, metric value = 0.464
    Validate: dataset length = 1, metric value = 0.427
    Validate: dataset length = 1, metric value = 0.631
    Validate: dataset length = 1, metric value = 0.307
    Validate: dataset length = 1, metric value = 0.895
    Validate: dataset length = 1, metric value = 0.895
    Validate: dataset length = 1, metric value = 0.531
    Validate: dataset length = 1, metric value = 0.518
    Validate: dataset length = 1, metric value = 0.696
    Validate: dataset length = 1, metric value = 0.647
    Validate: dataset length = 1, metric value = 0.142
    Validate: dataset length = 1, metric value = 0.205
    Validate: dataset length = 1, metric value = 0.487
    Validate: dataset length = 1, metric value = 0.331
    Validate: dataset length = 1, metric value = 0.348
    Validate: dataset length = 1, metric value = 0.415
    Validate: dataset length = 1, metric value = 0.542
    Validate: dataset length = 1, metric value = 0.333
    Validate: dataset length = 1, metric value = 0.489
    Validate: dataset length = 1, metric value = 0.270
    Validate: dataset length = 1, metric value = 0.067
    Validate: dataset length = 1, metric value = 0.564
    Validate: dataset length = 1, metric value = 0.764
    Validate: dataset length = 1, metric value = 0.301
    Validate: dataset length = 1, metric value = 0.400
    Validate: dataset length = 1, metric value = 0.392
    Validate: dataset length = 1, metric value = 0.696
    Validate: dataset length = 1, metric value = 0.193
    Validate: dataset length = 1, metric value = 0.199
    Validate: dataset length = 1, metric value = 0.267
    Validate: dataset length = 1, metric value = 0.484
    Validate: dataset length = 1, metric value = 0.299
    Validate: dataset length = 1, metric value = 0.299
    Validate: dataset length = 1, metric value = 0.255
    Validate: dataset length = 1, metric value = 0.431
    Validate: dataset length = 1, metric value = 0.697
    Validate: dataset length = 1, metric value = 0.623
    Validate: dataset length = 1, metric value = 0.348
    Validate: dataset length = 1, metric value = 0.763
    Validate: dataset length = 1, metric value = 0.354
    Validate: dataset length = 1, metric value = 0.129
    Validate: dataset length = 1, metric value = 0.507
    Validate: dataset length = 1, metric value = 0.082
    Validate: dataset length = 1, metric value = 0.855
    Validate: dataset length = 1, metric value = 0.398
    Validate: dataset length = 1, metric value = 0.746
    Validate: dataset length = 1, metric value = 0.381
    Validate: dataset length = 1, metric value = 0.384
    Validate: dataset length = 1, metric value = 0.586
    Validate: dataset length = 1, metric value = 0.503
    Validate: dataset length = 1, metric value = 0.172
    Validate: dataset length = 1, metric value = 0.540
    Validate: dataset length = 1, metric value = 0.027
    Validate: dataset length = 1, metric value = 0.561
    Validate: dataset length = 1, metric value = 0.945
    Validate: dataset length = 1, metric value = 0.170
    Validate: dataset length = 1, metric value = 0.409
    Validate: dataset length = 1, metric value = 0.272
    Validate: dataset length = 1, metric value = 0.507
    Validate: dataset length = 1, metric value = 0.945
    Validate: dataset length = 1, metric value = 0.377
    Validate: dataset length = 1, metric value = 0.343
    Validate: dataset length = 1, metric value = 0.564
    Validate: dataset length = 1, metric value = 0.080
    Validate: dataset length = 1, metric value = 0.721
    Validate: dataset length = 1, metric value = 0.174
    Validate: dataset length = 1, metric value = 0.564
    Validate: dataset length = 1, metric value = 0.497
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.746
    Validate: dataset length = 1, metric value = 0.454
    Validate: dataset length = 1, metric value = 0.536
    Validate: dataset length = 1, metric value = 0.647
    Validate: dataset length = 1, metric value = 0.746
    Validate: dataset length = 1, metric value = 0.461
    Validate: dataset length = 1, metric value = 0.697
    Validate: dataset length = 1, metric value = 0.746
    Validate: dataset length = 1, metric value = 0.332
    Validate: dataset length = 1, metric value = 0.218
    Validate: dataset length = 1, metric value = 0.547
    Validate: dataset length = 1, metric value = 0.309
    Validate: dataset length = 1, metric value = 0.995
    Validate: dataset length = 1, metric value = 0.995
    Validate: dataset length = 1, metric value = 0.597
    Validate: dataset length = 1, metric value = 0.398
    Validate: dataset length = 1, metric value = 0.309
    Validate: dataset length = 1, metric value = 0.423
    Validate: dataset length = 1, metric value = 0.146
    Validate: dataset length = 1, metric value = 0.535
    Validate: dataset length = 1, metric value = 0.274
    Validate: dataset length = 1, metric value = 0.166
    Validate: dataset length = 1, metric value = 0.111
    Validate: dataset length = 1, metric value = 0.585
    Validate: dataset length = 1, metric value = 0.351
    Validate: dataset length = 1, metric value = 0.327
    Validate: dataset length = 1, metric value = 0.260
    Validate: dataset length = 1, metric value = 0.411
    Validate: dataset length = 1, metric value = 0.788
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.265
    Validate: dataset length = 1, metric value = 0.442
    Validate: dataset length = 1, metric value = 0.398
    Validate: dataset length = 1, metric value = 0.029
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.613
    Validate: dataset length = 1, metric value = 0.610
    Validate: dataset length = 1, metric value = 0.796
    Validate: dataset length = 1, metric value = 0.457
    Validate: dataset length = 1, metric value = 0.323
    Validate: dataset length = 1, metric value = 0.348
    Validate: dataset length = 1, metric value = 0.600
    Validate: dataset length = 1, metric value = 0.854
    Validate: dataset length = 1, metric value = 0.597
    Validate: dataset length = 1, metric value = 0.567
    Validate: dataset length = 1, metric value = 0.995
    Validate: dataset length = 1, metric value = 0.325
    Validate: dataset length = 1, metric value = 0.398
    Validate: dataset length = 1, metric value = 0.796
    INFO:nncf:Elapsed Time: 00:00:04
    INFO:nncf:Accuracy drop: 0.02420510027008016 (DropType.ABSOLUTE)
    INFO:nncf:Accuracy drop: 0.02420510027008016 (DropType.ABSOLUTE)
    INFO:nncf:Total number of quantized operations in the model: 91
    INFO:nncf:Number of parallel processes to rank quantized operations: 1
    INFO:nncf:ORIGINAL metric is used to rank quantizers
    INFO:nncf:Calculating ranking score for groups of quantizers
    Validate: dataset length = 25, metric value = 0.523
    Validate: dataset length = 25, metric value = 0.517
    Validate: dataset length = 25, metric value = 0.504
    Validate: dataset length = 25, metric value = 0.516
    Validate: dataset length = 25, metric value = 0.502
    Validate: dataset length = 25, metric value = 0.507
    Validate: dataset length = 25, metric value = 0.505
    Validate: dataset length = 25, metric value = 0.503
    Validate: dataset length = 25, metric value = 0.504
    Validate: dataset length = 25, metric value = 0.501
    Validate: dataset length = 25, metric value = 0.502
    Validate: dataset length = 25, metric value = 0.503
    Validate: dataset length = 25, metric value = 0.500
    Validate: dataset length = 25, metric value = 0.502
    Validate: dataset length = 25, metric value = 0.509
    Validate: dataset length = 25, metric value = 0.507
    Validate: dataset length = 25, metric value = 0.506
    Validate: dataset length = 25, metric value = 0.505
    Validate: dataset length = 25, metric value = 0.504
    Validate: dataset length = 25, metric value = 0.505
    Validate: dataset length = 25, metric value = 0.503
    Validate: dataset length = 25, metric value = 0.503
    Validate: dataset length = 25, metric value = 0.501
    Validate: dataset length = 25, metric value = 0.502
    Validate: dataset length = 25, metric value = 0.500
    Validate: dataset length = 25, metric value = 0.505
    Validate: dataset length = 25, metric value = 0.508
    Validate: dataset length = 25, metric value = 0.505
    Validate: dataset length = 25, metric value = 0.506
    Validate: dataset length = 25, metric value = 0.506
    Validate: dataset length = 25, metric value = 0.501
    Validate: dataset length = 25, metric value = 0.500
    Validate: dataset length = 25, metric value = 0.502
    Validate: dataset length = 25, metric value = 0.502
    Validate: dataset length = 25, metric value = 0.502
    Validate: dataset length = 25, metric value = 0.512
    Validate: dataset length = 25, metric value = 0.504
    Validate: dataset length = 25, metric value = 0.510
    Validate: dataset length = 25, metric value = 0.514
    Validate: dataset length = 25, metric value = 0.510
    Validate: dataset length = 25, metric value = 0.508
    Validate: dataset length = 25, metric value = 0.507
    Validate: dataset length = 25, metric value = 0.509
    Validate: dataset length = 25, metric value = 0.495
    Validate: dataset length = 25, metric value = 0.510
    Validate: dataset length = 25, metric value = 0.511
    Validate: dataset length = 25, metric value = 0.502
    Validate: dataset length = 25, metric value = 0.511
    Validate: dataset length = 25, metric value = 0.507
    Validate: dataset length = 25, metric value = 0.506
    Validate: dataset length = 25, metric value = 0.515
    Validate: dataset length = 25, metric value = 0.506
    Validate: dataset length = 25, metric value = 0.499
    Validate: dataset length = 25, metric value = 0.492
    Validate: dataset length = 25, metric value = 0.505
    Validate: dataset length = 25, metric value = 0.499
    Validate: dataset length = 25, metric value = 0.519
    Validate: dataset length = 25, metric value = 0.522
    Validate: dataset length = 25, metric value = 0.516
    INFO:nncf:Elapsed Time: 00:02:45
    INFO:nncf:Changing the scope of quantizer nodes was started
    INFO:nncf:Reverted 1 operations to the floating-point precision: 
    	/model.22/Mul_5
    Validate: dataset length = 128, metric value = 0.353
    INFO:nncf:Accuracy drop with the new quantization scope is 0.013362079004897942 (DropType.ABSOLUTE)
    INFO:nncf:Reverted 1 operations to the floating-point precision: 
    	/model.1/conv/Conv/WithoutBiases
    Validate: dataset length = 128, metric value = 0.353
    INFO:nncf:Accuracy drop with the new quantization scope is 0.013092546237331526 (DropType.ABSOLUTE)
    INFO:nncf:Reverted 1 operations to the floating-point precision: 
    	/model.2/cv1/conv/Conv/WithoutBiases
    Validate: dataset length = 128, metric value = 0.359
    INFO:nncf:Algorithm completed: achieved required accuracy drop 0.006690894581248108 (DropType.ABSOLUTE)
    INFO:nncf:3 out of 91 were reverted back to the floating-point precision:
    	/model.22/Mul_5
    	/model.1/conv/Conv/WithoutBiases
    	/model.2/cv1/conv/Conv/WithoutBiases


Compare Accuracy and Performance of the Original and Quantized Models
###############################################################################################################################

Now we can compare metrics of the Original non-quantized
OpenVINO IR model and Quantized OpenVINO IR model to make sure that the
``max_drop`` is not exceeded.

.. code:: ipython3

    core = ov.Core()
    quantized_compiled_model = core.compile_model(model=quantized_model, device_name='CPU')
    compiled_ov_model = core.compile_model(model=ov_model, device_name='CPU')
    
    pt_result = validation_ac(compiled_ov_model, data_loader, validator)
    quantized_result = validation_ac(quantized_compiled_model, data_loader, validator)
    
    
    print(f'[Original OpenVino]: {pt_result:.4f}')
    print(f'[Quantized OpenVino]: {quantized_result:.4f}')


.. parsed-literal::

    Validate: dataset length = 128, metric value = 0.368
    Validate: dataset length = 128, metric value = 0.361
    [Original OpenVino]: 0.3677
    [Quantized OpenVino]: 0.3605


And compare performance.

.. code:: ipython3

    from pathlib import Path
    # Set model directory
    MODEL_DIR = Path("model")
    MODEL_DIR.mkdir(exist_ok=True)
    
    ir_model_path = MODEL_DIR / 'ir_model.xml'
    quantized_model_path = MODEL_DIR / 'quantized_model.xml'
    
    # Save models to use them in the commandline banchmark app
    ov.save_model(ov_model, ir_model_path, compress_to_fp16=False)
    ov.save_model(quantized_model, quantized_model_path, compress_to_fp16=False)

.. code:: ipython3

    # Inference Original model (OpenVINO IR)
    ! benchmark_app -m $ir_model_path -shape "[1,3,640,640]" -d CPU -api async


.. parsed-literal::

    /bin/bash: benchmark_app: command not found


.. code:: ipython3

    # Inference Quantized model (OpenVINO IR)
    ! benchmark_app -m $quantized_model_path -shape "[1,3,640,640]" -d CPU -api async


.. parsed-literal::

    /bin/bash: benchmark_app: command not found

