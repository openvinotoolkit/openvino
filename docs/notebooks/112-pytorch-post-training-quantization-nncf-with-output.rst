Post-Training Quantization of PyTorch models with NNCF
======================================================

The goal of this tutorial is to demonstrate how to use the NNCF (Neural
Network Compression Framework) 8-bit quantization in post-training mode
(without the fine-tuning pipeline) to optimize a PyTorch model for the
high-speed inference via OpenVINO™ Toolkit. The optimization process
contains the following steps:

1. Evaluate the original model.
2. Transform the original model to a quantized one.
3. Export optimized and original models to OpenVINO IR.
4. Compare performance of the obtained ``FP32`` and ``INT8`` models.

This tutorial uses a ResNet-50 model, pre-trained on Tiny ImageNet,
which contains 100000 images of 200 classes (500 for each class)
downsized to 64×64 colored images. The tutorial will demonstrate that
only a tiny part of the dataset is needed for the post-training
quantization, not demanding the fine-tuning of the model.

   **NOTE**: This notebook requires that a C++ compiler is accessible on
   the default binary search path of the OS you are running the
   notebook.

Preparations
------------

.. code:: ipython3

    # On Windows, this script adds the directory that contains cl.exe to the PATH to enable PyTorch to find the
    # required C++ tools. This code assumes that Visual Studio 2019 is installed in the default
    # directory. If you have a different C++ compiler, add the correct path to os.environ["PATH"]
    # directly.
    
    # Adding the path to os.environ["LIB"] is not always required - it depends on the system configuration.
    
    import sys
    
    if sys.platform == "win32":
        import distutils.command.build_ext
        import os
        from pathlib import Path
    
        VS_INSTALL_DIR = r"C:/Program Files (x86)/Microsoft Visual Studio"
        cl_paths = sorted(list(Path(VS_INSTALL_DIR).glob("**/Hostx86/x64/cl.exe")))
        if len(cl_paths) == 0:
            raise ValueError(
                "Cannot find Visual Studio. This notebook requires C++. If you installed "
                "a C++ compiler, please add the directory that contains cl.exe to "
                "`os.environ['PATH']`"
            )
        else:
            # If multiple versions of MSVC are installed, get the most recent one.
            cl_path = cl_paths[-1]
            vs_dir = str(cl_path.parent)
            os.environ["PATH"] += f"{os.pathsep}{vs_dir}"
            # The code for finding the library dirs is from
            # https://stackoverflow.com/questions/47423246/get-pythons-lib-path
            d = distutils.core.Distribution()
            b = distutils.command.build_ext.build_ext(d)
            b.finalize_options()
            os.environ["LIB"] = os.pathsep.join(b.library_dirs)
            print(f"Added {vs_dir} to PATH")

Imports
~~~~~~~

.. code:: ipython3

    import os
    import time
    import zipfile
    from pathlib import Path
    from typing import List, Tuple
    
    import nncf
    from openvino.runtime import Core, serialize
    from openvino.tools import mo
    
    import torch
    from torchvision.datasets import ImageFolder
    from torchvision.models import resnet50
    import torchvision.transforms as transforms
    
    sys.path.append("../utils")
    from notebook_utils import download_file


.. parsed-literal::

    2023-05-29 22:55:16.336185: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-05-29 22:55:16.370677: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-05-29 22:55:16.922906: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/offline_transformations/__init__.py:10: FutureWarning: The module is private and following namespace `offline_transformations` will be removed in the future, use `openvino.runtime.passes` instead!
      warnings.warn(


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Settings
~~~~~~~~

.. code:: ipython3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    MODEL_DIR = Path("model")
    OUTPUT_DIR = Path("output")
    BASE_MODEL_NAME = "resnet50"
    IMAGE_SIZE = [64, 64]
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Paths where PyTorch and OpenVINO IR models will be stored.
    fp32_checkpoint_filename = Path(BASE_MODEL_NAME + "_fp32").with_suffix(".pth")
    fp32_onnx_path = OUTPUT_DIR / Path(BASE_MODEL_NAME + "_fp32").with_suffix(".onnx")
    fp32_ir_path = OUTPUT_DIR / Path(BASE_MODEL_NAME + "_fp32").with_suffix(".xml")
    int8_onnx_path = OUTPUT_DIR / Path(BASE_MODEL_NAME + "_int8").with_suffix(".onnx")
    int8_ir_path = OUTPUT_DIR / Path(BASE_MODEL_NAME + "_int8").with_suffix(".xml")
    
    
    fp32_pth_url = "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/304_resnet50_fp32.pth"
    download_file(fp32_pth_url, directory=MODEL_DIR, filename=fp32_checkpoint_filename)


.. parsed-literal::

    Using cpu device



.. parsed-literal::

    model/resnet50_fp32.pth:   0%|          | 0.00/91.5M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/notebooks/112-pytorch-post-training-quantization-nncf/model/resnet50_fp32.pth')



Download and Prepare Tiny ImageNet dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  100k images of shape 3x64x64,
-  200 different classes: snake, spider, cat, truck, grasshopper, gull,
   etc.

.. code:: ipython3

    def download_tiny_imagenet_200(
        output_dir: Path,
        url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        tarname: str = "tiny-imagenet-200.zip",
    ):
        archive_path = output_dir / tarname
        download_file(url, directory=output_dir, filename=tarname)
        zip_ref = zipfile.ZipFile(archive_path, "r")
        zip_ref.extractall(path=output_dir)
        zip_ref.close()
        print(f"Successfully downloaded and extracted dataset to: {output_dir}")
    
    
    def create_validation_dir(dataset_dir: Path):
        VALID_DIR = dataset_dir / "val"
        val_img_dir = VALID_DIR / "images"
    
        fp = open(VALID_DIR / "val_annotations.txt", "r")
        data = fp.readlines()
    
        val_img_dict = {}
        for line in data:
            words = line.split("\t")
            val_img_dict[words[0]] = words[1]
        fp.close()
    
        for img, folder in val_img_dict.items():
            newpath = val_img_dir / folder
            if not newpath.exists():
                os.makedirs(newpath)
            if (val_img_dir / img).exists():
                os.rename(val_img_dir / img, newpath / img)
    
    
    DATASET_DIR = OUTPUT_DIR / "tiny-imagenet-200"
    if not DATASET_DIR.exists():
        download_tiny_imagenet_200(OUTPUT_DIR)
        create_validation_dir(DATASET_DIR)



.. parsed-literal::

    output/tiny-imagenet-200.zip:   0%|          | 0.00/237M [00:00<?, ?B/s]


.. parsed-literal::

    Successfully downloaded and extracted dataset to: output


Helpers classes and functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code below will help to count accuracy and visualize validation
process.

.. code:: ipython3

    class AverageMeter(object):
        """Computes and stores the average and current value"""
    
        def __init__(self, name: str, fmt: str = ":f"):
            self.name = name
            self.fmt = fmt
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    
        def update(self, val: float, n: int = 1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
        def __str__(self):
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)
    
    
    class ProgressMeter(object):
        """Displays the progress of validation process"""
    
        def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix
    
        def display(self, batch: int):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print("\t".join(entries))
    
        def _get_batch_fmtstr(self, num_batches: int):
            num_digits = len(str(num_batches // 1))
            fmt = "{:" + str(num_digits) + "d}"
            return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
    
    def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
    
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
    
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
    
            return res

Validation function
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from typing import Union
    from openvino.runtime.ie_api import CompiledModel
    
    
    def validate(val_loader: torch.utils.data.DataLoader, model: Union[torch.nn.Module, CompiledModel]):
        """Compute the metrics using data from val_loader for the model"""
        batch_time = AverageMeter("Time", ":3.3f")
        top1 = AverageMeter("Acc@1", ":2.2f")
        top5 = AverageMeter("Acc@5", ":2.2f")
        progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")
        start_time = time.time()
        # Switch to evaluate mode.
        if not isinstance(model, CompiledModel):
            model.eval()
            model.to(device)
    
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(device)
                target = target.to(device)
    
                # Compute the output.
                if isinstance(model, CompiledModel):
                    output_layer = model.output(0)
                    output = model(images)[output_layer]
                    output = torch.from_numpy(output)
                else:
                    output = model(images)
    
                # Measure accuracy and record loss.
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
    
                # Measure elapsed time.
                batch_time.update(time.time() - end)
                end = time.time()
    
                print_frequency = 10
                if i % print_frequency == 0:
                    progress.display(i)
    
            print(
                " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Total time: {total_time:.3f}".format(top1=top1, top5=top5, total_time=end - start_time)
            )
        return top1.avg

Create and load original uncompressed model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ResNet-50 from the `torchivision
repository <https://github.com/pytorch/vision>`__ is pre-trained on
ImageNet with more prediction classes than Tiny ImageNet, so the model
is adjusted by swapping the last FC layer to one with fewer output
values.

.. code:: ipython3

    def create_model(model_path: Path):
        """Creates the ResNet-50 model and loads the pretrained weights"""
        model = resnet50()
        # Update the last FC layer for Tiny ImageNet number of classes.
        NUM_CLASSES = 200
        model.fc = torch.nn.Linear(in_features=2048, out_features=NUM_CLASSES, bias=True)
        model.to(device)
        if model_path.exists():
            checkpoint = torch.load(str(model_path), map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        else:
            raise RuntimeError("There is no checkpoint to load")
        return model
    
    
    model = create_model(MODEL_DIR / fp32_checkpoint_filename)

Create train and validation dataloaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def create_dataloaders(batch_size: int = 128):
        """Creates train dataloader that is used for quantization initialization and validation dataloader for computing the model accruacy"""
        train_dir = DATASET_DIR / "train"
        val_dir = DATASET_DIR / "val" / "images"
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_dataset = ImageFolder(
            train_dir,
            transforms.Compose(
                [
                    transforms.Resize(IMAGE_SIZE),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = ImageFolder(
            val_dir,
            transforms.Compose(
                [transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), normalize]
            ),
        )
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            sampler=None,
        )
    
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, val_loader
    
    
    train_loader, val_loader = create_dataloaders()

Model quantization and benchmarking
-----------------------------------

With the validation pipeline, model files, and data-loading procedures
for model calibration now prepared, it’s time to proceed with the actual
post-training quantization using NNCF.

I. Evaluate the loaded model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    acc1 = validate(val_loader, model)
    print(f"Test accuracy of FP32 model: {acc1:.3f}")


.. parsed-literal::

    Test: [ 0/79]	Time 0.257 (0.257)	Acc@1 81.25 (81.25)	Acc@5 92.19 (92.19)
    Test: [10/79]	Time 0.230 (0.235)	Acc@1 56.25 (66.97)	Acc@5 86.72 (87.50)
    Test: [20/79]	Time 0.231 (0.236)	Acc@1 67.97 (64.29)	Acc@5 85.16 (87.35)
    Test: [30/79]	Time 0.230 (0.239)	Acc@1 53.12 (62.37)	Acc@5 77.34 (85.33)
    Test: [40/79]	Time 0.232 (0.239)	Acc@1 67.19 (60.86)	Acc@5 90.62 (84.51)
    Test: [50/79]	Time 0.231 (0.238)	Acc@1 60.16 (60.80)	Acc@5 88.28 (84.42)
    Test: [60/79]	Time 0.253 (0.237)	Acc@1 66.41 (60.46)	Acc@5 86.72 (83.79)
    Test: [70/79]	Time 0.232 (0.236)	Acc@1 52.34 (60.21)	Acc@5 80.47 (83.33)
     * Acc@1 60.740 Acc@5 83.960 Total time: 18.431
    Test accuracy of FP32 model: 60.740


II. Create and initialize quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NNCF enables post-training quantization by adding the quantization
layers into the model graph and then using a subset of the training
dataset to initialize the parameters of these additional quantization
layers. The framework is designed so that modifications to your original
training code are minor. Quantization is the simplest scenario and
requires a few modifications. For more information about NNCF Post
Training Quantization (PTQ) API, refer to the `Basic Quantization Flow
Guide <https://docs.openvino.ai/latest/basic_qauntization_flow.html#doxid-basic-qauntization-flow>`__.

1. Create a transformation function that accepts a sample from the
   dataset and returns data suitable for model inference. This enables
   the creation of an instance of the nncf.Dataset class, which
   represents the calibration dataset (based on the training dataset)
   necessary for post-training quantization.

.. code:: ipython3

    def transform_fn(data_item):
        images, _ = data_item
        return images
    
    
    calibration_dataset = nncf.Dataset(train_loader, transform_fn)

2. Create a quantized model from the pre-trained ``FP32`` model and the
   calibration dataset.

.. code:: ipython3

    quantized_model = nncf.quantize(model, calibration_dataset)


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


.. parsed-literal::

    INFO:nncf:Collecting tensor statistics |█████           | 1 / 3
    INFO:nncf:Collecting tensor statistics |██████████      | 2 / 3
    INFO:nncf:Collecting tensor statistics |████████████████| 3 / 3
    INFO:nncf:Compiling and loading torch extension: quantized_functions_cpu...
    INFO:nncf:Finished loading torch extension: quantized_functions_cpu


3. Evaluate the new model on the validation set after initialization of
   quantization. The accuracy should be close to the accuracy of the
   floating-point ``FP32`` model for a simple case like the one being
   demonstrated now.

.. code:: ipython3

    acc1 = validate(val_loader, quantized_model)
    print(f"Accuracy of initialized INT8 model: {acc1:.3f}")


.. parsed-literal::

    Test: [ 0/79]	Time 0.385 (0.385)	Acc@1 80.47 (80.47)	Acc@5 91.41 (91.41)
    Test: [10/79]	Time 0.380 (0.380)	Acc@1 52.34 (66.48)	Acc@5 85.94 (87.50)
    Test: [20/79]	Time 0.377 (0.379)	Acc@1 68.75 (63.95)	Acc@5 85.94 (87.20)
    Test: [30/79]	Time 0.382 (0.379)	Acc@1 51.56 (62.22)	Acc@5 73.44 (85.23)
    Test: [40/79]	Time 0.380 (0.379)	Acc@1 67.19 (60.63)	Acc@5 89.84 (84.34)
    Test: [50/79]	Time 0.377 (0.379)	Acc@1 61.72 (60.66)	Acc@5 87.50 (84.24)
    Test: [60/79]	Time 0.372 (0.379)	Acc@1 64.84 (60.32)	Acc@5 85.94 (83.71)
    Test: [70/79]	Time 0.380 (0.378)	Acc@1 50.78 (60.00)	Acc@5 79.69 (83.27)
     * Acc@1 60.570 Acc@5 83.850 Total time: 29.644
    Accuracy of initialized INT8 model: 60.570


It should be noted that the inference time for the quantized PyTorch
model is longer than that of the original model, as fake quantizers are
added to the model by NNCF. However, the model’s performance will
significantly improve when it is in the OpenVINO Intermediate
Representation (IR) format.

III. Convert the models to OpenVINO Intermediate Representation (OpenVINO IR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Model Optimizer Python API to convert the Pytorch models to OpenVINO
IR. The models will be saved to the ‘OUTPUT’ directory for latter
benchmarking.

For more information about Model Optimizer, refer to the `Model
Optimizer Developer
Guide <https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__.

Before converting models export them to ONNX. Executing the following
command may take a while.

.. code:: ipython3

    dummy_input = torch.randn(128, 3, *IMAGE_SIZE)
    
    torch.onnx.export(model, dummy_input, fp32_onnx_path)
    model_ir = mo.convert_model(input_model=fp32_onnx_path, input_shape=[-1, 3, *IMAGE_SIZE])
    
    serialize(model_ir, str(fp32_ir_path))

.. code:: ipython3

    torch.onnx.export(quantized_model, dummy_input, int8_onnx_path)
    quantized_model_ir = mo.convert_model(input_model=int8_onnx_path, input_shape=[-1, 3, *IMAGE_SIZE])
    
    serialize(quantized_model_ir, str(int8_ir_path))

Evaluate the FP32 and INT8 models.

.. code:: ipython3

    core = Core()
    fp32_compiled_model = core.compile_model(model_ir)
    acc1 = validate(val_loader, fp32_compiled_model)
    print(f"Accuracy of FP32 IR model: {acc1:.3f}")


.. parsed-literal::

    Test: [ 0/79]	Time 0.168 (0.168)	Acc@1 81.25 (81.25)	Acc@5 92.19 (92.19)
    Test: [10/79]	Time 0.118 (0.122)	Acc@1 56.25 (66.97)	Acc@5 86.72 (87.50)
    Test: [20/79]	Time 0.116 (0.120)	Acc@1 67.97 (64.29)	Acc@5 85.16 (87.35)
    Test: [30/79]	Time 0.118 (0.119)	Acc@1 53.12 (62.37)	Acc@5 77.34 (85.33)
    Test: [40/79]	Time 0.117 (0.119)	Acc@1 67.19 (60.86)	Acc@5 90.62 (84.51)
    Test: [50/79]	Time 0.117 (0.119)	Acc@1 60.16 (60.80)	Acc@5 88.28 (84.42)
    Test: [60/79]	Time 0.118 (0.119)	Acc@1 66.41 (60.46)	Acc@5 86.72 (83.79)
    Test: [70/79]	Time 0.118 (0.119)	Acc@1 52.34 (60.21)	Acc@5 80.47 (83.33)
     * Acc@1 60.740 Acc@5 83.960 Total time: 9.280
    Accuracy of FP32 IR model: 60.740


.. code:: ipython3

    int8_compiled_model = core.compile_model(quantized_model_ir)
    acc1 = validate(val_loader, int8_compiled_model)
    print(f"Accuracy of INT8 IR model: {acc1:.3f}")


.. parsed-literal::

    Test: [ 0/79]	Time 0.116 (0.116)	Acc@1 80.47 (80.47)	Acc@5 91.41 (91.41)
    Test: [10/79]	Time 0.076 (0.082)	Acc@1 54.69 (66.83)	Acc@5 85.94 (87.71)
    Test: [20/79]	Time 0.077 (0.079)	Acc@1 69.53 (63.95)	Acc@5 85.94 (87.28)
    Test: [30/79]	Time 0.078 (0.079)	Acc@1 51.56 (62.17)	Acc@5 73.44 (85.26)
    Test: [40/79]	Time 0.079 (0.079)	Acc@1 68.75 (60.75)	Acc@5 89.84 (84.30)
    Test: [50/79]	Time 0.078 (0.078)	Acc@1 60.94 (60.71)	Acc@5 87.50 (84.15)
    Test: [60/79]	Time 0.078 (0.078)	Acc@1 64.84 (60.35)	Acc@5 85.94 (83.64)
    Test: [70/79]	Time 0.077 (0.078)	Acc@1 51.56 (60.05)	Acc@5 79.69 (83.24)
     * Acc@1 60.580 Acc@5 83.830 Total time: 6.118
    Accuracy of INT8 IR model: 60.580


IV. Compare performance of INT8 model and FP32 model in OpenVINO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/latest/openvino_inference_engine_tools_benchmark_tool_README.html>`__
- an inference performance measurement tool in OpenVINO. By default,
Benchmark Tool runs inference for 60 seconds in asynchronous mode on
CPU. It returns inference speed as latency (milliseconds per image) and
throughput (frames per second) values.

   **NOTE**: This notebook runs benchmark_app for 15 seconds to give a
   quick indication of performance. For more accurate performance, it is
   recommended to run benchmark_app in a terminal/command prompt after
   closing other applications. Run ``benchmark_app -m model.xml -d CPU``
   to benchmark async inference on CPU for one minute. Change CPU to GPU
   to benchmark on GPU. Run ``benchmark_app --help`` to see an overview
   of all command-line options.

.. code:: ipython3

    def parse_benchmark_output(benchmark_output: str):
        """Prints the output from benchmark_app in human-readable format"""
        parsed_output = [line for line in benchmark_output if 'FPS' in line]
        print(*parsed_output, sep='\n')
    
    
    print('Benchmark FP32 model (OpenVINO IR)')
    benchmark_output = ! benchmark_app -m "$fp32_ir_path" -d CPU -api async -t 15 -shape "[1, 3, 512, 512]"
    parse_benchmark_output(benchmark_output)
    
    print('Benchmark INT8 model (OpenVINO IR)')
    benchmark_output = ! benchmark_app -m "$int8_ir_path" -d CPU -api async -t 15 -shape "[1, 3, 512, 512]"
    parse_benchmark_output(benchmark_output)
    
    print('Benchmark FP32 model (OpenVINO IR) synchronously')
    benchmark_output = ! benchmark_app -m "$fp32_ir_path" -d CPU -api sync -t 15 -shape "[1, 3, 512, 512]"
    parse_benchmark_output(benchmark_output)
    
    print('Benchmark INT8 model (OpenVINO IR) synchronously')
    benchmark_output = ! benchmark_app -m "$int8_ir_path" -d CPU -api sync -t 15 -shape "[1, 3, 512, 512]"
    parse_benchmark_output(benchmark_output)


.. parsed-literal::

    Benchmark FP32 model (OpenVINO IR)
    [ INFO ] Throughput:   37.57 FPS
    Benchmark INT8 model (OpenVINO IR)
    [ INFO ] Throughput:   157.46 FPS
    Benchmark FP32 model (OpenVINO IR) synchronously
    [ INFO ] Throughput:   38.73 FPS
    Benchmark INT8 model (OpenVINO IR) synchronously
    [ INFO ] Throughput:   140.53 FPS


Show CPU Information for reference:

.. code:: ipython3

    ie = Core()
    ie.get_property("CPU", "FULL_DEVICE_NAME")




.. parsed-literal::

    'Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz'


