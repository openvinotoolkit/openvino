Quantization Aware Training with NNCF, using PyTorch framework
==============================================================

This notebook is based on `ImageNet training in
PyTorch <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`__.

The goal of this notebook is to demonstrate how to use the Neural
Network Compression Framework
`NNCF <https://github.com/openvinotoolkit/nncf>`__ 8-bit quantization to
optimize a PyTorch model for inference with OpenVINO Toolkit. The
optimization process contains the following steps:

-  Transforming the original ``FP32`` model to ``INT8``
-  Using fine-tuning to restore the accuracy.
-  Exporting optimized and original models to OpenVINO IR
-  Measuring and comparing the performance of models.

For more advanced usage, refer to these
`examples <https://github.com/openvinotoolkit/nncf/tree/develop/examples>`__.

This tutorial uses the ResNet-18 model with the Tiny ImageNet-200
dataset. ResNet-18 is the version of ResNet models that contains the
fewest layers (18). Tiny ImageNet-200 is a subset of the larger ImageNet
dataset with smaller images. The dataset will be downloaded in the
notebook. Using the smaller model and dataset will speed up training and
download time. To see other ResNet models, visit `PyTorch
hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__.

   **NOTE**: This notebook requires a C++ compiler.

**Table of contents:**


-  `Imports and Settings <#imports-and-settings>`__
-  `Pre-train Floating-Point
   Model <#pre-train-floating-point-model>`__

   -  `Train Function <#train-function>`__
   -  `Validate Function <#validate-function>`__
   -  `Helpers <#helpers>`__
   -  `Get a Pre-trained FP32
      Model <#get-a-pre-trained-fp-model>`__

-  `Create and Initialize
   Quantization <#create-and-initialize-quantization>`__
-  `Fine-tune the Compressed
   Model <#fine-tune-the-compressed-model>`__
-  `Export INT8 Model to OpenVINO
   IR <#export-int-model-to-openvino-ir>`__
-  `Benchmark Model Performance by Computing Inference
   Time <#benchmark-model-performance-by-computing-inference-time>`__

.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu  "openvino>=2023.1.0" "torch" "torchvision"
    %pip install -q "nncf>=2.6.0"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.


Imports and Settings 
--------------------------------------------------------------

On Windows, add the required C++ directories to the system PATH.

Import NNCF and all auxiliary packages from your Python code. Set a name
for the model, and the image width and height that will be used for the
network. Also define paths where PyTorch and OpenVINO IR versions of the
models will be stored.

   **NOTE**: All NNCF logging messages below ERROR level (INFO and
   WARNING) are disabled to simplify the tutorial. For production use,
   it is recommended to enable logging by removing
   ``set_log_level(logging.ERROR)``.

.. code:: ipython3

    # On Windows, add the directory that contains cl.exe to the PATH to enable PyTorch to find the
    # required C++ tools. This code assumes that Visual Studio 2019 is installed in the default
    # directory. If you have a different C++ compiler, add the correct path to os.environ["PATH"]
    # directly. Note that the C++ Redistributable is not enough to run this notebook.
    
    # Adding the path to os.environ["LIB"] is not always required - it depends on the system configuration
    
    import sys
    
    if sys.platform == "win32":
        import distutils.command.build_ext
        import os
        from pathlib import Path
    
        VS_INSTALL_DIR = r"C:/Program Files (x86)/Microsoft Visual Studio"
        cl_paths = sorted(list(Path(VS_INSTALL_DIR).glob("**/Hostx86/x64/cl.exe")))
        if len(cl_paths) == 0:
            raise ValueError(
                "Cannot find Visual Studio. This notebook requires a C++ compiler. If you installed "
                "a C++ compiler, please add the directory that contains cl.exe to `os.environ['PATH']`."
            )
        else:
            # If multiple versions of MSVC are installed, get the most recent one.
            cl_path = cl_paths[-1]
            vs_dir = str(cl_path.parent)
            os.environ["PATH"] += f"{os.pathsep}{vs_dir}"
            # The code for finding the library dirs is from:
            # https://stackoverflow.com/questions/47423246/get-pythons-lib-path
            d = distutils.core.Distribution()
            b = distutils.command.build_ext.build_ext(d)
            b.finalize_options()
            os.environ["LIB"] = os.pathsep.join(b.library_dirs)
            print(f"Added {vs_dir} to PATH")

.. code:: ipython3

    import sys
    import time
    import warnings  # To disable warnings on export model
    import zipfile
    from pathlib import Path
    import logging
    
    import torch
    import nncf  # Important - should be imported directly after torch.
    
    import torch.nn as nn
    import torch.nn.parallel
    import torch.optim
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.datasets as datasets
    import torchvision.models as models
    import torchvision.transforms as transforms
    
    from nncf.common.logging.logger import set_log_level
    set_log_level(logging.ERROR)  # Disables all NNCF info and warning messages.
    from nncf import NNCFConfig
    from nncf.torch import create_compressed_model, register_default_init_args
    import openvino as ov
    from torch.jit import TracerWarning
    
    sys.path.append("../utils")
    from notebook_utils import download_file
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    MODEL_DIR = Path("model")
    OUTPUT_DIR = Path("output")
    DATA_DIR = Path("data")
    BASE_MODEL_NAME = "resnet18"
    image_size = 64
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Paths where PyTorch and OpenVINO IR models will be stored.
    fp32_pth_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".pth")
    fp32_ir_path = fp32_pth_path.with_suffix(".xml")
    int8_ir_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_int8")).with_suffix(".xml")
    
    # It is possible to train FP32 model from scratch, but it might be slow. Therefore, the pre-trained weights are downloaded by default.
    pretrained_on_tiny_imagenet = True
    fp32_pth_url = "https://storage.openvinotoolkit.org/repositories/nncf/openvino_notebook_ckpts/302_resnet18_fp32_v1.pth"
    download_file(fp32_pth_url, directory=MODEL_DIR, filename=fp32_pth_path.name)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'


.. parsed-literal::

    Using cpu device



.. parsed-literal::

    model/resnet18_fp32.pth:   0%|          | 0.00/43.1M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/302-pytorch-quantization-aware-training/model/resnet18_fp32.pth')



Download Tiny ImageNet dataset

-  100k images of shape 3x64x64
-  200 different classes: snake, spider, cat, truck, grasshopper, gull,
   etc.

.. code:: ipython3

    def download_tiny_imagenet_200(
        data_dir: Path,
        url="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        tarname="tiny-imagenet-200.zip",
    ):
        archive_path = data_dir / tarname
        download_file(url, directory=data_dir, filename=tarname)
        zip_ref = zipfile.ZipFile(archive_path, "r")
        zip_ref.extractall(path=data_dir)
        zip_ref.close()
    
    def prepare_tiny_imagenet_200(dataset_dir: Path):
        # Format validation set the same way as train set is formatted.
        val_data_dir = dataset_dir / 'val'
        val_annotations_file = val_data_dir / 'val_annotations.txt'
        with open(val_annotations_file, 'r') as f:
            val_annotation_data = map(lambda line: line.split('\t')[:2], f.readlines())
        val_images_dir = val_data_dir / 'images'
        for image_filename, image_label in val_annotation_data:
            from_image_filepath = val_images_dir / image_filename
            to_image_dir = val_data_dir / image_label
            if not to_image_dir.exists():
                to_image_dir.mkdir()
            to_image_filepath = to_image_dir / image_filename
            from_image_filepath.rename(to_image_filepath)
        val_annotations_file.unlink()
        val_images_dir.rmdir()
        
    
    DATASET_DIR = DATA_DIR / "tiny-imagenet-200"
    if not DATASET_DIR.exists():
        download_tiny_imagenet_200(DATA_DIR)
        prepare_tiny_imagenet_200(DATASET_DIR)
        print(f"Successfully downloaded and prepared dataset at: {DATASET_DIR}")



.. parsed-literal::

    data/tiny-imagenet-200.zip:   0%|          | 0.00/237M [00:00<?, ?B/s]


.. parsed-literal::

    Successfully downloaded and prepared dataset at: data/tiny-imagenet-200


Pre-train Floating-Point Model 
------------------------------------------------------------------------

Using NNCF for model compression assumes that a pre-trained model and a
training pipeline are already in use.

This tutorial demonstrates one possible training pipeline: a ResNet-18
model pre-trained on 1000 classes from ImageNet is fine-tuned with 200
classes from Tiny-ImageNet.

Subsequently, the training and validation functions will be reused as is
for quantization-aware training.

Train Function 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def train(train_loader, model, criterion, optimizer, epoch):
        batch_time = AverageMeter("Time", ":3.3f")
        losses = AverageMeter("Loss", ":2.3f")
        top1 = AverageMeter("Acc@1", ":2.2f")
        top5 = AverageMeter("Acc@5", ":2.2f")
        progress = ProgressMeter(
            len(train_loader), [batch_time, losses, top1, top5], prefix="Epoch:[{}]".format(epoch)
        )
    
        # Switch to train mode.
        model.train()
    
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            images = images.to(device)
            target = target.to(device)
    
            # Compute output.
            output = model(images)
            loss = criterion(output, target)
    
            # Measure accuracy and record loss.
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
    
            # Compute gradient and do opt step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Measure elapsed time.
            batch_time.update(time.time() - end)
            end = time.time()
    
            print_frequency = 50
            if i % print_frequency == 0:
                progress.display(i)

Validate Function 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def validate(val_loader, model, criterion):
        batch_time = AverageMeter("Time", ":3.3f")
        losses = AverageMeter("Loss", ":2.3f")
        top1 = AverageMeter("Acc@1", ":2.2f")
        top5 = AverageMeter("Acc@5", ":2.2f")
        progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5], prefix="Test: ")
    
        # Switch to evaluate mode.
        model.eval()
    
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(device)
                target = target.to(device)
    
                # Compute output.
                output = model(images)
                loss = criterion(output, target)
    
                # Measure accuracy and record loss.
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
    
                # Measure elapsed time.
                batch_time.update(time.time() - end)
                end = time.time()
    
                print_frequency = 10
                if i % print_frequency == 0:
                    progress.display(i)
    
            print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5))
        return top1.avg

Helpers 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    class AverageMeter(object):
        """Computes and stores the average and current value"""
    
        def __init__(self, name, fmt=":f"):
            self.name = name
            self.fmt = fmt
            self.reset()
    
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
    
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
        def __str__(self):
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
            return fmtstr.format(**self.__dict__)
    
    
    class ProgressMeter(object):
        def __init__(self, num_batches, meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix
    
        def display(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print("\t".join(entries))
    
        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = "{:" + str(num_digits) + "d}"
            return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
    
    def accuracy(output, target, topk=(1,)):
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

Get a Pre-trained FP32 Model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

А pre-trained floating-point model is a prerequisite for quantization.
It can be obtained by tuning from scratch with the code below. However,
this usually takes a lot of time. Therefore, this code has already been
run and received good enough weights after 4 epochs (for the sake of
simplicity, tuning was not done until the best accuracy). By default,
this notebook just loads these weights without launching training. To
train the model yourself on a model pre-trained on ImageNet, set
``pretrained_on_tiny_imagenet = False`` in the Imports and Settings
section at the top of this notebook.

.. code:: ipython3

    num_classes = 200  # 200 is for Tiny ImageNet, default is 1000 for ImageNet
    init_lr = 1e-4
    batch_size = 128
    epochs = 4
    
    model = models.resnet18(pretrained=not pretrained_on_tiny_imagenet)
    # Update the last FC layer for Tiny ImageNet number of classes.
    model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    model.to(device)
    
    # Data loading code.
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, sampler=None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Define loss function (criterion) and optimizer.
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
      warnings.warn(msg)


.. code:: ipython3

    if pretrained_on_tiny_imagenet:
        #
        # ** WARNING: The `torch.load` functionality uses Python's pickling module that
        # may be used to perform arbitrary code execution during unpickling. Only load data that you
        # trust.
        #
        checkpoint = torch.load(str(fp32_pth_path), map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        acc1_fp32 = checkpoint["acc1"]
    else:
        best_acc1 = 0
        # Training loop.
        for epoch in range(0, epochs):
            # Run a single training epoch.
            train(train_loader, model, criterion, optimizer, epoch)
    
            # Evaluate on validation set.
            acc1 = validate(val_loader, model, criterion)
    
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
    
            if is_best:
                checkpoint = {"state_dict": model.state_dict(), "acc1": acc1}
                torch.save(checkpoint, fp32_pth_path)
        acc1_fp32 = best_acc1
        
    print(f"Accuracy of FP32 model: {acc1_fp32:.3f}")


.. parsed-literal::

    Accuracy of FP32 model: 55.520


Export the ``FP32`` model to OpenVINO™ Intermediate Representation, to
benchmark it in comparison with the ``INT8`` model.

.. code:: ipython3

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    ov_model = ov.convert_model(model, example_input=dummy_input, input=[1, 3, image_size, image_size])
    ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
    print(f"FP32 model was exported to {fp32_ir_path}.")


.. parsed-literal::

    FP32 model was exported to model/resnet18_fp32.xml.


Create and Initialize Quantization 
----------------------------------------------------------------------------

NNCF enables compression-aware training by integrating into regular
training pipelines. The framework is designed so that modifications to
your original training code are minor. Quantization is the simplest
scenario and requires only 3 modifications.

1. Configure NNCF parameters to specify compression

.. code:: ipython3

    nncf_config_dict = {
        "input_info": {"sample_size": [1, 3, image_size, image_size]},
        "log_dir": str(OUTPUT_DIR),  # The log directory for NNCF-specific logging outputs.
        "compression": {
            "algorithm": "quantization",  # Specify the algorithm here.
        },
    }
    nncf_config = NNCFConfig.from_dict(nncf_config_dict)

2. Provide a data loader to initialize the values of quantization ranges
   and determine which activation should be signed or unsigned from the
   collected statistics, using a given number of samples.

.. code:: ipython3

    nncf_config = register_default_init_args(nncf_config, train_loader)

3. Create a wrapped model ready for compression fine-tuning from a
   pre-trained ``FP32`` model and a configuration object.

.. code:: ipython3

    compression_ctrl, model = create_compressed_model(model, nncf_config)


.. parsed-literal::

    2023-10-31 00:15:43.733728: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-10-31 00:15:43.767038: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-31 00:15:44.314131: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Evaluate the new model on the validation set after initialization of
quantization. The accuracy should be close to the accuracy of the
floating-point ``FP32`` model for a simple case like the one being
demonstrated here.

.. code:: ipython3

    acc1 = validate(val_loader, model, criterion)
    print(f"Accuracy of initialized INT8 model: {acc1:.3f}")


.. parsed-literal::

    Test: [ 0/79]	Time 0.168 (0.168)	Loss 0.981 (0.981)	Acc@1 78.91 (78.91)	Acc@5 89.84 (89.84)
    Test: [10/79]	Time 0.156 (0.165)	Loss 1.905 (1.623)	Acc@1 46.88 (60.51)	Acc@5 82.03 (84.09)
    Test: [20/79]	Time 0.155 (0.160)	Loss 1.734 (1.692)	Acc@1 63.28 (58.63)	Acc@5 79.69 (83.04)
    Test: [30/79]	Time 0.154 (0.157)	Loss 2.282 (1.781)	Acc@1 50.00 (57.31)	Acc@5 69.53 (81.50)
    Test: [40/79]	Time 0.152 (0.156)	Loss 1.540 (1.825)	Acc@1 62.50 (55.83)	Acc@5 85.94 (80.96)
    Test: [50/79]	Time 0.152 (0.156)	Loss 1.972 (1.820)	Acc@1 57.03 (56.05)	Acc@5 75.00 (80.73)
    Test: [60/79]	Time 0.152 (0.156)	Loss 1.731 (1.846)	Acc@1 57.81 (55.51)	Acc@5 85.16 (80.21)
    Test: [70/79]	Time 0.154 (0.155)	Loss 2.412 (1.872)	Acc@1 47.66 (55.15)	Acc@5 71.88 (79.61)
     * Acc@1 55.540 Acc@5 80.200
    Accuracy of initialized INT8 model: 55.540


Fine-tune the Compressed Model 
------------------------------------------------------------------------

At this step, a regular fine-tuning process is applied to further
improve quantized model accuracy. Normally, several epochs of tuning are
required with a small learning rate, the same that is usually used at
the end of the training of the original model. No other changes in the
training pipeline are required. Here is a simple example.

.. code:: ipython3

    compression_lr = init_lr / 10
    optimizer = torch.optim.Adam(model.parameters(), lr=compression_lr)
    
    # Train for one epoch with NNCF.
    train(train_loader, model, criterion, optimizer, epoch=0)
    
    # Evaluate on validation set after Quantization-Aware Training (QAT case).
    acc1_int8 = validate(val_loader, model, criterion)
    
    print(f"Accuracy of tuned INT8 model: {acc1_int8:.3f}")
    print(f"Accuracy drop of tuned INT8 model over pre-trained FP32 model: {acc1_fp32 - acc1_int8:.3f}")


.. parsed-literal::

    Epoch:[0][  0/782]	Time 0.412 (0.412)	Loss 0.740 (0.740)	Acc@1 84.38 (84.38)	Acc@5 96.88 (96.88)
    Epoch:[0][ 50/782]	Time 0.383 (0.387)	Loss 0.911 (0.802)	Acc@1 78.91 (80.15)	Acc@5 92.97 (94.42)
    Epoch:[0][100/782]	Time 0.387 (0.388)	Loss 0.631 (0.798)	Acc@1 84.38 (80.24)	Acc@5 95.31 (94.38)
    Epoch:[0][150/782]	Time 0.381 (0.388)	Loss 0.836 (0.792)	Acc@1 80.47 (80.48)	Acc@5 94.53 (94.43)
    Epoch:[0][200/782]	Time 0.369 (0.386)	Loss 0.873 (0.780)	Acc@1 75.00 (80.65)	Acc@5 94.53 (94.59)
    Epoch:[0][250/782]	Time 0.385 (0.386)	Loss 0.735 (0.778)	Acc@1 84.38 (80.77)	Acc@5 95.31 (94.53)
    Epoch:[0][300/782]	Time 0.368 (0.386)	Loss 0.615 (0.771)	Acc@1 85.16 (80.99)	Acc@5 97.66 (94.58)
    Epoch:[0][350/782]	Time 0.392 (0.386)	Loss 0.599 (0.767)	Acc@1 85.16 (81.14)	Acc@5 95.31 (94.58)
    Epoch:[0][400/782]	Time 0.382 (0.386)	Loss 0.798 (0.765)	Acc@1 82.03 (81.21)	Acc@5 92.97 (94.56)
    Epoch:[0][450/782]	Time 0.377 (0.386)	Loss 0.630 (0.762)	Acc@1 85.16 (81.26)	Acc@5 96.88 (94.58)
    Epoch:[0][500/782]	Time 0.367 (0.386)	Loss 0.633 (0.757)	Acc@1 85.94 (81.45)	Acc@5 96.88 (94.63)
    Epoch:[0][550/782]	Time 0.406 (0.386)	Loss 0.749 (0.755)	Acc@1 82.03 (81.49)	Acc@5 92.97 (94.65)
    Epoch:[0][600/782]	Time 0.397 (0.385)	Loss 0.927 (0.753)	Acc@1 78.12 (81.53)	Acc@5 88.28 (94.67)
    Epoch:[0][650/782]	Time 0.392 (0.385)	Loss 0.645 (0.749)	Acc@1 84.38 (81.60)	Acc@5 95.31 (94.71)
    Epoch:[0][700/782]	Time 0.399 (0.386)	Loss 0.816 (0.749)	Acc@1 82.03 (81.62)	Acc@5 91.41 (94.69)
    Epoch:[0][750/782]	Time 0.404 (0.386)	Loss 0.811 (0.746)	Acc@1 80.47 (81.69)	Acc@5 94.53 (94.72)
    Test: [ 0/79]	Time 0.166 (0.166)	Loss 1.092 (1.092)	Acc@1 75.00 (75.00)	Acc@5 86.72 (86.72)
    Test: [10/79]	Time 0.150 (0.140)	Loss 1.917 (1.526)	Acc@1 48.44 (62.64)	Acc@5 78.12 (83.88)
    Test: [20/79]	Time 0.137 (0.138)	Loss 1.631 (1.602)	Acc@1 64.06 (60.68)	Acc@5 81.25 (83.71)
    Test: [30/79]	Time 0.136 (0.138)	Loss 2.037 (1.691)	Acc@1 57.81 (59.25)	Acc@5 71.09 (82.23)
    Test: [40/79]	Time 0.134 (0.137)	Loss 1.563 (1.743)	Acc@1 64.84 (58.02)	Acc@5 82.81 (81.33)
    Test: [50/79]	Time 0.137 (0.137)	Loss 1.926 (1.750)	Acc@1 52.34 (57.77)	Acc@5 76.56 (81.04)
    Test: [60/79]	Time 0.137 (0.137)	Loss 1.559 (1.781)	Acc@1 67.19 (57.24)	Acc@5 84.38 (80.58)
    Test: [70/79]	Time 0.137 (0.137)	Loss 2.353 (1.806)	Acc@1 46.88 (56.81)	Acc@5 72.66 (80.08)
     * Acc@1 57.320 Acc@5 80.730
    Accuracy of tuned INT8 model: 57.320
    Accuracy drop of tuned INT8 model over pre-trained FP32 model: -1.800


Export INT8 Model to OpenVINO IR 
--------------------------------------------------------------------------

.. code:: ipython3

    if not int8_ir_path.exists():
        warnings.filterwarnings("ignore", category=TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        # Export INT8 model to OpenVINO™ IR
        ov_model = ov.convert_model(model, example_input=dummy_input, input=[1, 3, image_size, image_size])
        ov.save_model(ov_model, int8_ir_path)
        print(f"INT8 Omodel exported to {int8_ir_path}.")


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.
    INT8 Omodel exported to model/resnet18_int8.xml.


Benchmark Model Performance by Computing Inference Time 
-------------------------------------------------------------------------------------------------

Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2023.0/openvino_inference_engine_tools_benchmark_tool_README.html>`__
- inference performance measurement tool in OpenVINO. By default,
Benchmark Tool runs inference for 60 seconds in asynchronous mode on
CPU. It returns inference speed as latency (milliseconds per image) and
throughput (frames per second) values.

   **NOTE**: This notebook runs ``benchmark_app`` for 15 seconds to give
   a quick indication of performance. For more accurate performance, it
   is recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications. Run
   ``benchmark_app -m model.xml -d CPU`` to benchmark async inference on
   CPU for one minute. Change CPU to GPU to benchmark on GPU. Run
   ``benchmark_app --help`` to see an overview of all command-line
   options.

.. code:: ipython3

    def parse_benchmark_output(benchmark_output):
        parsed_output = [line for line in benchmark_output if 'FPS' in line]
        print(*parsed_output, sep='\n')
    
    
    print('Benchmark FP32 model (IR)')
    benchmark_output = ! benchmark_app -m $fp32_ir_path -d CPU -api async -t 15
    parse_benchmark_output(benchmark_output)
    
    print('Benchmark INT8 model (IR)')
    benchmark_output = ! benchmark_app -m $int8_ir_path -d CPU -api async -t 15
    parse_benchmark_output(benchmark_output)


.. parsed-literal::

    Benchmark FP32 model (IR)
    [ INFO ] Throughput:   2952.65 FPS
    Benchmark INT8 model (IR)
    [ INFO ] Throughput:   11986.34 FPS


Show CPU Information for reference.

.. code:: ipython3

    ie = ov.Core()
    ie.get_property("CPU", "FULL_DEVICE_NAME")




.. parsed-literal::

    'Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz'


