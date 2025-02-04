Quantization-Sparsity Aware Training with NNCF, using PyTorch framework
=======================================================================

This notebook is based on `ImageNet training in
PyTorch <https://github.com/pytorch/examples/blob/master/imagenet/main.py>`__.

The goal of this notebook is to demonstrate how to use the Neural
Network Compression Framework
`NNCF <https://github.com/openvinotoolkit/nncf>`__ 8-bit quantization to
optimize a PyTorch model for inference with OpenVINO Toolkit. The
optimization process contains the following steps:

-  Transforming the original dense ``FP32`` model to sparse ``INT8``
-  Using fine-tuning to improve the accuracy.
-  Exporting optimized and original models to OpenVINO IR
-  Measuring and comparing the performance of models.

For more advanced usage, refer to these
`examples <https://github.com/openvinotoolkit/nncf/tree/develop/examples>`__.

This tutorial uses the ResNet-50 model with the ImageNet dataset. The
dataset must be downloaded separately. To see ResNet models, visit
`PyTorch hub <https://pytorch.org/hub/pytorch_vision_resnet/>`__.


**Table of contents:**


-  `Imports and Settings <#imports-and-settings>`__
-  `Pre-train Floating-Point Model <#pre-train-floating-point-model>`__

   -  `Train Function <#train-function>`__
   -  `Validate Function <#validate-function>`__
   -  `Helpers <#helpers>`__
   -  `Get a Pre-trained FP32 Model <#get-a-pre-trained-fp32-model>`__

-  `Create and Initialize
   Quantization <#create-and-initialize-quantization>`__
-  `Fine-tune the Compressed Model <#fine-tune-the-compressed-model>`__
-  `Export INT8 Sparse Model to OpenVINO
   IR <#export-int8-model-to-openvino-ir>`__
-  `Benchmark Model Performance by Computing Inference
   Time <#benchmark-model-performance-by-computing-inference-time>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu  "openvino>=2024.0.0" "torch" "torchvision" "tqdm"
    %pip install -q "nncf>=2.9.0"

Imports and Settings
--------------------



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

    import time
    import warnings  # To disable warnings on export model
    from pathlib import Path
    
    import torch
    
    import torch.nn as nn
    import torch.nn.parallel
    import torch.optim
    import torch.utils.data
    import torch.utils.data.distributed
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torchvision.models as models
    
    import openvino as ov
    from torch.jit import TracerWarning
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    
    MODEL_DIR = Path("model")
    OUTPUT_DIR = Path("output")
    # DATA_DIR = Path("...")  # Insert path to folder containing imagenet folder
    # DATASET_DIR = DATA_DIR / "imagenet"

.. code:: ipython3

    # Fetch `notebook_utils` module
    import zipfile
    import requests
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, device_widget
    
    DATA_DIR = Path("data")
    
    
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
        val_data_dir = dataset_dir / "val"
        val_annotations_file = val_data_dir / "val_annotations.txt"
        with open(val_annotations_file, "r") as f:
            val_annotation_data = map(lambda line: line.split("\t")[:2], f.readlines())
        val_images_dir = val_data_dir / "images"
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
    
    BASE_MODEL_NAME = "resnet18"
    image_size = 64
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    # Paths where PyTorch and OpenVINO IR models will be stored.
    fp32_pth_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".pth")
    fp32_ir_path = fp32_pth_path.with_suffix(".xml")
    int8_sparse_ir_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_int8_sparse")).with_suffix(".xml")
    
    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry
    
    collect_telemetry("pytorch-quantization-sparsity-aware-training.ipynb")

Train Function
~~~~~~~~~~~~~~



.. code:: ipython3

    def train(train_loader, model, compression_ctrl, criterion, optimizer, epoch):
        batch_time = AverageMeter("Time", ":3.3f")
        losses = AverageMeter("Loss", ":2.3f")
        top1 = AverageMeter("Acc@1", ":2.2f")
        top5 = AverageMeter("Acc@5", ":2.2f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, losses, top1, top5],
            prefix="Epoch:[{}]".format(epoch),
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
            compression_ctrl.scheduler.step()

Validate Function
~~~~~~~~~~~~~~~~~



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
~~~~~~~



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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



А pre-trained floating-point model is a prerequisite for quantization.
It can be obtained by tuning from scratch with the code below.

.. code:: ipython3

    num_classes = 1000
    init_lr = 1e-4
    batch_size = 128
    epochs = 20
    
    # model = models.resnet50(pretrained=True)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
    model.to(device)
    
    
    # Data loading code.
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
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
                transforms.Resize([256, 256]),
                transforms.CenterCrop([image_size, image_size]),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        sampler=None,
    )
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    # Define loss function (criterion) and optimizer.
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

Export the ``FP32`` model to OpenVINO™ Intermediate Representation, to
benchmark it in comparison with the ``INT8`` model.

.. code:: ipython3

    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    ov_model = ov.convert_model(model, example_input=dummy_input, input=[1, 3, image_size, image_size])
    ov.save_model(ov_model, fp32_ir_path, compress_to_fp16=False)
    print(f"FP32 model was exported to {fp32_ir_path}.")

Create and Initialize Quantization and Sparsity Training
--------------------------------------------------------



NNCF enables compression-aware training by integrating into regular
training pipelines. The framework is designed so that modifications to
your original training code are minor.

.. code:: ipython3

    from nncf import NNCFConfig
    from nncf.torch import create_compressed_model, register_default_init_args
    
    # load
    nncf_config = NNCFConfig.from_json("config.json")
    nncf_config = register_default_init_args(nncf_config, train_loader)
    
    # Creating a compressed model
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
    compression_ctrl.scheduler.epoch_step()

Validate Compressed Model

Evaluate the new model on the validation set after initialization of
quantization and sparsity.

.. code:: ipython3

    acc1 = validate(val_loader, compressed_model, criterion)
    print(f"Accuracy of initialized sparse INT8 model: {acc1:.3f}")

Fine-tune the Compressed Model
------------------------------



At this step, a regular fine-tuning process is applied to further
improve quantized model accuracy. Normally, several epochs of tuning are
required with a small learning rate, the same that is usually used at
the end of the training of the original model. No other changes in the
training pipeline are required. Here is a simple example.

.. code:: ipython3

    compression_lr = init_lr / 10
    optimizer = torch.optim.Adam(compressed_model.parameters(), lr=compression_lr)
    nr_epochs = 10
    # Train for one epoch with NNCF.
    print("Training")
    for epoch in range(nr_epochs):
        compression_ctrl.scheduler.epoch_step()
        train(train_loader, compressed_model, compression_ctrl, criterion, optimizer, epoch=epoch)
    
    # Evaluate on validation set after Quantization-Aware Training (QAT case).
    print("Validating")
    acc1_int8_sparse = validate(val_loader, compressed_model, criterion)
    
    print(f"Accuracy of tuned INT8 sparse model: {acc1_int8_sparse:.3f}")
    print(f"Accuracy drop of tuned INT8 sparse model over pre-trained FP32 model: {acc1 - acc1_int8_sparse:.3f}")

Export INT8 Sparse Model to OpenVINO IR
---------------------------------------



.. code:: ipython3

    warnings.filterwarnings("ignore", category=TracerWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    # Export INT8 model to OpenVINO™ IR
    ov_model = ov.convert_model(compressed_model, example_input=dummy_input, input=[1, 3, image_size, image_size])
    ov.save_model(ov_model, int8_sparse_ir_path)
    print(f"INT8 sparse model exported to {int8_sparse_ir_path}.")

Benchmark Model Performance by Computing Inference Time
-------------------------------------------------------



Finally, measure the inference performance of the ``FP32`` and ``INT8``
models, using `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
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

    # Initialize OpenVINO runtime
    core = ov.Core()
    device = device_widget()
    
    device

.. code:: ipython3

    def parse_benchmark_output(benchmark_output):
        parsed_output = [line for line in benchmark_output if "FPS" in line]
        print(*parsed_output, sep="\n")
    
    
    print("Benchmark FP32 model (IR)")
    benchmark_output = ! benchmark_app -m $fp32_ir_path -d $device.value -api async -t 15
    parse_benchmark_output(benchmark_output)
    
    print("Benchmark INT8 sparse model (IR)")
    benchmark_output = ! benchmark_app -m $int8_ir_path -d $device.value -api async -t 15
    parse_benchmark_output(benchmark_output)

Show Device Information for reference.

.. code:: ipython3

    import openvino.properties as props
    
    
    core.get_property(device.value, props.device.full_name)
