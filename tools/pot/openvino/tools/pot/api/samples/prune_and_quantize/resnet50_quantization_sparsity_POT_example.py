import os
import argparse
import time
from pathlib import Path

import torch
from torchvision import models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import openvino.runtime as ov
from openvino.runtime import serialize
from openvino.tools import mo
from openvino.tools.pot import IEEngine, load_model, save_model, compress_model_weights, create_pipeline


# Create the parser
parser = argparse.ArgumentParser(description="Example script to demonstrate multiple data types with argparse.")

# Add arguments
parser.add_argument("-p", "--path", type=str, required=True, help="Path to imagenet folder")
parser.add_argument("-i", "--input_to_255", action='store_true', help="Scale input images between 0 and 255, otherwise 0 to 1")
parser.add_argument("-b", "--bgr", action='store_true', help="Channels to bgr, instead of rgb")
parser.add_argument("-s", "--sparsity", type=float, help="A float argument, dictating the sparsity level, zero means no sparsity applied")
parser.add_argument("-m", "--processing_in_model", action='store_true', help="Include preprocessing in model")
parser.add_argument("-v", "--validate_base_model", action='store_true', help="Validate the base model in addition to the quantized model")

# Parse arguments
args = parser.parse_args()

DATASET_DIR = Path(args.path)


def center_crop(img, output_size):
    crop_height, crop_width = (output_size, output_size)
    image_width, image_height = img.size
    img = img.crop(((image_width - crop_width)   // 2,
                    (image_height - crop_height) // 2,
                    (image_width + crop_width)   // 2,
                    (image_height + crop_height) // 2))
    return img


# Define the custom transform class
class RGBtoBGR:
    def __call__(self, image):
        # Flip the channels from RGB to BGR
        return image[[2, 1, 0], :, :]


def create_dataloader(batch_size, input_0_255, include_preprocess_in_model, rgb_to_bgr):
    """Creates train dataloader that is used for quantization initialization and validation dataloader for computing the model accuracy"""
    val_dir = DATASET_DIR / "val"

    transforms_list = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
                ]
    
    if input_0_255:
        transforms_list.append(ScaleByScalar())

    if not include_preprocess_in_model:
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        
        if input_0_255:
            mean *= mean
            std *= std
            
        transforms_list.append(transforms.Normalize(
            mean=mean, 
            std=std))
    
    if rgb_to_bgr:
        transforms_list.append(RGBtoBGR())
    
    val_dataset = ImageFolder(
        val_dir,
        transforms.Compose(
            transforms_list
        )
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return val_dataset, val_dataloader



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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


class ScaleByScalar:
    def __call__(self, img):
        return img * 255
    
    
def validate(val_loader_path, forward_fun, batch_size, include_preprocess_in_model=False, bgr_to_rgb=False, input_0_255=False):
    criterion = torch.nn.CrossEntropyLoss()
    
    _, val_loader = create_dataloader(batch_size, input_0_255, include_preprocess_in_model, bgr_to_rgb)
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            
            # compute output
            output = forward_fun(images)
            target = target.to(output.device)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 1000 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def generate_baseline_ov_model(ckpt_model, 
                               output_file_name, 
                               onnx_folder, 
                               openvino_folder, 
                               fp32_to_fp16=True, 
                               include_preprocess=True, 
                               input_0_255=False, 
                               rgb_to_bgr=False):
    
    model = models.resnet50(pretrained=True, progress=False)
    model.eval()

    if print(ckpt_model) is not None:
        print(ckpt_model)
        resuming_checkpoint = torch.load(f"{ckpt_model}", map_location='cpu')
        print(model.load_state_dict(resuming_checkpoint))
        
    model.eval()

    file_name = output_file_name
    os.makedirs(onnx_folder, exist_ok=True)
    
    x = (torch.rand([1,3,224,224]))
    with torch.no_grad():
        # Export the model
        torch.onnx.export(model,                     # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        f"./{onnx_folder}/{file_name}.onnx",        # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=12,          # the ONNX version to export the model to
                        input_names = ['images'],   # the model's input names
                        output_names = ['output']) # the model's output names
    
    mean_values = [0.485, 0.456, 0.406]
    scale_values = [0.229, 0.224, 0.225]
    
    if include_preprocess:
        if input_0_255:
            mean_values = [value*255 for value in mean_values]
            scale_values = [value*255 for value in scale_values]

        ov_model = mo.convert_model(f"./{onnx_folder}/{file_name}.onnx", 
                            compress_to_fp16=fp32_to_fp16, 
                            reverse_input_channels=rgb_to_bgr,
                            mean_values=mean_values,
                            scale_values=scale_values)
        
    else:
        ov_model = mo.convert_model(f"./{onnx_folder}/{file_name}.onnx", 
                                    compress_to_fp16=fp32_to_fp16, 
                                    reverse_input_channels=rgb_to_bgr)
    
    serialize(ov_model, f"{openvino_folder}/{output_file_name}.xml")
    print(f"Generated {openvino_folder}/{output_file_name}.xml")


def generate_pot_quantized_model(input_file_name,
                                 openvino_folder,
                                 output_file_name,
                                 include_preprocess_in_model=False, 
                                 input_0_255=True,
                                 rgb_to_bgr=False,
                                 sparsify=False):
                                     
    print(f"Quantizing {openvino_folder}/{input_file_name}.xml")
    
    model_config ={
        "model_name": "model",
        "model": f"{openvino_folder}/{input_file_name}.xml",
        "weights": f"{openvino_folder}/{input_file_name}.bin",
    }

    # Engine config.
    engine_config = {"device": "CPU"}
    algorithms = []
    
    if sparsify > 0:
        algorithms.append(
            {
                "name": "WeightSparsity",
                "params": {
                    "target_device": "CPU",
                    "sparsity_level": sparsify,
                    "stat_subset_size": 500,
                    "ignored_scope": "images", 
                }
            })
        
    algorithms.append(
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "CPU",
                "stat_subset_size": 500,
                "preset": "performance",
            },
        })

    # Step 1: Implement and create a user data loader.
    print("Step 1")
    # create_dataloader
    val_dataset, _ = create_dataloader(batch_size=1, 
                                                include_preprocess_in_model=include_preprocess_in_model, 
                                                input_0_255=input_0_255, 
                                                rgb_to_bgr=rgb_to_bgr)
    
    # Step 2: Load a model.
    print("Step 2")
    model = load_model(model_config=model_config)

    # Step 3: Initialize the engine for metric calculation and statistics collection.
    print("Step 3")
    engine = IEEngine(config=engine_config, data_loader=val_dataset)
    
    # Step 4: Create a pipeline of compression algorithms and run it.
    print("Step 4")
    pipeline = create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(model=model)

    # Step 5 (Optional): Compress model weights to quantized precision
    #                     to reduce the size of the final .bin file.
    print("Step 5")
    compress_model_weights(compressed_model)

    # Step 6: Save the compressed model to the desired path.
    # Set save_path to the directory where the model should be saved.
    print("Step 6")
    compressed_model_paths = save_model(
        model=compressed_model,
        save_path=openvino_folder,
        model_name=output_file_name,
    )

    print(f"Generated {openvino_folder}/{output_file_name}.xml")


def run_model_on_validation_set(file_name, 
                                openvino_folder,
                                validation_image_folder,
                                include_preprocess_in_model=False,
                                rgb_to_bgr=False,
                                input_0_255=False):
    core = ov.Core()

    model_xml = f"{openvino_folder}/{file_name}.xml"
    model_bin = f"{openvino_folder}/{file_name}.bin"

    print(model_xml)

    net = core.read_model(model=model_xml, weights=model_bin)
    exec_net = core.compile_model(net, "CPU")

    input_layer = exec_net.input(0)
    output_layer = exec_net.output(0)
    print(output_layer)

    infer_request = exec_net.create_infer_request()

    batch_size = 1 
    def forward_fun(images):
        infer_request.infer(inputs={input_layer.any_name: images})
        result = infer_request.get_output_tensor(output_layer.index).data
        return torch.tensor(result)
        
    validate(validation_image_folder, forward_fun, batch_size, include_preprocess_in_model, rgb_to_bgr, input_0_255)


if __name__ == "__main__":
    print("This script requires OpenVINO 2023.1")
    ckpt_model = None # Pass model weights path here, if wanted
    dataset_dir = args.path
    
    validation_image_folder = dataset_dir + "val"

    fp32_to_fp16 = False                                      #  if true, fp16 ov model output
    rgb_to_bgr = args.bgr                                     # if true, expect bgr inputs, else rgb
    include_preprocess_in_model = args.processing_in_model    # if true, preprocessing in model and not applied to inputs before model, else opposite
    sparsify = args.sparsity                                  # if greater than 0, sparsify to this amount
    input_0_255 = args.input_to_255                           # if true, scaled expects inputs scaled from 0-255, else 0-1

    temp_base_ov_generated = "base_model"
    onnx_save_folder_name = "onnx_2023_1"
    openvino_folder = "openvino_2023_1"

    fp32_to_fp16_tag = "fp16" if fp32_to_fp16 else "fp32"
    rgb_to_bgr_tag = "bgr" if rgb_to_bgr else "rgb"
    incpreproc_tag = "incpreproc_" if include_preprocess_in_model else ""
    sparsify_tag = f"sp{int(sparsify*100)}_" if sparsify else ""
    input_0_255_tag = "scale_0_255" if input_0_255 else "scale_0_1"

    quantized_file_name = f"resnet50v1_50_ov2023_1_{fp32_to_fp16_tag}_nchw_{sparsify_tag}{rgb_to_bgr_tag}_{incpreproc_tag}{input_0_255_tag}_cpu_defaultquantization"

    os.makedirs(onnx_save_folder_name, exist_ok=True)

    generate_baseline_ov_model(ckpt_model=ckpt_model, 
                               output_file_name=temp_base_ov_generated,
                               onnx_folder=onnx_save_folder_name,
                               openvino_folder=openvino_folder,
                               fp32_to_fp16=fp32_to_fp16, 
                               include_preprocess=include_preprocess_in_model,
                               input_0_255=input_0_255, 
                               rgb_to_bgr=rgb_to_bgr)
    
    if args.validate_base_model:
        run_model_on_validation_set(temp_base_ov_generated, 
                                    openvino_folder,
                                    validation_image_folder,
                                    include_preprocess_in_model=include_preprocess_in_model,
                                    rgb_to_bgr=rgb_to_bgr,
                                    input_0_255=input_0_255)

    generate_pot_quantized_model(input_file_name=temp_base_ov_generated,
                                 openvino_folder=openvino_folder,
                                 output_file_name=quantized_file_name,
                                 include_preprocess_in_model=include_preprocess_in_model, 
                                 input_0_255=input_0_255,
                                 rgb_to_bgr=rgb_to_bgr,
                                 sparsify=sparsify)

    run_model_on_validation_set(quantized_file_name, 
                                openvino_folder,
                                validation_image_folder,
                                include_preprocess_in_model=include_preprocess_in_model,
                                rgb_to_bgr=rgb_to_bgr,
                                input_0_255=input_0_255)