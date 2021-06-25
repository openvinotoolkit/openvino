# Convert ONNX* RNNT to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_RNNT}

This instruction covers conversion of RNN-t model from [MLCommons](https://github.com/mlcommons) repository. Before
converting into IR PyTorch model need to be exported into ONNX. In order to that follow the steps below.

**Step 1**. Clone RNN-t PyTorch implementation from MLCommons repository. In order to pull only RNN-t model 
without full repository make a shallow clone. If you alreade have a full repository skip this and go to **Step 2**:
```bash
git clone -n https://github.com/mlcommons/inference rnnt_for_openvino --depth 1
cd rnnt_for_openvino
git checkout HEAD speech_recognition/rnnt 
```

**Step 2**. If you already have a full clone of MLCommons inference repository create a folder for 
pre-trained PyTorch model and where conversion into IR will take place. Skip this step if you did a shallow clone.
Also further you will need to specify path to your full clone:
```bash
mkdir rnnt_for_openvino 
cd rnnt_for_openvino
```

**Step 3**. Download pre-trained weights for PyTorch implementation from https://zenodo.org/record/3662521#.YG21DugzZaQ:
For UNIX* like systems you can use wget
```bash
wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt
```
The link was taken from `setup.sh` in the `speech_recoginitin/rnnt` subfolder. You will get exactly the same weights as 
if you have done steps from https://github.com/mlcommons/inference/tree/master/speech_recognition/rnnt

**Step 4**. Install required python packages:
```bash
pip3 install torch toml
```

**Step 5**. Export RNN-t model into ONNX with the script below. Copy it into a file with a name `export_rnnt_to_onnx.py` 
and run in the current directory `rnnt_for_openvino`:

> **note**: If you already had a full clone of MLComons inferene repository then you need to
> specify `mlcommons_inference_path` variable.

```python
import toml
import torch
import sys

def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict

mlcommons_inference_path = './'  # specify relative path for MLCommons inferene
checkpoint_path = 'DistributedDataParallel_1576581068.9962234-epoch-100.pt'
config_toml = 'speech_recognition/rnnt/pytorch/configs/rnnt.toml'
config = toml.load(config_toml)
rnnt_vocab = config['labels']['labels']
sys.path.insert(0,  mlcommons_inference_path + 'speech_recognition/rnnt/pytorch')

from model_separable_rnnt import RNNT
model = RNNT(config['rnnt'], len(rnnt_vocab) + 1, feature_config=config['input_eval'])
model.load_state_dict(load_and_migrate_checkpoint(checkpoint_path))

seq_length, batch_size, feature_length = 157, 1, 240
inp = torch.randn([seq_length, batch_size, feature_length])
feature_length = torch.LongTensor([seq_length])
x_padded, x_lens = model.encoder(inp, feature_length)
torch.onnx.export(model.encoder, (inp, feature_length), "rnnt_encoder.onnx", opset_version=12,
                  input_names=['input.1', '1'], dynamic_axes={'input.1': {0: 'seq_len', 1: 'batch'}})

symbol = torch.LongTensor([[20]])
hidden = torch.randn([2, batch_size, 320]), torch.randn([2, batch_size, 320])
g, hidden = model.prediction.forward(symbol, hidden)
torch.onnx.export(model.prediction, (symbol, hidden), "rnnt_prediction.onnx", opset_version=12,
                  input_names=['input.1', '1', '2'],
                  dynamic_axes={'input.1': {0: 'batch'}, '1': {1: 'batch'}, '2': {1: 'batch'}})

f = torch.randn([batch_size, 1, 1024])
model.joint.forward(f, g)
torch.onnx.export(model.joint, (f, g), "rnnt_joint.onnx", opset_version=12,
                  input_names=['0', '1'], dynamic_axes={'0': {0: 'batch'}, '1': {0: 'batch'}})
```

```bash
python3 export_rnnt_to_onnx.py
```

After that files rnnt_encoder.onnx, rnnt_prediction.onnx, and rnnt_joint.onnx will be saved in the current directory. 

**Step 6**. Run the conversion command:

```bash
python3 {path_to_openvino}/mo.py --input_model rnnt_encoder.onnx --input "input.1[157 1 240],1->157"
python3 {path_to_openvino}/mo.py --input_model rnnt_prediction.onnx --input "input.1[1 1],1[2 1 320],2[2 1 320]"
python3 {path_to_openvino}/mo.py --input_model rnnt_joint.onnx --input "0[1 1 1024],1[1 1 320]"
```
Please note that hardcoded value for sequence length = 157 was taken from the MLCommons but conversion to IR preserves 
network reshape-ability, this means you can change input shapes manually to any value either during conversion or 
during inference.