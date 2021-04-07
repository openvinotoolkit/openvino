# Convert ONNX* RNNT to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_RNNT}

This instruction convers conversion of RNN-t model from [MLCommons](https://github.com/mlcommons) repository. 
If you already have a downloaded MLCommons repository you can go right away to **Step 5**.

**Step 1**. Clone the MLCommons repository for inference:
```bash
git clone https://github.com/mlcommons/inference
```

**Step 2**. create a folder for storing pre-trained model and onnx files:
```bash
mkdir rnnt_for_openvino 
cd rnnt_for_openvino
```

**Step 3**. Download pre-trained weights for PyTorch implementation from https://zenodo.org/record/3662521#.YG21DugzZaQ:
For UNIX* like systems you can use wget
```bash
wget https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt
```

**Step 4**. Install required python packages:
```bash
pip install torch toml
```

**Step 5**. Save the following script in the folder created at the **Step 2**:
Before running the script specify path where you cloned MLCommons repository so that script could find `config_toml`. 

```python
import argparse
import toml
import torch
from model_separable_rnnt import RNNT


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


checkpoint_path = 'DistributedDataParallel_1576581068.9962234-epoch-100.pt'
config_toml = '{path_to_mlcommons}/inference/speech_recognition/rnnt/pytorch/configs/rnnt.toml'
config = toml.load(config_toml)
rnnt_vocab = config['labels']['labels']

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

Also you need to add path to `.../rnnt/pytorch` subfolder from the cloned MLCommons to your `PYTHONPATH`. 
For UNIX* like systems you can simply run:
```bash
PYTHONPATH={path_to_mlcommons}/inference/speech_recognition/rnnt/pytorch:$PYTHONPATH python export_rnn_to_oonnx.py
```

After that onnx files for encoder, prediction, and joint parts of the network will be generated. Please note that 
value for sequence length was taken from the MLCommons but you can change it manually to any value. Regardless 
of hardcoded sequence length it does not break reshape-ability of the final IR after conversion.
 
**Step 6**. Run the conversion command:

```bash
python3 {path_to_openvino}/mo.py --input_model rnnt_encoder.onnx --input "input.1[157 1 240],1->157"
python3 {path_to_openvino}/mo.py --input_model rnnt_prediction.onnx --input "input.1[1 1],1[2 1 320],2[2 1 320]"
python3 {path_to_openvino}/mo.py --input_model rnnt_joint.onnx --input "0[1 1 1024],1[1 1 320]"
```
