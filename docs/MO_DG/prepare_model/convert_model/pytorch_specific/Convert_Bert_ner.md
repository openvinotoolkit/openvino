# Convert PyTorch* BERT-NER to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_pytorch_specific_Convert_Bert_ner}

## Download and Convert the Model to ONNX*

To download a pre-trained model or train the model yourself, refer
to the [instruction](https://github.com/kamalkraj/BERT-NER/blob/dev/README.md) in the
BERT-NER model repository. The model with config files is stored in the `out_base` directory.

To convert the model to ONNX* format, create and run the script with the following content in the root
directory of the model repository. If you download the pre-trained model, you need
to download [`bert.py`](https://github.com/kamalkraj/BERT-NER/blob/dev/bert.py) to run the script.
The instruction was tested with the repository hash commit `e5be564156f194f1becb0d82aeaf6e762d9eb9ed`.

```python
import torch

from bert import Ner

ner = Ner("out_base")

input_ids, input_mask, segment_ids, valid_positions = ner.preprocess('Steve went to Paris')
input_ids = torch.tensor([input_ids], dtype=torch.long, device=ner.device)
input_mask = torch.tensor([input_mask], dtype=torch.long, device=ner.device)
segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=ner.device)
valid_ids = torch.tensor([valid_positions], dtype=torch.long, device=ner.device)

ner_model, tknizr, model_config = ner.load_model("out_base")

with torch.no_grad():
    logits = ner_model(input_ids, segment_ids, input_mask, valid_ids)
torch.onnx.export(ner_model,
                  (input_ids, segment_ids, input_mask, valid_ids),
                  "bert-ner.onnx",
                  input_names=['input_ids', 'segment_ids', 'input_mask', 'valid_ids'],
                  output_names=['output'],
                  dynamic_axes={
                      "input_ids": {0: "batch_size"},
                      "segment_ids": {0: "batch_size"},
                      "input_mask": {0: "batch_size"},
                      "valid_ids": {0: "batch_size"},
                      "output": {0: "output"}
                  },
                  opset_version=11,
                  )
```

The script generates ONNX* model file `bert-ner.onnx`.

## Convert ONNX* BERT-NER model to IR

```bash
python mo.py --input_model bert-ner.onnx --input "input_mask[1 128],segment_ids[1 128],input_ids[1 128]"
```

where `1` is `batch_size` and `128` is `sequence_length`.