from transformers import AutoTokenizer, AutoModel
from openvino import convert_model

tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
model = AutoModel.from_pretrained("google/fnet-base", torchscript=True)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
inputs = dict(encoded_input)

m = convert_model(model, example_input=inputs)
