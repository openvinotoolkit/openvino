# Filter Pruning of Convolutional Models {#filter_pruning}

# Introduction
Filter pruning is an advanced optimization method which allows reducing computational complexity of the model by removing redundant or unimportant filters from convolutional operations of the model. This removal is done in two steps: 
1. Unimportant filters are zeroed out by the NNCF optimization with fine-tuning.
2. Zero filters are removed from the model during the export to OpenVINO&trade; Intermediate Representation (IR).

Filter pruning method from the NNCF can be used stand-alone but we usually recommend to stack it with 8-bit quantization for two reasons. First, 8-bit quantization is the best method in term of achieving the highest accuracy-performance trade-offs so stacking it with filter pruning can give even better performance results. Second, applying quantization along with filter pruning does not hurt accuracy a lot since filter pruning removes a noisy filters from the model which narrows down ranges for weights and activations and helps 

> **NOTE**: 

## Applying filter pruning with fine-tuning

## Removing zero filters from the pruned model