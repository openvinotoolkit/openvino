# Inference precision debug

When developing/debugging accuracy issue related to inference precision feature, you can define following environment variables to control the type & number of nodes for which the specified inference precision is enforced:

 - `OV_CPU_INFER_PRC_TYPES` : comma separated list of node types for which infer-precision is enforced. use node type with prefix `-` as negative pattern to disable particular node types from enforcement;
 - `OV_CPU_INFER_PRC_CNT`   : total number of nodes allowed to enforce;

Just adjust these two settings before each run until accuracy issue happens/disappears, from the log we can spot the first node introduces issue when enabled.
