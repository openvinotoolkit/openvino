# Inference precision debug

When developing/debugging accuracy issue related to inference precision feature, you can define following environment variables to control the type & number of nodes for which the specified inference precision is enforced:

- `OV_CPU_INFER_PRC_POS_PATTERN` : positive regex pattern to filter node type & orgname.
- `OV_CPU_INFER_PRC_NEG_PATTERN` : negative regex pattern to filter after pos-pattern was matched.
- `OV_CPU_INFER_PRC_CNT`   : total number of nodes allowed to enforce;

Just adjust these two settings before each run until accuracy issue happens/disappears, from the log we can spot the first node introduces issue when enabled.

When any of these pattern env-var was set to a valid raw C++ regular expression string, console output will show the nodes finally passed the filter.
so we suggest first `export OV_CPU_INFER_PRC_POS_PATTERN=".*"` to see the full list of nodes, and then add filters one by one.

for example, for a LLM model we set pos-pattern as ".*" first and following log was generated on console:

```bash
OV_CPU_INFER_PRC_POS_PATTERN=".*"
infer precision enforced Types: Broadcast,Concatenation,Convert,Eltwise,FullyConnected,Gather,Math,Range,Reduce,Reshape,RoPE,ScaledDotProductAttention,ShapeOf,Transpose,  total number of nodes: 1918
    ...
Eltwise : 
	Eltwise@__module.model.layers.0.input_layernorm/aten::pow/Power
	Eltwise@__module.model.layers.0.input_layernorm/aten::add/Add
	Eltwise@__module.model.layers.0.input_layernorm/aten::rsqrt/Sqrt
	Eltwise@__module.model.layers.0.input_layernorm/aten::rsqrt/Divide
	Eltwise@__module.model.layers.0.input_layernorm/aten::mul/Multiply
	Eltwise@__module.model.layers.0.input_layernorm/aten::mul/Multiply_1
    ...
FullyConnected : 
	FullyConnected@__module.model.layers.0.self_attn.q_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.0.self_attn.k_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.0.self_attn.v_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.0.self_attn.o_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.0.mlp.gate_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.0.mlp.up_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.0.mlp.down_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.1.self_attn.q_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.1.self_attn.k_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.1.self_attn.v_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.1.self_attn.o_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.1.mlp.gate_proj/aten::linear/MatMul
	FullyConnected@__module.model.layers.1.mlp.up_proj/aten::linear/MatMul
    ...
```

then we save the log and replace ".*" with the pattern that we selectivly enables to see whether accuracy issue is fixed, for example
suppose we only enables `input_layernorm` for all layers we set pos-pattern as `"__module\.model\.layers\.(\d+)\.input_layernorm/"`, notice
the escape for regex reserved special char . and sub-pattern (\d+) for matching all layer indices.

if the accuracy issue is gone, we can add more nodes to this pattern, for example we can add `self_attn.q_proj` from all layers, the our pattern
becomes `"__module\.model\.layers\.(\d+)\.input_layernorm/|__module\.model\.layers\.(\d+)\.self_attn\.q_proj/`, then run test again to see whether
the accuracy issue appears. if not, repeat above procedure.

we can use neg-pattern to elimite nodes from being enforced bf16/f16, for example, set the pattern above to neg-pattern so only matched nodes
are eliminated, and check if accuracy is gone, if not, add more nodes into the pattern.

notice that the actual string being matched with pattern is in the format of `NodeType@orginialLayers`, so by using `"^FullyConnected@"` as the pattern we can
select all nodes with specific type - `FullyConnected` instead of with specific original layer name.
