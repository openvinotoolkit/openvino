# Subgraph extraction

Extracts a subgraph from an `ov::Model` as a standalone `ov::Model`, without modifying the original model.

Useful for inspecting, serializing, or testing a specific portion of a larger model during development or debugging. This is especially valuable when working with large models where full compilation or inference takes a significant amount of time, or where the serialized IR is too large to read or render in graph visualization tools — extracting only the relevant subgraph lets you iterate on it in isolation instead of processing the entire model on each run.

Once extracted, the subgraph can be serialized to IR via `ov::pass::Serialize` and reloaded independently, so subsequent investigation or reproduction work can be done on the smaller model without access to the original.

## Example

```cpp
#include "transformations/utils/extract_subgraph.hpp"

// Extract the subgraph between two named nodes
const std::multimap<std::string, size_t> inputs  = {{"MatMul_0", 0}};
const std::multimap<std::string, size_t> outputs = {{"Softmax_0", 0}};

auto subgraph = ov::op::util::extract_subgraph(model, inputs, outputs);

// The original model is unchanged; subgraph is an independent ov::Model.
// Serialize it to IR so further investigation can be done on the isolated subgraph
// without rerunning the full model pipeline.
ov::pass::Serialize("subgraph.xml", "subgraph.bin").run_on_model(subgraph);
```

## Behavior

- The original model is **not modified**.
- The extracted model is a deep clone — changes to it do not affect the source model.
- Subgraph `Parameter` nodes inherit the element type and partial shape of their corresponding input ports.
- Node order within the extracted model follows topological order.
