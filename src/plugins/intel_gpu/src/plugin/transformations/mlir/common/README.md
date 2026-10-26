# OV-nodes to MLIR-linalg converters

This folder contains converters and helper classes to convert certain OV operations to MLIR-linalg code.

The module is intentionally self-contained: it depends on nothing but OV core (`ov::Node`, `ov::PartialShape`,
symbols) and MLIR itself (linalg / tensor / arith dialects). It knows nothing about the GPU plugin or about the
GPU MLIR execution path — everything specific to that path (deciding which subgraphs are eligible, partitioning
the model, invoking the Graph Compiler, passing OpenCL runtime handles) lives one level up, in
[`transformations/mlir/`](../README.md).

Because of that, this folder can be potentially lifted to a common / public OV location and reused by another component or
plugin that needs to lower `ov::Model` fragments to linalg. The only thing such a consumer has to provide is a
graph-level driver that owns a `ConversionContext` and resolves node inputs and dynamic dimensions for it — see
[Whole-graph conversion](#whole-graph-conversion).

## Structure

### `converters/`

Contains `.hpp` files, each representing a converter for a specific OV operation or a class of operations
(matmul, reduction, binary-elementwise). A converter must implement the following interface — it takes a
conversion context and an OV node, produces a sequence of MLIR operations, and returns the final op:

```c++
mlir::Operation* operator()(ov::intel_gpu::mlir::ConversionContext& context, const std::shared_ptr<ov::Node>& node)
```

In the GPU plugin this contract is spelled as the `GraphConverter::Convertor` alias, see
[`graph_converter.hpp`](../graph_converter.hpp).

### `conversion_context`

`ov::intel_gpu::mlir::ConversionContext` is a class that provides converters with:
- `mlir::Context`
- `mlir::OpBuilder`
- Mapping between `ov::Node` inputs and MLIR tensors
- Mapping between dynamic-dimension symbols and their MLIR values

For the exact declaration (constructor and the `getInputs` / `getDimValue` callbacks it expects) see
[`conversion_context.hpp`](conversion_context.hpp).

### Whole-graph conversion

A higher-level class responsible for whole-graph conversion is expected to encapsulate a `ConversionContext`
instance and provide the necessary callbacks to resolve mapped inputs.

The snippet below is **only an illustration** — a sketch of how these converters could be driven from a
component that reuses this module, not code that exists in the repository. For the real implementation used by
the GPU plugin see `GraphConverter` in [`graph_converter.hpp`](../graph_converter.hpp) /
[`graph_converter.cpp`](../graph_converter.cpp).

```c++
class OvGraphImporter {
  ov::intel_gpu::mlir::ConversionContext _ctx;
  ov::Model _model;
  mlir::ModuleOp _module;
  ...
  using Converter = std::function<Operation*(ConversionContext&, NodePtr)>;

public:
  OvGraphImporter(mlir::Context ctx, mlir::OpBuilder builder, ov::Model model):
    _ctx(
      ctx, builder,
      [this](Node node){this->getInputs(node)},
      [this](Dimension dim) {this->getDimension(dim)}
    )
  {
    // build main-func-op in _module;
    // build ov-inputs -> mlir-tensors mapping
  }
  
  void import() {
    for (auto op : model.get_ordered_ops()) {
      // find a suitable converter from 'mlir/common/converters/*.hpp'
      Converter converter = findMatchingConverter(op);
      auto mlirOp = converter(_ctx, op);
      addToOutputsMapping(mlirOp, op);
    }
  }
  ...
}
```

### `convert_common`

Contains general utility functions (creating MLIR locations/constants, shape/type importing) used by the
converters.
