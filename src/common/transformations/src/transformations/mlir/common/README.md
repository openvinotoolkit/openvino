# OV-nodes to MLIR-linalg converters

This folder contains converters and helper classes to convert certain OV operations to MLIR-linalg code.

## Structure

### `converters/`

Contains `.hpp` files, each representing a converter for a specific OV operation or a class of operations (matmul, reduction, binary-elementwise). A converter must implement the following interface — it takes a conversion context and an OV node, produces a sequence of MLIR operations, and returns the final op:

```c++
mlir::Operation* operator()(ov::mlir::ConversionContext& context, std::shared_ptr<ov::Node> node)
```

### `conversion_context`

`ov::mlir::ConversionContext` is a class that provides converters with:
- `mlir::Context`
- `mlir::OpBuilder`
- Mapping between `ov::Node` inputs and MLIR tensors
- Mapping between dynamic-dimension symbols and their MLIR values

```c++
class ConversionContext {
public:
  using getInputsFn = std::function<SmallVector<mlir::Value>(NodePtr)>;
  using getDimValueFn = std::function<Value(const Dimension&)>;

  getInputsFn getInputs;
  getDimValueFn getDimValue;

  ConversionContext(
    mlir::MLIRContext* context, mlir::OpBuilder* block_builder,
    getInputsFn getInputs, getDimValueFn getDimValue
  );
  ...
```

A higher-level class responsible for whole-graph conversion is expected to encapsulate a `ConversionContext` instance and provide the necessary callbacks to resolve mapped inputs.

Example:

```c++
class OvGraphImporter {
  ov::mlir::ConversionContext _ctx;
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

Contains general utility functions (creating MLIR locations/constants, shape/type importing) used by the converters.
