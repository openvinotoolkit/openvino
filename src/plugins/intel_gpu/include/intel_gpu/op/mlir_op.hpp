// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::intel_gpu {
namespace mlir {
class MLIREvaluateBase;
}  // namespace mlir
namespace op {

using OVOutputTypes = std::vector<std::tuple<ov::element::Type, ov::PartialShape>>;

// Maps [output index][dimension index] -> [input index][dimension index] to
// infer shapes for the entire subgraph.
using DimensionsMap = std::vector<std::vector<std::tuple<size_t, size_t>>>;

class MLIROp : public ov::op::Op {
    std::shared_ptr<mlir::MLIREvaluateBase> engine;
    OVOutputTypes output_types;
    DimensionsMap dimensions_map;

public:
    OPENVINO_OP("MLIROp");

    MLIROp() = default;

    MLIROp(const ov::OutputVector& args,
           std::shared_ptr<mlir::MLIREvaluateBase> engine,
           const OVOutputTypes& output_types,
           const DimensionsMap& dimensions_map);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs,
                  const ov::EvaluationContext& evaluationContext) const override;
    bool has_evaluate() const override;
    std::vector<ov::PartialShape> shape_infer(const std::vector<ov::PartialShape>& input_shapes) const;
};

}  // namespace op
}  // namespace ov::intel_gpu
