// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu {
namespace mlir {
class MLIRGpuProgram;
}  // namespace mlir
namespace op {

using OVOutputTypes = std::vector<std::tuple<ov::element::Type, ov::PartialShape>>;

// Maps [output index][dimension index] -> [input index][dimension index] to
// infer shapes for the entire subgraph.
using DimensionsMap = std::vector<std::vector<std::tuple<size_t, size_t>>>;

// Physical dimension order the compiled program expects for some inputs (key == input_index).
// Inputs without an entry are plain (row-major). E.g.:
//   input_orders[1] = {0, 2, 1};  // input 1 is a logical [E, N, G] tensor stored in memory as [E, G, N]
using InputOrders = std::map<size_t, std::vector<size_t>>;

class MLIROp : public ov::op::Op {
    std::shared_ptr<mlir::MLIRGpuProgram> program;
    OVOutputTypes output_types;
    DimensionsMap dimensions_map;
    InputOrders input_orders;

public:
    OPENVINO_OP("MLIROp");

    MLIROp() = default;

    MLIROp(const ov::OutputVector& args,
           std::shared_ptr<mlir::MLIRGpuProgram> program,
           OVOutputTypes output_types,
           DimensionsMap dimensions_map,
           InputOrders input_orders = {});

    const std::shared_ptr<mlir::MLIRGpuProgram>& get_program() const {
        return program;
    }

    const InputOrders& get_input_orders() const {
        return input_orders;
    }

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    std::vector<ov::PartialShape> shape_infer(const std::vector<ov::PartialShape>& input_shapes) const;
};

}  // namespace op
}  // namespace ov::intel_gpu
