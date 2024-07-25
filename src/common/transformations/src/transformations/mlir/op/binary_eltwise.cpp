// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include <openvino/op/relu.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "binary_eltwise.hpp"

namespace {

using namespace ov;
using namespace ov::mlir;
using ::mlir::ValueRange;

class ConvertBinaryEltwise {

    BinaryEltwisePatternBase::Builder m_op_builder;

public:

    ConvertBinaryEltwise(BinaryEltwisePatternBase::Builder op_builder) : m_op_builder(op_builder) {}

    void operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto inputs = context.getInputs(node);
        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_output_shape = node->get_output_partial_shape(0);
        auto outType = importTensor(context.context, ov_output_shape, ov_output_element_type);
        const int output_rank = ov_output_shape.rank().get_length();

        SmallVector<Value> dynamic_dimensions = context.get_dynamic_dimension_values(ov_output_shape);

        SmallVector<Value> broadcasted_inputs;
        for(size_t i = 0; i < inputs.size(); ++i) {
            auto [collapse_groups, dimensions] = broadcast_dimensions(node->get_input_partial_shape(i), ov_output_shape);
            if(!dimensions.empty()) {
                // FIXME: Find a way to avoid dimension squeezing before applying linalg.broadcast
                // Step 1: Squeeze input shape to eliminate broadcasted dimensions
                auto squeezed = builder.create<tensor::CollapseShapeOp>(loc, inputs[i], collapse_groups);
                // Step 2: Broadcast squeezed shape to the target shape
                auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamic_dimensions);
                auto op = builder.create<linalg::BroadcastOp>(loc, squeezed, empty, dimensions);
                broadcasted_inputs.push_back(op.getResult()[0]);
            } else {
                broadcasted_inputs.push_back(inputs[i]);
            }
        }

        auto empty = builder.create<tensor::EmptyOp>(loc, outType, dynamic_dimensions);
        auto op = m_op_builder(builder, loc, ValueRange(broadcasted_inputs), ValueRange{empty});
        context.addOutputs(node, op);
    }
};

}  // namespace

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;

BinaryEltwisePatternBase::BinaryEltwisePatternBase(NodeTypeInfo wrapped_type, Builder op_builder, const std::set<element::Type>& element_types)
    : MarkPattern(
        std::make_shared<pass::pattern::op::WrapType>(
            wrapped_type,
            [element_types](const Output<Node>& output) {
                if(!element_types.empty() && !element_types.count(output.get_element_type())) {
                    return false;
                }
                auto node = output.get_node_shared_ptr();
                for(const auto& input: node->inputs()) {
                    if(!statically_broadcastable(input.get_partial_shape(), output.get_partial_shape())) {
                        return false;
                    }
                }
                return true;
            },
            OutputVector{any_input(), any_input()}),
        ConvertBinaryEltwise(op_builder))
    {}

}  // namespace mlir
}  // namespace ov
