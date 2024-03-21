// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/avg_pool.hpp" // Changed to AvgPool
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_avg_pool_base(const NodeContext& context, const Output& input,
                                     const Shape& kernel, const Strides& strides = Strides{},
                                     const Shape& pads = Shape{}, const Strides& dilations = Strides{},
                                     RoundingType rounding_type = RoundingType::FLOOR,
                                     int pooling_dims = 2) {
    num_inputs_check(context, 3, 6); // Ensure correct number of inputs
    auto kernel_shape = context.const_input<Shape>(1); // Extract kernel shape
    Strides strides;
    if (!context.input_is_none(2)) {
        strides = context.const_input<Strides>(2); // Extract strides if provided
    }
    const bool use_kernel = context.input_is_none(2) || (strides.size() == 0);
    if (use_kernel) {
        // In case strides are not provided, default is kernel
        strides = kernel;
    }
    Shape pads;
    if (context.input_is_none(3)) {
        pads = Shape(kernel.size(), 0); // Default padding if not provided
    } else {
        pads = context.const_input<Shape>(3); // Extract padding if provided
    }
    Strides dilations;
    if (!context.input_is_none(4)) {
        dilations = context.const_input<Strides>(4); // Extract dilations if provided
    }
    RoundingType rounding_type;
    if (context.input_is_none(5)) {
        rounding_type = RoundingType::FLOOR; // Default rounding type
    } else {
        rounding_type = context.const_input<bool>(5) ? RoundingType::CEIL : RoundingType::FLOOR; // Rounding type based on input
    }

    // Call average pooling operation with extracted parameters
    auto res = context.mark_node(std::make_shared<v8::AvgPool>(input, // Changed to AvgPool
                                                               strides,
                                                               dilations,
                                                               pads,
                                                               pads,
                                                               kernel,
                                                               rounding_type,
                                                               PadType::EXPLICIT,
                                                               element::i64,
                                                               pooling_dims));
    if (context.get_output_size() == 2) {
        auto out1 = res->output(0);
        auto out2 = res->output(1);
        return {std::move(out1), std::move(out2)};
    } else {
        return {res};
    }
}

OutputVector translate_avg_pool1d(const NodeContext& context) {
    const auto kernel_shape = Shape{context.const_input(1).to_vector<int64_t>()};
    return translate_avg_pool_base(context, context.const_input(0), kernel_shape, {}, {}, {}, RoundingType::FLOOR, 1); // Pooling dimension set for 1D
}

OutputVector translate_avg_pool2d(const NodeContext& context) {
    const auto kernel_shape = Shape{context.const_input(1).to_vector<int64_t>()};
    const auto input_size = context.const_input(0).sizes();
    const int64_t num_dims = input_size.size();
    
    Strides strides, pads, dilations;
    if (num_dims > 3) {
        strides = Strides{context.const_input(5).to_vector<int64_t>()};
        pads = Strides{context.const_input(3).to_vector<int64_t>()};
        dilations = Strides{context.const_input(4).to_vector<int64_t>()};
    }
    
    return translate_avg_pool_base(context, context.const_input(0), kernel_shape, strides, pads, dilations, RoundingType::FLOOR, 2); // Pooling dimension set for 2D
}

OutputVector translate_avg_pool3d(const NodeContext& context) {
    const auto kernel_shape = Shape{context.const_input(1).to_vector<int64_t>()};
    return translate_avg_pool_base(context, context.const_input(0), kernel_shape, {}, {}, {}, RoundingType::FLOOR, 3); // Pooling dimension set for 3D
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
