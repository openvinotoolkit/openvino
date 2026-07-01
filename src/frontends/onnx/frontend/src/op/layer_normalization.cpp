// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
using namespace ov::op;
using namespace ov::op::v0;
using namespace ov::op::v1;
using namespace ov::util;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ov::Shape;

ov::Output<ov::Node> rank(const ov::Output<ov::Node>& source) {
    return std::make_shared<Squeeze>(std::make_shared<v3::ShapeOf>(std::make_shared<v3::ShapeOf>(source)));
}

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector layer_normalization(const ov::frontend::onnx::Node& node) {
    // Operator definition: https://github.com/onnx/onnx/blob/main/onnx/defs/nn/defs.cc#L2562:L2611
    const auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();
    CHECK_VALID_NODE(node,
                     num_inputs == 2 || num_inputs == 3,
                     "LayerNormalization expects 2 or 3 input tensors. Got: ",
                     num_inputs);
    const auto num_outputs = node.get_outputs_size();
    CHECK_VALID_NODE(node,
                     num_outputs >= 1 && num_outputs <= 3,
                     "LayerNormalization expects 1, 2 or 3 output tensors. Got: ",
                     num_outputs);

    auto default_stash_type_i = static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT);
    int64_t stash_type_i = node.get_attribute_value<int64_t>("stash_type", default_stash_type_i);
    element::Type stash_type = common::get_ov_element_type(stash_type_i);

    ov::Output<ov::Node> data = inputs.at(0);
    element::Type original_type = data.get_element_type();
    bool needs_type_casting = stash_type != original_type;

    if (needs_type_casting)
        data = std::make_shared<Convert>(data, stash_type);

    float epsilon = node.get_attribute_value<float>("epsilon", 1e-5f);

    auto axis = node.get_attribute_value<std::int64_t>("axis", -1);
    // ONNX operator semantics says that `axis` attribute points to the first normalization dimension. We have to
    // figure out all dimensions for normalization:
    // axis < 0:  range(axis, 0)     Example: axis = -2; Axes: [-2, -1]
    // axis >= 0: range(axis, rank)  Example: axis = 2, rank = 4; Axes: [2, 3]
    auto axes =
        std::make_shared<v4::Range>(Constant::create(element::i64, {}, {axis}),
                                    (axis < 0 ? Constant::create(element::i64, {}, {0})->output(0) : rank(data)),
                                    Constant::create(element::i64, {}, {1}),
                                    element::i64);

    const auto normalize_variance = true;
    ov::Output<ov::Node> normalized =
        std::make_shared<v6::MVN>(data, axes, normalize_variance, epsilon, MVNEpsMode::INSIDE_SQRT);

    if (needs_type_casting)
        normalized = std::make_shared<ConvertLike>(normalized, inputs.at(0));

    // Use int32 max as the slice stop value; int64 max is not supported by all plugins (WA).
    constexpr auto slice_stop = std::numeric_limits<std::int32_t>::max();
    auto sub_shape = std::make_shared<v8::Slice>(std::make_shared<v0::ShapeOf>(normalized),
                                                 Constant::create(element::i64, {1}, {axis}),
                                                 Constant::create(element::i64, {1}, {slice_stop}),
                                                 Constant::create(element::i64, {1}, {1}));
    const auto normalized_rank = normalized.get_partial_shape().rank();
    const auto reshape_to_sub_shape = [&](ov::Output<ov::Node> param) -> ov::Output<ov::Node> {
        const auto param_rank = param.get_partial_shape().rank();
        const bool both_dynamic = param_rank.is_dynamic() && normalized_rank.is_dynamic();
        const bool size_mismatch = param_rank.is_static() && normalized_rank.is_static() &&
                                   param_rank.get_length() + normalize_axis(axis, normalized_rank.get_length()) !=
                                       static_cast<size_t>(normalized_rank.get_length());
        if (both_dynamic || size_mismatch) {
            return std::make_shared<v1::Reshape>(param, sub_shape, false);
        }
        return param;
    };

    ov::Output<ov::Node> y = std::make_shared<Multiply>(normalized, reshape_to_sub_shape(inputs.at(1)));
    if (common::is_input_valid(node, 2)) {
        y = std::make_shared<Add>(y, reshape_to_sub_shape(inputs.at(2)));
    }

    ov::OutputVector results{y};
    if (num_outputs == 1) {
        return results;
    }

    // Mean and InvStdDev are emitted in stash_type. MVN doesn't expose them, so they're recomputed via the
    // spec's reference decomposition (reduce over the same axes with keep_dims=true).
    const auto& output_names = node.get_output_names();
    const auto wanted = [&](size_t i) {
        return num_outputs > i && output_names.size() > i && !output_names[i].get().empty();
    };
    const auto null_output = []() {
        return std::make_shared<NullNode>()->output(0);
    };

    // Only build the reference decomposition when Mean and/or InvStdDev are actually requested, so inference-only
    // models that keep the extra outputs but leave them empty don't get redundant ReduceMean nodes.
    constexpr auto keep_dims = true;
    std::shared_ptr<ov::Node> mean;
    if (wanted(1) || wanted(2)) {
        mean = std::make_shared<v1::ReduceMean>(data, axes, keep_dims);
    }
    if (num_outputs >= 2) {
        results.push_back(wanted(1) ? mean->output(0) : null_output());
    }
    if (num_outputs >= 3) {
        if (wanted(2)) {
            auto deviation = std::make_shared<v1::Subtract>(data, mean);
            auto variance =
                std::make_shared<v1::ReduceMean>(std::make_shared<Multiply>(deviation, deviation), axes, keep_dims);
            auto std_dev = std::make_shared<v0::Sqrt>(
                std::make_shared<v1::Add>(variance, Constant::create(stash_type, {}, {epsilon})));
            results.push_back(std::make_shared<v1::Divide>(Constant::create(stash_type, {}, {1}), std_dev)->output(0));
        } else {
            results.push_back(null_output());
        }
    }
    return results;
}

ONNX_OP("LayerNormalization", OPSET_SINCE(1), ai_onnx::opset_1::layer_normalization);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
