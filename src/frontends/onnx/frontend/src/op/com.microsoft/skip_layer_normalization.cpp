// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector skip_layer_normalization(const ov::frontend::onnx::Node& node) {
    auto nodes = node.get_ov_inputs();
    const auto num_nodes = nodes.size();
    FRONT_END_GENERAL_CHECK(num_nodes >= 3 && num_nodes <= 5,
                            "SkipLayerNormalization takes 3, 4 or 5 inputs. Provided " + std::to_string(num_nodes));

    // input + skip
    std::shared_ptr<ov::Node> input = std::make_shared<v1::Add>(nodes[0], nodes[1]);
    // add bias if available
    if (num_nodes == 5) {
        input = std::make_shared<v1::Add>(input, nodes[4]);
    }
    const float eps = node.get_attribute_value<float>("epsilon", 1e-12f);
    // reduce over last dimension (default for regular LayerNormalization)
    const int last_dimension = -1;
    const auto reduction_axes = v0::Constant::create(ov::element::i32, ov::Shape{1}, {last_dimension});
    std::shared_ptr<ov::Node> result =
        std::make_shared<v6::MVN>(input, reduction_axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);
    // multiply by gamma
    result = std::make_shared<v1::Multiply>(result, nodes[2]);
    // add beta if available
    if (num_nodes > 3) {
        result = std::make_shared<v1::Add>(result, nodes[3]);
    }

    ov::OutputVector results{result->output(0)};
    const auto num_outputs = node.get_outputs_size();
    if (num_outputs == 1) {
        return results;
    }

    // Spec defines up to 4 outputs: output, mean, inv_std_var, input_skip_bias_sum.
    // MVN doesn't expose mean/inv_std_var, so they are recomputed via the reference decomposition, but only
    // when actually consumed downstream (input_skip_bias_sum is commonly used to chain consecutive
    // SkipLayerNormalization nodes and is already available as `input`).
    const auto& output_names = node.get_output_names();
    const auto wanted = [&](size_t i) {
        return num_outputs > i && output_names.size() > i && !output_names[i].get().empty();
    };
    const auto null_output = []() {
        return std::make_shared<NullNode>()->output(0);
    };

    constexpr auto keep_dims = true;
    std::shared_ptr<ov::Node> mean;
    if (wanted(1) || wanted(2)) {
        mean = std::make_shared<v1::ReduceMean>(input, reduction_axes, keep_dims);
    }
    if (num_outputs >= 2) {
        results.push_back(wanted(1) ? mean->output(0) : null_output());
    }
    if (num_outputs >= 3) {
        if (wanted(2)) {
            const auto eps_const = v0::Constant::create(input->get_element_type(), {}, {eps});
            const auto one_const = v0::Constant::create(input->get_element_type(), {}, {1});
            auto deviation = std::make_shared<v1::Subtract>(input, mean);
            auto variance = std::make_shared<v1::ReduceMean>(std::make_shared<v1::Multiply>(deviation, deviation),
                                                             reduction_axes,
                                                             keep_dims);
            auto std_dev = std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(variance, eps_const));
            results.push_back(std::make_shared<v1::Divide>(one_const, std_dev)->output(0));
        } else {
            results.push_back(null_output());
        }
    }
    if (num_outputs >= 4) {
        results.push_back(wanted(3) ? input->output(0) : null_output());
    }
    return results;
}
ONNX_OP("SkipLayerNormalization", OPSET_SINCE(1), com_microsoft::opset_1::skip_layer_normalization, MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
