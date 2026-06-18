// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder_norm.hpp"

#include "model_builder_internal.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

ov::Output<ov::Node> LayerNorm::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    float w_val = 1.0f + fill_value_from_name(name + ".weight") * 0.1f;
    float b_val = fill_value_from_name(name + ".bias") * 0.01f;
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, w_val));
    weight->set_friendly_name(name + ".weight");

    auto bias =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, b_val));
    bias->set_friendly_name(name + ".bias");

    auto axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});

    auto mvn = std::make_shared<ov::opset11::MVN>(input, axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);
    mvn->set_friendly_name(name + "_mvn");

    auto mul = std::make_shared<ov::opset11::Multiply>(mvn, weight);

    auto add = std::make_shared<ov::opset11::Add>(mul, bias);
    add->set_friendly_name(name);

    return add->output(0);
}

ov::Output<ov::Node> RMSNorm::operator()(const ov::Output<ov::Node>& input, const std::string& name) const {
    float w_val = 1.0f + fill_value_from_name(name + ".weight") * 0.1f;
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, w_val));
    weight->set_friendly_name(name + ".weight");

    auto squared = std::make_shared<ov::opset11::Multiply>(input, input);

    auto axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});

    auto mean = std::make_shared<ov::opset11::ReduceMean>(squared, axes, true);

    auto eps_const = ov::opset11::Constant::create(precision, ov::Shape{}, {eps});

    auto mean_eps = std::make_shared<ov::opset11::Add>(mean, eps_const);

    auto rsqrt = std::make_shared<ov::opset11::Sqrt>(mean_eps);

    auto normalized = std::make_shared<ov::opset11::Divide>(input, rsqrt);

    auto scaled = std::make_shared<ov::opset11::Multiply>(normalized, weight);
    scaled->set_friendly_name(name);

    return scaled->output(0);
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
