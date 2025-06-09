// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_gated_mlp.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/activation.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/op/subgraph.hpp"

namespace ov {
namespace test {
namespace snippets {

std::ostream& operator<<(std::ostream& os, GatedMLPFunction::WeightFormat type) {
    switch (type) {
    case GatedMLPFunction::WeightFormat::FP32:
        os << "FP32";
        break;
    case GatedMLPFunction::WeightFormat::FP16:
        os << "FP16";
        break;
    default:
        OPENVINO_THROW("Unexpected weight format!");
    }
    return os;
}

std::shared_ptr<ov::Node> GatedMLPFunction::makeWeights(const Shape& shape, int32_t seed) const {
    utils::InputGenerateData gen_data;
    gen_data.seed = seed;
    switch (m_wei_format) {
    case WeightFormat::FP32:
        return ov::test::utils::make_constant(ov::element::f32, shape, gen_data);
    case WeightFormat::FP16:
        return ov::test::utils::make_constant(ov::element::f16, shape, gen_data);
    default:
        OPENVINO_THROW("Unexpected weight format!");
    }
}

std::shared_ptr<ov::Node> GatedMLPFunction::makeFC(const ov::Output<ov::Node>& output, const ov::Output<ov::Node>& weights) const {
    return std::make_shared<ov::op::v0::MatMul>(output, weights, false, true);
}

std::shared_ptr<ov::Node> GatedMLPFunction::makeFC(const ov::Output<ov::Node>& output, const Shape& shape, int32_t seed) const {
    std::shared_ptr<ov::Node> weights = makeWeights(shape, seed);
    if (weights->get_output_element_type(0) != output.get_element_type()) {
        weights = std::make_shared<ov::op::v0::Convert>(weights, output.get_element_type());
    }
    return makeFC(output, weights->output(0));
}

std::shared_ptr<ov::Model> GatedMLPFunction::initOriginal() const {
    auto param = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);

    auto fc_gate = makeFC(param, m_weights_shapes[0], 3);
    auto fc_up = makeFC(param, m_weights_shapes[1], 7);
    auto act = ov::test::utils::make_activation(fc_gate, precision, m_act_type, ov::Shape{}, std::vector<float>{0.5});
    auto mul = std::make_shared<ov::op::v1::Multiply>(act, fc_up);
    auto fc_down = makeFC(mul, m_weights_shapes[2], 11);

    auto result = std::make_shared<ov::op::v0::Result>(fc_down);
    return std::make_shared<Model>(ResultVector{result}, ParameterVector{param});
}

std::shared_ptr<ov::Model> GatedMLPFunction::initReference() const {
    auto data = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);

    auto fc_gate_weights = makeWeights(m_weights_shapes[0], 3);
    auto fc_up_weights = makeWeights(m_weights_shapes[1], 7);
    auto fc_down_weights = makeWeights(m_weights_shapes[2], 11);

    NodeVector subgraph_inputs = {data, fc_gate_weights, data, fc_up_weights, fc_down_weights};

    auto param0 = std::make_shared<ov::op::v0::Parameter>(data->get_element_type(), data->get_output_partial_shape(0));
    auto param1 = std::make_shared<ov::op::v0::Parameter>(data->get_element_type(), data->get_output_partial_shape(0));
    auto param2 = std::make_shared<ov::op::v0::Parameter>(fc_gate_weights->get_element_type(), fc_gate_weights->get_output_partial_shape(0));
    auto param3 = std::make_shared<ov::op::v0::Parameter>(fc_up_weights->get_element_type(), fc_up_weights->get_output_partial_shape(0));
    auto param4 = std::make_shared<ov::op::v0::Parameter>(fc_down_weights->get_element_type(), fc_down_weights->get_output_partial_shape(0));

    auto fc_gate = makeFC(param0, param2);
    auto fc_up = makeFC(param1, param3);
    auto act = ov::test::utils::make_activation(fc_gate, precision, m_act_type, ov::Shape{}, std::vector<float>{0.5});
    auto mul = std::make_shared<ov::op::v1::Multiply>(act, fc_up);
    auto fc_down = makeFC(mul, param4);

    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
        subgraph_inputs,
        std::make_shared<ov::Model>(OutputVector{fc_down}, ParameterVector{param0, param2, param1, param3, param4}));

    return std::make_shared<Model>(OutputVector{subgraph}, ParameterVector{data});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
