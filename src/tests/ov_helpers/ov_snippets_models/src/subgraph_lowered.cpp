// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_lowered.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include <snippets/op/broadcastload.hpp>
#include <snippets/op/broadcastmove.hpp>
#include <snippets/op/convert_saturation.hpp>
#include <snippets/op/load.hpp>
#include <snippets/op/store.hpp>
#include <snippets/op/scalar.hpp>
#include <snippets/op/brgemm.hpp>

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> AddFunctionLoweredBroadcast::initLowered() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    std::shared_ptr<Node> add_input0 = nullptr;
    if (!broadcast_shapes[0].empty() && broadcast_shapes[0].back() != input_shapes[0].rbegin()->get_length()) {
        add_input0 = std::make_shared<ov::snippets::op::BroadcastLoad>(data0, *broadcast_shapes[0].rbegin());
    } else {
        add_input0 = std::make_shared<ov::snippets::op::Load>(data0);
    }

    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    std::shared_ptr<Node> add_input1 = nullptr;
    if (!broadcast_shapes[1].empty() && broadcast_shapes[1].back() != input_shapes[1].rbegin()->get_length()) {
        add_input1 = std::make_shared<ov::snippets::op::BroadcastLoad>(data1, *broadcast_shapes[1].rbegin());
    } else {
        add_input1 = std::make_shared<ov::snippets::op::Load>(data1);
    }
    auto add = std::make_shared<op::v1::Add>(add_input0, add_input1);
    auto store = std::make_shared<ov::snippets::op::Store>(add);
    ParameterVector input_params {data0, data1};
    return std::make_shared<ov::Model>(OutputVector{store}, input_params);
}
std::shared_ptr<ov::Model> EltwiseThreeInputsLoweredFunction::initLowered() const {
    // todo: implement conversion between std::vector<size_t> and std::vector<Shape>
    ov::ParameterVector input_params{std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]),
                                     std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]),
                                     std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[2])};
    auto load_or_broadcastload = [&](size_t i) -> std::shared_ptr<Node> {
        // user specified that no broadcasting is required
        if (broadcast_shapes[i].empty()) {
            return std::make_shared<ov::snippets::op::Load>(input_params[i]);
        // broadcasting is required: could be Load + BroadcastMove or BroiadcastLoad
        } else {
            // The last dim is processed by vector Tile, so BroadcastLoad is required if the last dim being broadcasted
            if (input_shapes[i].rbegin()->get_length() == 1 && broadcast_shapes[i].back() != 1) {
                return std::make_shared<ov::snippets::op::BroadcastLoad>(input_params[i], *broadcast_shapes[i].rbegin());
            // Todo: Cover this logics with functional tests, Review FakeBroadcast Emitter
            // Broadcasting of other dims is handled by BroadcastMove. Strictly speaking, broadcasting is achieved via
            // appropriate pointer arithmetics in this case.
            } else {
                auto load = std::make_shared<ov::snippets::op::Load>(input_params[i]);
                return std::make_shared<ov::snippets::op::BroadcastMove>(load, *broadcast_shapes[i].rbegin());
            }
        }
    };
    auto add = std::make_shared<op::v1::Add>(load_or_broadcastload(0), load_or_broadcastload(1));

    const std::vector<float> const_values = ov::test::utils::generate_float_numbers(1, -10., 10.);
    auto sub_scalar = std::make_shared<ov::snippets::op::Scalar>(precision, Shape{1}, const_values[0]);
    std::shared_ptr<Node> sub_load;
    sub_load = std::make_shared<ov::snippets::op::Load>(input_params[2]);
    auto sub = std::make_shared<op::v1::Subtract>(sub_load, sub_scalar);
    std::shared_ptr<Node> sub_out;
    if (broadcast_shapes[2].empty())
        sub_out = sub;
    else
        sub_out = std::make_shared<ov::snippets::op::BroadcastMove>(sub, *broadcast_shapes[2].rbegin());
    auto mul = std::make_shared<op::v1::Multiply>(add, sub_out);
    auto store = std::make_shared<ov::snippets::op::Store>(mul);
    return std::make_shared<ov::Model>(OutputVector{store}, input_params);
}

std::shared_ptr<ov::Model> Transpose0213MatMulLoweredFunction::initLowered() const {
    ParameterVector data{std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]),
                         std::make_shared<op::v0::Parameter>(precisions[1], input_shapes[1])};
    std::vector<size_t> layout{0, 2, 1, 3};
    // Note: validity of transpose_position values is checked in Transpose0213MatMulSinhFunction constructor
    if (transpose_position < 2) {
        const auto& anchor = data[transpose_position]->output(0);
        const auto& td = ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(anchor);
        const auto& tensor = td->get_shape();
        const auto& subtensor = td->get_subtensor();
    }
    auto matmul = std::make_shared<ov::snippets::op::Brgemm>(data[0], data[1], 0, 0, 0, transpose_position == 0 ? layout : std::vector<size_t>{},
                                                                                        transpose_position == 1 ? layout : std::vector<size_t>{},
                                                                                        transpose_position == 2 ? layout : std::vector<size_t>{});
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    if (transpose_position == 2) {
        const auto& anchor = matmul->output(0);
        const auto& td = ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(anchor);
        const auto& tensor = td->get_shape();
        const auto& subtensor = td->get_subtensor();
        ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(anchor,
                                                                        std::make_shared<ov::snippets::lowered::PortDescriptor>(tensor,
                                                                                                                                subtensor,
                                                                                                                                layout));
    }
    if (transpose_position < 2) {
        const auto& anchor = data[transpose_position]->output(0);
        const auto& td = ov::snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(anchor);
        const auto& tensor = td->get_shape();
        const auto& subtensor = td->get_subtensor();
        ov::snippets::lowered::PortDescriptorUtils::set_port_descriptor_ptr(matmul->input(transpose_position),
                                                                        std::make_shared<ov::snippets::lowered::PortDescriptor>(tensor,
                                                                                                                                subtensor,
                                                                                                                                layout));
    }
    matmul->validate_and_infer_types();
    return std::make_shared<ov::Model>(OutputVector{matmul}, data);
}

std::shared_ptr<ov::Model> BroadcastAddLoweredFunction::initLowered() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    ov::NodeVector datas = {data0, data1};
    auto last_dim = std::max(input_shapes[0].get_shape().back(), std::max(input_shapes[1].get_shape().back(), m_target_shape.get_shape().back()));
    ov::NodeVector loads(datas.size(), nullptr);
    for (auto i = 0; i < datas.size(); i++) {
        if (input_shapes[i].get_shape().back() != last_dim) {
            loads[i] = std::make_shared<ov::snippets::op::BroadcastLoad>(datas[i], ov::Dimension(last_dim));
        } else {
            loads[i] = std::make_shared<ov::snippets::op::Load>(datas[i]);
        }
    }
    auto add = std::make_shared<op::v1::Add>(loads[0], loads[1]);
    auto store = std::make_shared<ov::snippets::op::Store>(add);
    return std::make_shared<Model>(OutputVector{store}, ParameterVector{data0, data1});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
