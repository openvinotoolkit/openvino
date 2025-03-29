// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/subgraph_builders/ti_with_lstm_cell.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tensor_iterator.hpp"

namespace ov {
namespace test {
namespace utils {
std::shared_ptr<ov::Model> make_ti_with_lstm_cell(ov::element::Type type, size_t N, size_t L, size_t I, size_t H) {
    auto SENT = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{N, L, I});

    auto H_init = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{N, 1, H});
    auto C_init = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{N, 1, H});

    auto H_t = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{N, 1, H});
    auto C_t = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{N, 1, H});

    // Body
    auto X = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{N, 1, I});
    std::vector<uint64_t> dataW(4 * H * I, 0);
    auto W_body = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{4 * H, I}, dataW);
    std::vector<uint64_t> dataR(4 * H * H, 0);
    auto R_body = std::make_shared<ov::op::v0::Constant>(type, ov::Shape{4 * H, H}, dataR);
    std::vector<uint64_t> inShape = {N, H};
    auto constantH = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, inShape);
    inShape = {N, I};
    auto constantX = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2}, inShape);
    auto LSTM_cell =
        std::make_shared<ov::op::v4::LSTMCell>(std::make_shared<ov::op::v1::Reshape>(X, constantX, false),
                                               std::make_shared<ov::op::v1::Reshape>(H_t, constantH, false),
                                               std::make_shared<ov::op::v1::Reshape>(C_t, constantH, false),
                                               W_body,
                                               R_body,
                                               H);
    inShape = {N, 1, H};
    auto constantHo = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, inShape);
    auto H_o = std::make_shared<ov::op::v1::Reshape>(LSTM_cell->output(0), constantHo, false);
    auto C_o = std::make_shared<ov::op::v1::Reshape>(LSTM_cell->output(1), constantHo, false);
    auto body = std::make_shared<ov::Model>(ov::OutputVector{H_o, C_o}, ov::ParameterVector{X, H_t, C_t});

    auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
    tensor_iterator->set_body(body);
    // start=0, stride=1, part_size=1, end=39, axis=1
    tensor_iterator->set_sliced_input(X, SENT, 0, 1, 1, -1, 1);
    // H_t is Hinit on the first iteration, Ho after that
    tensor_iterator->set_merged_input(H_t, H_init, H_o);
    tensor_iterator->set_merged_input(C_t, C_init, C_o);

    // Output 0 is last Ho, result 0 of body
    auto out0 = tensor_iterator->get_iter_value(H_o, -1);
    // Output 1 is last Co, result 1 of body
    auto out1 = tensor_iterator->get_iter_value(C_o, -1);

    auto results =
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(out0), std::make_shared<ov::op::v0::Result>(out1)};

    auto model = std::make_shared<ov::Model>(results, ov::ParameterVector{SENT, H_init, C_init});
    model->set_friendly_name("TIwithLSTMcell");
    return model;
}

}  // namespace utils
}  // namespace test
}  // namespace ov