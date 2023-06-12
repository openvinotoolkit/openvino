// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <type_traits>

#include "common_test_utils/test_assertions.hpp"
#include "gmock/gmock.h"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset12.hpp"
#include "util/type_prop.hpp"

namespace rnn_cell_test {
using namespace std;
using namespace ov;
using namespace op;
using namespace testing;

struct RNNCellParams {
    Dimension batch_size = 8;
    Dimension input_size = 4;
    Dimension hidden_size = 128;
    size_t outputs_size = 1;
    element::Type et = element::f32;
    int64_t gates_count = 1;
};

template <typename T, typename std::enable_if<std::is_same<T, v0::RNNCell>::value, bool>::type = true>
static std::shared_ptr<T> make_rnn_cell_based_op(RNNCellParams& p) {
    const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.input_size});
    const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.hidden_size});
    const auto W = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size, p.input_size});
    const auto R = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size, p.hidden_size});
    const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size});

    return std::make_shared<T>(X, H_t, W, R, B, p.hidden_size.get_max_length());
}

template <typename T, typename std::enable_if<std::is_same<T, v3::GRUCell>::value, bool>::type = true>
static std::shared_ptr<T> make_rnn_cell_based_op(RNNCellParams& p) {
    p.gates_count = 3;

    const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.input_size});
    const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.hidden_size});
    const auto W = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count, p.input_size});
    const auto R = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count, p.hidden_size});
    const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count});

    return std::make_shared<T>(X, H_t, W, R, B, p.hidden_size.get_max_length());
}

template <typename T,
          typename std::enable_if<std::is_same<T, v0::LSTMCell>::value || std::is_same<T, v4::LSTMCell>::value,
                                  bool>::type = true>
static std::shared_ptr<T> make_rnn_cell_based_op(RNNCellParams& p) {
    p.gates_count = 4;
    p.outputs_size = 2;

    const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.input_size});
    const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.hidden_size});
    const auto C_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.hidden_size});
    const auto W = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count, p.input_size});
    const auto R = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count, p.hidden_size});
    const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count});

    return std::make_shared<T>(X, H_t, C_t, W, R, B, p.hidden_size.get_max_length());
}

template <class T>
class RNNCellTest : public testing::Test {};

TYPED_TEST_SUITE_P(RNNCellTest);

TYPED_TEST_P(RNNCellTest, basic_shape_infer) {
    RNNCellParams params;

    auto op = make_rnn_cell_based_op<TypeParam>(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);
    for (size_t i = 0; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, params.hidden_size}));
    }
}

TYPED_TEST_P(RNNCellTest, static_labels_dims_shape_infer) {
    RNNCellParams params;
    params.batch_size = Dimension(8);
    ov::DimensionTracker::set_label(params.batch_size, 10);
    params.input_size = Dimension(64);
    ov::DimensionTracker::set_label(params.input_size, 11);
    params.hidden_size = Dimension(128);
    ov::DimensionTracker::set_label(params.hidden_size, 12);

    auto op = make_rnn_cell_based_op<TypeParam>(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    for (size_t i = 0; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, params.hidden_size}));
        EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), ElementsAre(10, 12));
    }
}

TYPED_TEST_P(RNNCellTest, interval_labels_dims_shape_infer) {
    RNNCellParams params;
    params.batch_size = Dimension(8, 16);
    ov::DimensionTracker::set_label(params.batch_size, 10);
    params.input_size = Dimension(64, 128);
    ov::DimensionTracker::set_label(params.input_size, 11);
    params.hidden_size = Dimension(128, 256);
    ov::DimensionTracker::set_label(params.hidden_size, 12);

    auto op = make_rnn_cell_based_op<TypeParam>(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    for (size_t i = 0; i < params.outputs_size; ++i) {
        if (ov::is_type<v0::LSTMCell>(op) || ov::is_type<v4::LSTMCell>(op)) {
            // For backward compatibility, if hidden_size dim is dynamic, set the value based on attribute
            EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, op->get_hidden_size()}));
            EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), ElementsAre(10, 0));
        } else {
            // For backward compatibility, hidden_size attribute is ignored
            EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, params.hidden_size}));
            EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), ElementsAre(10, 12));
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(RNNCellTest,
                            basic_shape_infer,
                            static_labels_dims_shape_infer,
                            interval_labels_dims_shape_infer);

using RNNCellBaseTypes = Types<op::v0::RNNCell, op::v3::GRUCell, op::v0::LSTMCell, op::v4::LSTMCell>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, RNNCellTest, RNNCellBaseTypes);

}  // namespace rnn_cell_test
