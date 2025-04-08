// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <type_traits>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset12.hpp"

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

template <class TOp>
class RNNCellTest : public TypePropOpTest<TOp> {
public:
    template <typename T = TOp, typename std::enable_if<std::is_same<T, v0::RNNCell>::value, bool>::type = true>
    std::shared_ptr<T> make_rnn_cell_based_op(RNNCellParams& p, bool use_default_ctor = false) {
        const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.input_size});
        const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.hidden_size});
        const auto W = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size, p.input_size});
        const auto R = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size, p.hidden_size});
        const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size});

        if (use_default_ctor) {
            auto op = std::make_shared<T>();
            op->set_hidden_size(p.hidden_size.get_max_length());
            op->set_arguments(OutputVector{X, H_t, W, R, B});
            op->validate_and_infer_types();
            return op;
        }

        return std::make_shared<T>(X, H_t, W, R, B, p.hidden_size.get_max_length());
    }

    template <typename T = TOp, typename std::enable_if<std::is_same<T, v3::GRUCell>::value, bool>::type = true>
    std::shared_ptr<T> make_rnn_cell_based_op(RNNCellParams& p, bool use_default_ctor = false) {
        p.gates_count = 3;

        const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.input_size});
        const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.hidden_size});
        const auto W = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count, p.input_size});
        const auto R = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count, p.hidden_size});
        const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count});

        if (use_default_ctor) {
            auto op = std::make_shared<T>();
            op->set_hidden_size(p.hidden_size.get_max_length());
            op->set_arguments(OutputVector{X, H_t, W, R, B});
            op->validate_and_infer_types();
            return op;
        }

        return std::make_shared<T>(X, H_t, W, R, B, p.hidden_size.get_max_length());
    }

    template <typename T = TOp,
              typename std::enable_if<std::is_same<T, v0::LSTMCell>::value || std::is_same<T, v4::LSTMCell>::value,
                                      bool>::type = true>
    std::shared_ptr<T> make_rnn_cell_based_op(RNNCellParams& p, bool use_default_ctor = false) {
        p.gates_count = 4;
        p.outputs_size = 2;

        const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.input_size});
        const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.hidden_size});
        const auto C_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.hidden_size});
        const auto W = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count, p.input_size});
        const auto R = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count, p.hidden_size});
        const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * p.gates_count});

        if (use_default_ctor) {
            auto op = std::make_shared<T>();
            op->set_hidden_size(p.hidden_size.get_max_length());
            auto inputs = OutputVector{X, H_t, C_t, W, R, B};
            if (ov::is_type<v0::LSTMCell>(op)) {
                const auto P = make_shared<v0::Parameter>(p.et, PartialShape{p.hidden_size * (p.gates_count - 1)});
                inputs.push_back(P);
            }
            op->set_arguments(inputs);
            op->validate_and_infer_types();
            return op;
        }

        return std::make_shared<T>(X, H_t, C_t, W, R, B, p.hidden_size.get_max_length());
    }
};

TYPED_TEST_SUITE_P(RNNCellTest);

TYPED_TEST_P(RNNCellTest, basic_shape_infer) {
    RNNCellParams params;

    auto op = this->make_rnn_cell_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);
    for (size_t i = 0; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, params.hidden_size}));
    }
}

TYPED_TEST_P(RNNCellTest, default_ctor) {
    RNNCellParams params;

    auto op = this->make_rnn_cell_based_op(params, true);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);
    for (size_t i = 0; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, params.hidden_size}));
    }
}

TYPED_TEST_P(RNNCellTest, static_symbols_dims_shape_infer) {
    RNNCellParams params;
    auto A = make_shared<Symbol>(), B = make_shared<Symbol>(), C = make_shared<Symbol>();
    params.batch_size = Dimension(8);
    params.batch_size.set_symbol(A);
    params.input_size = Dimension(64);
    params.input_size.set_symbol(B);
    params.hidden_size = Dimension(128);
    params.hidden_size.set_symbol(C);

    auto op = this->make_rnn_cell_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    for (size_t i = 0; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, params.hidden_size}));
        EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), ElementsAre(A, C));
    }
}

TYPED_TEST_P(RNNCellTest, interval_symbols_dims_shape_infer) {
    RNNCellParams params;
    auto A = make_shared<Symbol>(), B = make_shared<Symbol>(), C = make_shared<Symbol>();
    params.batch_size = Dimension(8, 16);
    params.batch_size.set_symbol(A);
    params.input_size = Dimension(64, 128);
    params.input_size.set_symbol(B);
    params.hidden_size = Dimension(128, 256);
    params.hidden_size.set_symbol(C);

    auto op = this->make_rnn_cell_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    for (size_t i = 0; i < params.outputs_size; ++i) {
        if (ov::is_type<v0::LSTMCell>(op) || ov::is_type<v4::LSTMCell>(op)) {
            // For backward compatibility, if hidden_size dim is dynamic, set the value based on attribute
            EXPECT_EQ(op->get_output_partial_shape(i),
                      (PartialShape{params.batch_size, static_cast<int64_t>(op->get_hidden_size())}));
            EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), ElementsAre(A, nullptr));
        } else {
            // For backward compatibility, hidden_size attribute is ignored
            EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, params.hidden_size}));
            EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), ElementsAre(A, C));
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(RNNCellTest,
                            basic_shape_infer,
                            default_ctor,
                            static_symbols_dims_shape_infer,
                            interval_symbols_dims_shape_infer);

using RNNCellBaseTypes = Types<op::v0::RNNCell, op::v3::GRUCell, op::v0::LSTMCell, op::v4::LSTMCell>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, RNNCellTest, RNNCellBaseTypes);

}  // namespace rnn_cell_test
