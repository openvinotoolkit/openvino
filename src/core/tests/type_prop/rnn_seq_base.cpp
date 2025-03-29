// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <type_traits>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gmock/gmock.h"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset12.hpp"

namespace rnn_seq_test {
using namespace std;
using namespace ov;
using namespace op;
using namespace testing;

struct RNNSeqParams {
    Dimension batch_size = 8;
    Dimension num_directions = 1;
    Dimension seq_length = 6;
    Dimension input_size = 4;
    Dimension hidden_size = 128;
    size_t outputs_size = 2;
    op::RecurrentSequenceDirection direction = op::RecurrentSequenceDirection::FORWARD;
    element::Type et = element::f32;
    int64_t gates_count = 1;
    bool linear_before_reset = false;
};

template <class TOp>
class RNNSeqBaseTest : public TypePropOpTest<TOp> {
public:
    template <typename T = TOp, typename std::enable_if<std::is_same<T, v5::RNNSequence>::value, bool>::type = true>
    std::shared_ptr<T> make_rnn_seq_based_op(RNNSeqParams& p, bool use_default_ctor = false) {
        const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.seq_length, p.input_size});
        const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.num_directions, p.hidden_size});
        const auto sequence_lengths = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size});
        const auto W =
            make_shared<v0::Parameter>(p.et,
                                       PartialShape{p.num_directions, p.hidden_size * p.gates_count, p.input_size});
        const auto R =
            make_shared<v0::Parameter>(p.et,
                                       PartialShape{p.num_directions, p.hidden_size * p.gates_count, p.hidden_size});
        const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.num_directions, p.hidden_size * p.gates_count});

        if (use_default_ctor) {
            auto op = std::make_shared<T>();
            op->set_direction(p.direction);
            op->set_hidden_size(p.hidden_size.get_max_length());
            op->set_arguments(OutputVector{X, H_t, sequence_lengths, W, R, B});
            op->validate_and_infer_types();
            return op;
        }

        return std::make_shared<T>(X, H_t, sequence_lengths, W, R, B, p.hidden_size.get_max_length(), p.direction);
    }

    template <typename T = TOp, typename std::enable_if<std::is_same<T, v5::GRUSequence>::value, bool>::type = true>
    std::shared_ptr<T> make_rnn_seq_based_op(RNNSeqParams& p, bool use_default_ctor = false) {
        p.gates_count = 3;

        const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.seq_length, p.input_size});
        const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.num_directions, p.hidden_size});
        const auto sequence_lengths = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size});
        const auto W =
            make_shared<v0::Parameter>(p.et,
                                       PartialShape{p.num_directions, p.hidden_size * p.gates_count, p.input_size});
        const auto R =
            make_shared<v0::Parameter>(p.et,
                                       PartialShape{p.num_directions, p.hidden_size * p.gates_count, p.hidden_size});
        const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.num_directions, p.hidden_size * p.gates_count});

        if (use_default_ctor) {
            auto op = std::make_shared<T>();
            op->set_direction(p.direction);
            op->set_hidden_size(p.hidden_size.get_max_length());
            op->set_arguments(OutputVector{X, H_t, sequence_lengths, W, R, B});
            op->validate_and_infer_types();
            return op;
        }
        return std::make_shared<T>(X, H_t, sequence_lengths, W, R, B, p.hidden_size.get_max_length(), p.direction);
    }

    template <typename T = TOp, typename std::enable_if<std::is_same<T, v5::LSTMSequence>::value, bool>::type = true>
    std::shared_ptr<T> make_rnn_seq_based_op(RNNSeqParams& p, bool use_default_ctor = false) {
        p.gates_count = 4;
        p.outputs_size = 3;

        const auto X = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.seq_length, p.input_size});
        const auto H_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.num_directions, p.hidden_size});
        const auto C_t = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size, p.num_directions, p.hidden_size});
        const auto sequence_lengths = make_shared<v0::Parameter>(p.et, PartialShape{p.batch_size});
        const auto W =
            make_shared<v0::Parameter>(p.et,
                                       PartialShape{p.num_directions, p.hidden_size * p.gates_count, p.input_size});
        const auto R =
            make_shared<v0::Parameter>(p.et,
                                       PartialShape{p.num_directions, p.hidden_size * p.gates_count, p.hidden_size});
        const auto B = make_shared<v0::Parameter>(p.et, PartialShape{p.num_directions, p.hidden_size * p.gates_count});

        if (use_default_ctor) {
            auto op = std::make_shared<T>();
            op->set_direction(p.direction);
            op->set_hidden_size(p.hidden_size.get_max_length());
            auto inputs = OutputVector{X, H_t, C_t, sequence_lengths, W, R, B};
            op->set_arguments(inputs);
            op->validate_and_infer_types();
            return op;
        }
        return std::make_shared<T>(X, H_t, C_t, sequence_lengths, W, R, B, p.hidden_size.get_max_length(), p.direction);
    }
};

TYPED_TEST_SUITE_P(RNNSeqBaseTest);

TYPED_TEST_P(RNNSeqBaseTest, basic_shape_infer) {
    RNNSeqParams params;
    auto op = this->make_rnn_seq_based_op(params);

    EXPECT_EQ(op->get_output_size(), params.outputs_size);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), Each(nullptr));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, default_ctor) {
    RNNSeqParams params;
    auto op = this->make_rnn_seq_based_op(params, true);

    EXPECT_EQ(op->get_output_size(), params.outputs_size);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), Each(nullptr));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, default_ctor_BIDIRECTIONAL) {
    RNNSeqParams params;
    params.direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;
    params.num_directions = Dimension(2);

    auto op = this->make_rnn_seq_based_op(params, true);

    EXPECT_EQ(op->get_output_size(), params.outputs_size);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 2, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), Each(nullptr));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 2, params.hidden_size}));
        EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), Each(nullptr));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, static_symbols_dims_shape_infer) {
    RNNSeqParams params;
    auto A = make_shared<Symbol>(), B = make_shared<Symbol>(), C = make_shared<Symbol>(), D = make_shared<Symbol>();
    params.batch_size = Dimension(8);
    params.batch_size.set_symbol(A);
    params.input_size = Dimension(64);
    params.seq_length.set_symbol(B);
    params.hidden_size = Dimension(128);
    params.hidden_size.set_symbol(C);
    params.num_directions = Dimension(1);
    params.num_directions.set_symbol(D);

    auto op = this->make_rnn_seq_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(A, D, B, C));

    for (size_t i = 1; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), ElementsAre(A, D, C));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, interval_symbols_dims_shape_infer_FORWARD) {
    RNNSeqParams params;
    auto A = make_shared<Symbol>(), B = make_shared<Symbol>(), C = make_shared<Symbol>(), D = make_shared<Symbol>();
    params.batch_size = Dimension(8, 16);
    params.batch_size.set_symbol(A);
    params.input_size = Dimension(64, 128);
    params.seq_length.set_symbol(B);
    params.hidden_size = Dimension(128, 256);
    params.hidden_size.set_symbol(C);
    params.num_directions = Dimension(1, 2);
    params.num_directions.set_symbol(D);

    auto op = this->make_rnn_seq_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(A, D, B, C));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        // For backward compatibility, hidden_size attribute is ignored
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), ElementsAre(A, D, C));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, interval_symbols_dims_shape_infer_REVERSE) {
    RNNSeqParams params;
    auto A = make_shared<Symbol>(), B = make_shared<Symbol>(), C = make_shared<Symbol>(), D = make_shared<Symbol>();
    params.batch_size = Dimension(8, 16);
    params.batch_size.set_symbol(A);
    params.input_size = Dimension(64, 128);
    params.seq_length.set_symbol(B);
    params.hidden_size = Dimension(128, 256);
    params.hidden_size.set_symbol(C);
    params.num_directions = Dimension(1, 2);
    params.num_directions.set_symbol(D);

    params.direction = op::RecurrentSequenceDirection::REVERSE;

    auto op = this->make_rnn_seq_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(A, D, B, C));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        // For backward compatibility, hidden_size attribute is ignored
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), ElementsAre(A, D, C));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, interval_symbols_dims_shape_infer_BIDIRECTIONAL) {
    RNNSeqParams params;
    auto A = make_shared<Symbol>(), B = make_shared<Symbol>(), C = make_shared<Symbol>(), D = make_shared<Symbol>();
    params.batch_size = Dimension(8, 16);
    params.batch_size.set_symbol(A);
    params.input_size = Dimension(64, 128);
    params.seq_length.set_symbol(B);
    params.hidden_size = Dimension(128, 256);
    params.hidden_size.set_symbol(C);
    params.num_directions = Dimension(1, 2);
    params.num_directions.set_symbol(D);

    params.direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;

    auto op = this->make_rnn_seq_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 2, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)), ElementsAre(A, D, B, C));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        // For backward compatibility, hidden_size attribute is ignored
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 2, params.hidden_size}));
        EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(i)), ElementsAre(A, D, C));
    }
}

REGISTER_TYPED_TEST_SUITE_P(RNNSeqBaseTest,
                            default_ctor,
                            default_ctor_BIDIRECTIONAL,
                            basic_shape_infer,
                            static_symbols_dims_shape_infer,
                            interval_symbols_dims_shape_infer_FORWARD,
                            interval_symbols_dims_shape_infer_REVERSE,
                            interval_symbols_dims_shape_infer_BIDIRECTIONAL);

using RNNSeqBaseTypes = Types<op::v5::RNNSequence, op::v5::GRUSequence, op::v5::LSTMSequence>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, RNNSeqBaseTest, RNNSeqBaseTypes);

}  // namespace rnn_seq_test
