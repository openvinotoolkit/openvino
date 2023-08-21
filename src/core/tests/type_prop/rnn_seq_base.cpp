// Copyright (C) 2018-2023 Intel Corporation
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

    template <
        typename T = TOp,
        typename std::enable_if<std::is_same<T, v0::LSTMSequence>::value || std::is_same<T, v5::LSTMSequence>::value,
                                bool>::type = true>
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
            if (ov::is_type<v0::LSTMSequence>(op)) {
                const auto P =
                    make_shared<v0::Parameter>(p.et,
                                               PartialShape{p.num_directions, p.hidden_size * (p.gates_count - 1)});
                inputs.push_back(P);
            }
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
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), Each(ov::no_label));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), Each(ov::no_label));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, default_ctor) {
    RNNSeqParams params;
    auto op = this->make_rnn_seq_based_op(params, true);

    EXPECT_EQ(op->get_output_size(), params.outputs_size);
    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), Each(ov::no_label));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), Each(ov::no_label));
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
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), Each(ov::no_label));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 2, params.hidden_size}));
        EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), Each(ov::no_label));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, static_labels_dims_shape_infer) {
    RNNSeqParams params;
    params.batch_size = Dimension(8);
    ov::DimensionTracker::set_label(params.batch_size, 10);
    params.input_size = Dimension(64);
    ov::DimensionTracker::set_label(params.seq_length, 11);
    params.hidden_size = Dimension(128);
    ov::DimensionTracker::set_label(params.hidden_size, 12);
    params.num_directions = Dimension(1);
    ov::DimensionTracker::set_label(params.num_directions, 13);

    auto op = this->make_rnn_seq_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 13, 11, 12));

    for (size_t i = 1; i < params.outputs_size; ++i) {
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), ElementsAre(10, 13, 12));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, interval_labels_dims_shape_infer_FORWARD) {
    RNNSeqParams params;
    params.batch_size = Dimension(8, 16);
    ov::DimensionTracker::set_label(params.batch_size, 10);
    params.input_size = Dimension(64, 128);
    ov::DimensionTracker::set_label(params.seq_length, 11);
    params.hidden_size = Dimension(128, 256);
    ov::DimensionTracker::set_label(params.hidden_size, 12);
    params.num_directions = Dimension(1, 2);
    ov::DimensionTracker::set_label(params.num_directions, 13);

    auto op = this->make_rnn_seq_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 13, 11, 12));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        // For backward compatibility, hidden_size attribute is ignored
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), ElementsAre(10, 13, 12));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, interval_labels_dims_shape_infer_REVERSE) {
    RNNSeqParams params;
    params.batch_size = Dimension(8, 16);
    ov::DimensionTracker::set_label(params.batch_size, 10);
    params.input_size = Dimension(64, 128);
    ov::DimensionTracker::set_label(params.seq_length, 11);
    params.hidden_size = Dimension(128, 256);
    ov::DimensionTracker::set_label(params.hidden_size, 12);
    params.num_directions = Dimension(1, 2);
    ov::DimensionTracker::set_label(params.num_directions, 13);

    params.direction = op::RecurrentSequenceDirection::REVERSE;

    auto op = this->make_rnn_seq_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 1, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 13, 11, 12));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        // For backward compatibility, hidden_size attribute is ignored
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 1, params.hidden_size}));
        EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), ElementsAre(10, 13, 12));
    }
}

TYPED_TEST_P(RNNSeqBaseTest, interval_labels_dims_shape_infer_BIDIRECTIONAL) {
    RNNSeqParams params;
    params.batch_size = Dimension(8, 16);
    ov::DimensionTracker::set_label(params.batch_size, 10);
    params.input_size = Dimension(64, 128);
    ov::DimensionTracker::set_label(params.seq_length, 11);
    params.hidden_size = Dimension(128, 256);
    ov::DimensionTracker::set_label(params.hidden_size, 12);
    params.num_directions = Dimension(1, 2);
    ov::DimensionTracker::set_label(params.num_directions, 13);

    params.direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;

    auto op = this->make_rnn_seq_based_op(params);
    EXPECT_EQ(op->get_output_size(), params.outputs_size);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{params.batch_size, 2, params.seq_length, params.hidden_size}));
    EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(0)), ElementsAre(10, 13, 11, 12));
    for (size_t i = 1; i < params.outputs_size; ++i) {
        // For backward compatibility, hidden_size attribute is ignored
        EXPECT_EQ(op->get_output_partial_shape(i), (PartialShape{params.batch_size, 2, params.hidden_size}));
        EXPECT_THAT(get_shape_labels(op->get_output_partial_shape(i)), ElementsAre(10, 13, 12));
    }
}

REGISTER_TYPED_TEST_SUITE_P(RNNSeqBaseTest,
                            default_ctor,
                            default_ctor_BIDIRECTIONAL,
                            basic_shape_infer,
                            static_labels_dims_shape_infer,
                            interval_labels_dims_shape_infer_FORWARD,
                            interval_labels_dims_shape_infer_REVERSE,
                            interval_labels_dims_shape_infer_BIDIRECTIONAL);

using RNNSeqBaseTypes = Types<op::v5::RNNSequence, op::v5::GRUSequence, op::v0::LSTMSequence, op::v5::LSTMSequence>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop, RNNSeqBaseTest, RNNSeqBaseTypes);

}  // namespace rnn_seq_test
