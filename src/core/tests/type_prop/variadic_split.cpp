// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/variadic_split.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/shape_of.hpp"
#include "sequnce_generator.hpp"

using namespace std;
using namespace ov;
using namespace testing;

using VSplitTypePropTestParam = std::tuple<PartialShape,          // Input shape
                                           int64_t,               // Split axis
                                           std::vector<int64_t>,  // Split lengths
                                           PartialShapes          // Expected shapes
                                           >;

class VariadicSplitTest : public TestWithParam<VSplitTypePropTestParam> {
protected:
    void SetUp() override {
        std::tie(p_shape, axis, split_lengths, exp_shapes) = GetParam();
    }

    PartialShapes get_output_partial_shapes(const Node& n) const {
        PartialShapes out;
        for (size_t i = 0; i < n.get_output_size(); ++i) {
            out.push_back(n.get_output_partial_shape(i));
        }

        return out;
    }

    std::pair<ov::TensorLabel, ov::TensorLabel> make_in_exp_labels() const {
        ov::TensorLabel in_labels;
        std::generate_n(std::back_inserter(in_labels), p_shape.size(), ov::SeqGen<ov::label_t>(10));

        auto exp_labels = in_labels;
        OPENVINO_SUPPRESS_DEPRECATED_START
        const auto n_axis = ov::normalize_axis("", axis, p_shape.rank());
        OPENVINO_SUPPRESS_DEPRECATED_END
        exp_labels[n_axis] = ov::no_label;

        return {in_labels, exp_labels};
    }

    int64_t axis;
    std::vector<int64_t> split_lengths;
    PartialShape p_shape;
    PartialShapes exp_shapes;
};

INSTANTIATE_TEST_SUITE_P(type_prop_static_shape,
                         VariadicSplitTest,
                         Values(std::make_tuple(PartialShape{6, 2}, 0, std::vector<int64_t>{6}, PartialShapes{{6, 2}}),
                                std::make_tuple(PartialShape{6, 2, 10},
                                                -1,
                                                std::vector<int64_t>{6, -1, 3},
                                                PartialShapes{{6, 2, 6}, {6, 2, 1}, {6, 2, 3}}),
                                std::make_tuple(PartialShape{1, 20, 3},
                                                1,
                                                std::vector<int64_t>{-1, 10, 3, 5},
                                                PartialShapes{{1, 2, 3}, {1, 10, 3}, {1, 3, 3}, {1, 5, 3}})),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    type_prop_dynamic_shape,
    VariadicSplitTest,
    Values(
        std::make_tuple(PartialShape{{2, 6}, 2}, 0, std::vector<int64_t>{4}, PartialShapes{{4, 2}}),
        std::make_tuple(PartialShape{{2, 6}, 2},
                        0,
                        std::vector<int64_t>{4, 1, -1},
                        PartialShapes{{4, 2}, {1, 2}, {-1, 2}}),
        std::make_tuple(PartialShape{12, Dimension()},
                        -2,
                        std::vector<int64_t>{7, -1, 2},
                        PartialShapes{{7, -1}, {3, -1}, {2, -1}}),
        std::make_tuple(PartialShape{Dimension(), Dimension(), 6},
                        2,
                        std::vector<int64_t>{3, 1, 2},
                        PartialShapes{{-1, -1, 3}, {-1, -1, 1}, {-1, -1, 2}}),
        std::make_tuple(PartialShape{Dimension(), 6}, 1, std::vector<int64_t>{6, 0}, PartialShapes{{-1, 6}, {-1, 0}}),
        std::make_tuple(PartialShape{{2, 4}, Dimension::dynamic()},
                        1,
                        std::vector<int64_t>{4, 1, -1, 3},
                        PartialShapes{{{2, 4}, 4}, {{2, 4}, 1}, {{2, 4}, -1}, {{2, 4}, 3}})),
    PrintToStringParamName());

TEST_P(VariadicSplitTest, dimension_propagation_axis_scalar) {
    constexpr auto dtype = element::i32;
    const auto data = make_shared<ov::op::v0::Parameter>(dtype, p_shape);
    const auto axis_node = make_shared<ov::op::v0::Constant>(element::i16, Shape{}, axis);
    const auto lengths_node =
        std::make_shared<ov::op::v0::Constant>(element::i16, Shape{split_lengths.size()}, split_lengths);

    const auto var_split = make_shared<op::v1::VariadicSplit>(data, axis_node, lengths_node);

    EXPECT_EQ(var_split->get_output_size(), split_lengths.size());
    EXPECT_THAT(var_split->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, dtype)));
    EXPECT_THAT(get_output_partial_shapes(*var_split), ElementsAreArray(exp_shapes));
}

TEST_P(VariadicSplitTest, dimension_propagation_axis_1d) {
    constexpr auto dtype = element::u64;
    const auto data = make_shared<ov::op::v0::Parameter>(dtype, p_shape);
    const auto axis_node = make_shared<ov::op::v0::Constant>(element::i32, Shape{1}, axis);
    const auto lengths_node =
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{split_lengths.size()}, split_lengths);

    const auto var_split = make_shared<op::v1::VariadicSplit>(data, axis_node, lengths_node);

    EXPECT_EQ(var_split->get_output_size(), split_lengths.size());
    EXPECT_THAT(var_split->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, dtype)));
    EXPECT_THAT(get_output_partial_shapes(*var_split), ElementsAreArray(exp_shapes));
}

TEST_P(VariadicSplitTest, use_default_ctor) {
    constexpr auto dtype = element::f32;
    const auto param = make_shared<ov::op::v0::Parameter>(dtype, p_shape);
    const auto axis_node = make_shared<ov::op::v0::Constant>(element::i64, Shape{}, axis);
    const auto lengths_node =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{split_lengths.size()}, split_lengths);

    const auto var_split = make_shared<op::v1::VariadicSplit>();
    var_split->set_arguments(NodeVector{param, axis_node, lengths_node});
    var_split->validate_and_infer_types();

    EXPECT_EQ(var_split->get_output_size(), split_lengths.size());
    EXPECT_THAT(var_split->outputs(), Each(Property("Element type", &Output<Node>::get_element_type, dtype)));
    EXPECT_THAT(get_output_partial_shapes(*var_split), ElementsAreArray(exp_shapes));
}

TEST_P(VariadicSplitTest, label_propagation) {
    ov::TensorLabel in_labels, exp_labels;
    std::tie(in_labels, exp_labels) = make_in_exp_labels();

    set_shape_labels(p_shape, in_labels);
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, p_shape);
    const auto axis_node = make_shared<ov::op::v0::Constant>(element::i64, Shape{}, axis);
    const auto lengths_node =
        std::make_shared<ov::op::v0::Constant>(element::i64, Shape{split_lengths.size()}, split_lengths);

    const auto var_split = make_shared<op::v1::VariadicSplit>(data, axis_node, lengths_node);
    EXPECT_EQ(var_split->get_output_size(), split_lengths.size());
    EXPECT_THAT(
        var_split->outputs(),
        Each(Property("Partial shape", &Output<Node>::get_partial_shape, ResultOf(get_shape_labels, exp_labels))));
}

class VariadicSplitBoundTest : public VariadicSplitTest {
protected:
    std::pair<ov::TensorLabel, std::vector<ov::TensorLabel>> make_in_exp_labels() const {
        ov::TensorLabel in_labels;
        std::generate_n(std::back_inserter(in_labels), p_shape.size(), ov::SeqGen<ov::label_t>(8));

        std::vector<ov::TensorLabel> exp_labels;

        auto label_it = in_labels.begin();
        for (auto split_length : split_lengths) {
            if (split_length == 0) {
                exp_labels.emplace_back(ov::TensorLabel(1, ov::no_label));
            } else if (split_length == -1) {
                split_length = std::accumulate(split_lengths.cbegin(),
                                               split_lengths.cend(),
                                               static_cast<int64_t>(p_shape.size()),
                                               [](const int64_t& a, const int64_t& v) {
                                                   return (v != -1) ? a - v : a;
                                               });
                exp_labels.emplace_back(label_it, label_it + split_length);
            } else {
                exp_labels.emplace_back(label_it, label_it + split_length);
            }
            label_it += split_length;
        }
        return {in_labels, exp_labels};
    }

    std::vector<PartialShape> out_shapes;
    std::vector<ov::TensorLabel> out_labels;
};

INSTANTIATE_TEST_SUITE_P(type_prop_bounds_propagate,
                         VariadicSplitBoundTest,
                         Values(std::make_tuple(PartialShape{{2, 6}, 2, 3},
                                                0,
                                                std::vector<int64_t>{2, 1, 0},
                                                PartialShapes{{{2, 6}, 2}, {3}, {1}}),
                                std::make_tuple(PartialShape{{2, 6}, 2, 3, {-1, 6}, 5},
                                                0,
                                                std::vector<int64_t>{1, -1, 0, 2},
                                                PartialShapes{{{2, 6}}, {2, 3}, {1}, {{-1, 6}, 5}}),
                                std::make_tuple(PartialShape{{2, 6}, 2, 3, 8, 10, {-1, 6}, 5},
                                                0,
                                                std::vector<int64_t>{-1, 3, 0, 2},
                                                PartialShapes{{{2, 6}, 2}, {3, 8, 10}, {1}, {{-1, 6}, 5}}),
                                std::make_tuple(PartialShape{{2, 6}, 2, 3, 5},
                                                0,
                                                std::vector<int64_t>{2, -1, 0},
                                                PartialShapes{{{2, 6}, 2}, {3, 5}, {1}})),
                         PrintToStringParamName());

TEST_P(VariadicSplitBoundTest, propagate_label_and_dynamic_value) {
    ov::TensorLabel in_labels;
    std::vector<ov::TensorLabel> exp_labels;
    std::tie(in_labels, exp_labels) = make_in_exp_labels();
    set_shape_labels(p_shape, in_labels);

    constexpr auto et = element::i64;
    const auto labeled_param = std::make_shared<ov::op::v0::Parameter>(et, p_shape);
    const auto labeled_shape_of = std::make_shared<op::v0::ShapeOf>(labeled_param);

    const auto zero = std::vector<int64_t>{0};
    const auto axis_node = std::make_shared<op::v0::Constant>(et, Shape{}, zero);
    const auto lengths_node = std::make_shared<ov::op::v0::Constant>(et, Shape{split_lengths.size()}, split_lengths);
    const auto var_split = std::make_shared<op::v1::VariadicSplit>(labeled_shape_of, axis_node, lengths_node);

    for (auto& output : var_split->outputs()) {
        const auto& bc = std::make_shared<op::v3::Broadcast>(
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, PartialShape{1}),
            output,
            "BIDIRECTIONAL");
        out_shapes.push_back(bc->get_output_partial_shape(0));
        out_labels.push_back(get_shape_labels(bc->get_output_partial_shape(0)));
    }

    EXPECT_EQ(out_shapes, exp_shapes);
    EXPECT_EQ(out_labels, exp_labels);
}
