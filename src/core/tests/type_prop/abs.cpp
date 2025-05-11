// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/abs.hpp"

#include <gtest/gtest.h>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/shape_of.hpp"
#include "unary_ops.hpp"

using Type = ::testing::Types<ov::op::v0::Abs>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_abs, UnaryOperator, Type);

struct Bound {
    int64_t lower, upper, expected_lower, expected_upper;
    std::shared_ptr<ov::Symbol> in_symbol;
};

using AbsTestParams = std::vector<Bound>;

class TypePropAbsV0Test : public testing::TestWithParam<AbsTestParams> {};

namespace {
std::shared_ptr<ov::Node> construct_graph_with_partial_value(const AbsTestParams& params) {
    auto shape = std::vector<ov::Dimension>();
    auto subtrahend = std::vector<int64_t>();

    for (const auto& i_bound : params) {
        if (i_bound.lower >= 0) {
            shape.emplace_back(i_bound.lower, i_bound.upper);
            subtrahend.push_back(0);
        } else {
            shape.emplace_back(0, (i_bound.upper - i_bound.lower));
            subtrahend.push_back(i_bound.lower);
        }
    }

    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::dynamic, shape);
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(parameter);
    auto subtract = std::make_shared<ov::op::v1::Add>(
        shape_of,
        ov::op::v0::Constant::create(element::i64, ov::Shape{subtrahend.size()}, subtrahend));
    return subtract;
}
}  // namespace

TEST_P(TypePropAbsV0Test, type_prop_abs) {
    const auto& data = GetParam();

    auto abs = std::make_shared<ov::op::v0::Abs>(construct_graph_with_partial_value(data));
    ov::TensorSymbol symbols;
    for (auto& item : data)
        symbols.push_back(item.in_symbol);
    abs->get_input_tensor(0).set_value_symbol(symbols);

    ov::PartialShape output_value;
    ASSERT_TRUE(ov::util::evaluate_as_partial_shape(abs->output(0), output_value));
    ASSERT_TRUE(output_value.rank().is_static());
    ASSERT_EQ(output_value.size(), data.size());

    for (size_t i = 0; i < output_value.size(); ++i) {
        ASSERT_EQ(output_value[i].get_min_length(), data[i].expected_lower);
        ASSERT_EQ(output_value[i].get_max_length(), data[i].expected_upper);
        if (data[i].lower == data[i].expected_lower && data[i].upper == data[i].expected_upper)
            ASSERT_TRUE(ov::symbol::are_equal(symbols[i], output_value[i].get_symbol()));
        else
            ASSERT_TRUE(output_value[i].get_symbol() == nullptr);
    }
}

INSTANTIATE_TEST_SUITE_P(type_prop_abs,
                         TypePropAbsV0Test,
                         testing::ValuesIn(std::vector<AbsTestParams>{{
                             //    lower, upper
                             // negative, negative (equal)
                             {-6, -6, 6, 6, std::make_shared<ov::Symbol>()},
                             // negative, negative (not equal)
                             {-5, -4, 4, 5, std::make_shared<ov::Symbol>()},
                             // negative, zero
                             {-4, 0, 0, 4, std::make_shared<ov::Symbol>()},
                             // negative, positive (abs makes them equal)
                             {-4, 4, 4, 4, std::make_shared<ov::Symbol>()},
                             // negative, positive (abs makes lower bigger than upper)
                             {-4, 3, 3, 4, std::make_shared<ov::Symbol>()},
                             // negative, positive (abs keeps upper bigger than lower)
                             {-3, 4, 3, 4, std::make_shared<ov::Symbol>()},
                             //     zero, zero
                             {0, 0, 0, 0, std::make_shared<ov::Symbol>()},
                             //     zero, positive
                             {0, 2, 0, 2, std::make_shared<ov::Symbol>()},
                             // positive, positive (equal)
                             {1, 1, 1, 1, std::make_shared<ov::Symbol>()},
                             // positive, positive (not equal)
                             {1, 42, 1, 42, std::make_shared<ov::Symbol>()},
                         }}));
