// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"

#define EXPECT_HAS_SUBSTRING(haystack, needle) EXPECT_PRED_FORMAT2(testing::IsSubstring, needle, haystack)

struct PrintToDummyParamName {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
        return "dummy" + std::to_string(info.index);
    }
};

/**
 * \brief Set symbols on all shape dimensions start from first symbol.
 *
 * \param p_shape      Shape to set symbols.
 * \return vector of set symbols
 */
ov::TensorSymbol get_shape_symbols(const ov::PartialShape& p_shape);

/**
 * \brief Set symbols on all shape dimensions start from first symbol.
 *
 * \param p_shape      Shape to set symbols.
 */
ov::TensorSymbol set_shape_symbols(ov::PartialShape& p_shape);
void set_shape_symbols(ov::PartialShape& p_shape, const ov::TensorSymbol& symbols);

/**
 * \brief Test fixture for Unsqueeze/Squeeze type_prop tests.
 */
class UnSqueezeFixture : public testing::Test {
protected:
    void SetUp() override {
        param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, p_shape);
    }

    ov::PartialShape p_shape, exp_shape;
    std::shared_ptr<ov::op::v0::Parameter> param;
};

/** \brief Test fixture for Unsqueeze/Squeeze type_prop bound tests. */
class UnSqueezeBoundTest : public testing::WithParamInterface<std::tuple<ov::PartialShape, ov::PartialShape>>,
                           public UnSqueezeFixture {
protected:
    void SetUp() override {
        std::tie(p_shape, exp_shape) = GetParam();
        param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    }

    template <typename T>
    auto create_squeeze(ov::PartialShape symboled_shape) -> std::shared_ptr<T> {
        constexpr auto et = ov::element::i64;
        const auto symboled_param = std::make_shared<ov::op::v0::Parameter>(et, symboled_shape);
        const auto symboled_shape_of = std::make_shared<ov::op::v0::ShapeOf>(symboled_param);

        const auto zero = std::vector<int64_t>{0};
        const auto axis = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, zero);
        const auto indices = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{}, zero);
        const auto gather = std::make_shared<ov::op::v7::Gather>(symboled_shape_of, indices, axis);
        const auto axis_1 = std::make_shared<ov::op::v0::Constant>(et, ov::Shape{2}, std::vector<int64_t>{0, 1});
        const auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(gather, axis_1);
        const auto squeeze = std::make_shared<T>(unsqueeze, axis);

        return squeeze;
    }

    ov::TensorSymbol in_symbols;
};

using PartialShapes = std::vector<ov::PartialShape>;
using Shapes = std::vector<ov::Shape>;

template <class TOp>
class TypePropOpTest : public testing::Test {
protected:
    template <class... Args>
    std::shared_ptr<TOp> make_op(Args&&... args) {
        return std::make_shared<TOp>(std::forward<Args>(args)...);
    }
};
