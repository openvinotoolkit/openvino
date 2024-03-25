// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"

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
