// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"

#define EXPECT_HAS_SUBSTRING(haystack, needle) EXPECT_PRED_FORMAT2(testing::IsSubstring, needle, haystack)

struct PrintToDummyParamName {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
        return "dummy" + std::to_string(info.index);
    }
};

std::vector<size_t> get_shape_labels(const ov::PartialShape& p_shape);

void set_shape_labels(ov::PartialShape& p_shape, const std::vector<size_t>& labels);

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

using BoundTestParam = std::tuple<ov::PartialShape, ov::PartialShape>;

/** \brief Test fixture for Unsqueeze/Squeeze type_prop bound tests. */
class UnSqueezeBoundTest : public testing::WithParamInterface<BoundTestParam>, public UnSqueezeFixture {
protected:
    void SetUp() override {
        std::tie(p_shape, exp_shape) = GetParam();
        param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1});
    }

    std::vector<size_t> in_labels;
};
