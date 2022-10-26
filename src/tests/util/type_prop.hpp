// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gmock/gmock.h"
#include "openvino/core/partial_shape.hpp"
#include "openvino/util/pp.hpp"

#define EXPECT_HAS_SUBSTRING(haystack, needle) EXPECT_PRED_FORMAT2(testing::IsSubstring, needle, haystack)

#define OV_EXPECT_THROW(statement, exception, exception_what_matcher) \
    try {                                                             \
        statement;                                                    \
        FAIL() << "Expected exception " << OV_PP_TOSTRING(exception); \
    } catch (const exception& ex) {                                   \
        EXPECT_THAT(ex.what(), exception_what_matcher);               \
    } catch (...) {                                                   \
        FAIL() << "Unknown exception";                                \
    }

struct PrintToDummyParamName {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
        return "dummy" + std::to_string(info.index);
    }
};

std::vector<size_t> get_shape_labels(const ov::PartialShape& p_shape);

void set_shape_labels(ov::PartialShape& p_shape, const std::vector<size_t>& labels);
