// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"

#define EXPECT_HAS_SUBSTRING(haystack, needle) EXPECT_PRED_FORMAT2(testing::IsSubstring, needle, haystack)

struct PrintToDummyParamName {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
        return "dummy" + std::to_string(info.index);
    }
};

/**
 * \brief Infinite generator of sequence increasing values.
 *
 * Start value can be specified.
 *
 * \tparam T Type of sequence values (must support `++` operator).
 */
template <class T>
class SeqGen {
    T _counter;

public:
    SeqGen(const T& start) : _counter{start} {}

    T operator()() {
        return _counter++;
    }
};
