// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

// Used as a custom test case printer for the google test.
// It will transform a string like "Normalized: true, NmsThreshold: 0.45"
//                            into "Normalized=true_NmsThreshold=0.45"
// to have it as a test case name.
// This parameter string will be printed when the test fails.
struct PrintTestCaseName {
    template <class ParamType>
    std::string operator()(const ::testing::TestParamInfo<ParamType>& info) const {
        auto paramsString = ::testing::PrintToString(info.param);

        const auto allowedChar = [](const unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '.' || c == ':' || c == ',';
        };
        const auto newEnd = std::remove_if(paramsString.begin(), paramsString.end(), std::not_fn(allowedChar));
        paramsString.erase(newEnd, paramsString.end());

        const auto isColon = [](const unsigned char c) {
            return c == ':';
        };
        std::replace_if(paramsString.begin(), paramsString.end(), isColon, '=');

        const auto isComma = [](const unsigned char c) {
            return c == ',';
        };
        std::replace_if(paramsString.begin(), paramsString.end(), isComma, '_');

        return paramsString;
    }
};
