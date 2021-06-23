// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
/**
 * @brief Base test class for Per Plugin tests
 *
 * Helper to handle test cases for all Plugins.
 * @file
 */

#include <gtest/gtest.h>
#include <cstddef>
#include <string>
#include <tuple>
#include <tests_common.hpp>
#include <ie_core.hpp>

/**
 * @brief Container for plugin_name and test params
 *
 * plugin_name is mandatory field.
 */
template<typename P>
using PlgTestParam = std::tuple<std::string, P>;

/**
 * @brief Base class for per plugin tests
 */
template<typename P = std::nullptr_t>
class PlgTest : public testing::TestWithParam<PlgTestParam<P>> {
protected:
    std::map<std::string, std::string>  config;
    virtual void SetUp() {
        device_name = std::get<0>(this->GetParam());
        std::transform(device_name.begin(), device_name.end(), 
            device_name.begin(), [] (char v) { return v == '_' ? ':' : v; });
    }

    const P &param() const {
        return std::get<1>(this->GetParam());
    }

    std::string device_name;
};

/**
 * @brief Helper to print name
 */
template<typename P>
class Named {
public:
    Named(std::function<std::string(P)> clb) : _clb(clb) {}

    const std::string operator() (const testing::TestParamInfo<PlgTestParam<P>> &p) {
        return _clb(std::get<1>(p.param));
    }
private:
    const std::function<std::string(P)> _clb;
};

/**
 * @brief Macros to specify Per Plugin Run Test Case with parameters.
 */
#define RUN_CASE_P_WITH_SUFFIX(_plugin, _suffix, _test, _params) \
    INSTANTIATE_TEST_SUITE_P(_plugin##_run##_suffix, _test, ::testing::Combine(::testing::Values(#_plugin), ::testing::ValuesIn(_params) ))

/**
 * @brief Macros to specify Per Plugin Run Test Case with Cartesian Product of parameters.
 */
#define RUN_CASE_CP_WITH_SUFFIX(_plugin, _suffix, _test, _params, ...) \
    INSTANTIATE_TEST_SUITE_P(_plugin##_run##_suffix, _test, ::testing::Combine(::testing::Values(#_plugin), _params), __VA_ARGS__ )
