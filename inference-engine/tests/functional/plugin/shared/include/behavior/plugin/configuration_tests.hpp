// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <common_test_utils/test_common.hpp>
#include <ie_core.hpp>
#include <ie_parameter.hpp>

using CustomComparator = std::function<bool(const InferenceEngine::Parameter&, const InferenceEngine::Parameter&)>;

struct DefaultParameter {
    std::string                _key;
    InferenceEngine::Parameter _parameter;
    CustomComparator           _comparator;
};

using DefaultConfigurationParameters = std::tuple<
    std::string,    //  device name
    DefaultParameter // default parameter key value comparator
>;

struct DefaultConfigurationTest : public CommonTestUtils::TestsCommon, public ::testing::WithParamInterface<DefaultConfigurationParameters> {
    enum {DeviceName, DefaultParamterId};
    static std::string getTestCaseName(const ::testing::TestParamInfo<DefaultConfigurationParameters> &obj);

protected:
    ~DefaultConfigurationTest() override = default;
    std::unique_ptr<InferenceEngine::Core> _core;

    void SetUp() override {
        _core.reset(new InferenceEngine::Core);
    }
    void TearDown() override {
        _core.reset();
    }
};
