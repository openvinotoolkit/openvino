// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include "single_layer_tests/op_impl_check/op_impl_check.hpp"
#include "common_test_utils/crash_handler.hpp"

namespace ov {
namespace test {
namespace subgraph {

void OpImplCheckTest::SetUp() {
    std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>> funcInfo;
    std::tie(funcInfo, targetDevice, configuration) = this->GetParam();
    function = funcInfo.second;
}

std::string OpImplCheckTest::getTestCaseName(const testing::TestParamInfo<OpImplParams> &obj) {
    std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>> funcInfo;
    std::string targetDevice;
    ov::AnyMap config;
    std::tie(funcInfo, targetDevice, config) = obj.param;

    std::ostringstream result;
    std::string friendlyName = funcInfo.first.name + std::string("_") + funcInfo.first.get_version();
    result << "Function=" << friendlyName << "_";
    result << "Device=" << targetDevice << "_";
    result << "Config=(";
    for (const auto& configItem : config) {
        result << configItem.first << "=";
        configItem.second.print(result);
        result << "_";
    }
    result << ")";
    return result.str();
}

}   // namespace subgraph
}   // namespace test
}   // namespace ov
