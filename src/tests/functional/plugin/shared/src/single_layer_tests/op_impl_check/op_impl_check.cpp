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

void OpImplCheckTest::run() {
    if (function == nullptr) {
        GTEST_FAIL() << "Target function is empty!";
    }

    // in case of crash save report and finish work
    auto crashHandler = std::unique_ptr<CommonTestUtils::CrashHandler>(new CommonTestUtils::CrashHandler(false));
    crashHandler->StartTimer();

    summary.setDeviceName(targetDevice);
    try {
        auto executableNetwork = core->compile_model(function, targetDevice, configuration);
        summary.updateOPsImplStatus(function, true);
    } catch (...) {
        summary.updateOPsImplStatus(function, false);
        GTEST_FAIL() << "Error in the LoadNetwork!";
    }
}

void OpImplCheckTest::SetUp() {
    std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>> funcInfo;
    std::tie(funcInfo, targetDevice, configuration) = this->GetParam();
    function = funcInfo.second;
    auto &s = LayerTestsUtils::Summary::getInstance();
    std::cout << "OPS LIST:";
    for (const auto &op : function->get_ordered_ops()) {
        if ((ngraph::is_type<ngraph::op::Parameter>(op) ||
            ngraph::is_type<ngraph::op::Constant>(op) ||
            ngraph::is_type<ngraph::op::Result>(op))) {
            continue;
        }
        std::cout << " " << s.getOpName(op->get_type_info());
    }
    std::cout << "\n";
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

TEST_P(OpImplCheckTest, checkPluginImplementation) {
    run();
}

}   // namespace subgraph
}   // namespace test
}   // namespace ov
