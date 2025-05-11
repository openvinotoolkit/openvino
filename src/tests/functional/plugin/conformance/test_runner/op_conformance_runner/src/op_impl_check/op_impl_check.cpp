// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include "common_test_utils/postgres_link.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/crash_handler.hpp"

#include "op_impl_check/op_impl_check.hpp"
#include "op_impl_check/single_op_graph.hpp"

#include "conformance.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {
namespace op_conformance {

void OpImplCheckTest::SetUp() {
    std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>> funcInfo;
    targetDevice = ov::test::utils::target_device;
    configuration = ov::test::utils::global_plugin_config;
    funcInfo = this->GetParam();
    function = funcInfo.second;

#ifdef ENABLE_CONFORMANCE_PGQL
    // Updating data in runtime. Should be set before possible call of a first GTEST status
    auto pgLink = this->GetPGLink();
    if (pgLink) {
        auto devNameProperty = core->get_property(this->targetDevice, "FULL_DEVICE_NAME");
        auto devName = devNameProperty.is<std::string>() ? devNameProperty.as<std::string>() : "";
        pgLink->set_custom_field("targetDeviceName", devName, true);
        if (this->targetDevice == "CPU") {
            pgLink->set_custom_field("targetDevice", this->targetDevice, true);
            pgLink->set_custom_field("targetDeviceArch", devName.find("ARM") != std::string::npos ? "arm" : "", true);
        } else if (this->targetDevice == "GPU") {
            if (devName.find("dGPU") != std::string::npos) {
                pgLink->set_custom_field("targetDevice", "DGPU", true);
            } else {
                pgLink->set_custom_field("targetDevice", this->targetDevice, true);
            }
        } else {
            pgLink->set_custom_field("targetDevice", this->targetDevice, true);
        }
        pgLink->manual_start();
    }
#endif
}

std::string OpImplCheckTest::getTestCaseName(const testing::TestParamInfo<OpImplParams> &obj) {
    std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>> funcInfo;
    std::string targetDevice = ov::test::utils::target_device;
    ov::AnyMap config = ov::test::utils::global_plugin_config;
    funcInfo = obj.param;

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
    if (function == nullptr) {
        GTEST_FAIL() << "Target model is empty!";
    }
    summary.updateOPsImplStatus(function, false);

    // in case of crash jump will be made and work will be continued
    ov::test::utils::CrashHandler crashHandler;
    // place to jump in case of a crash
    int jmpRes = 0;
#ifdef _WIN32
    jmpRes = setjmp(ov::test::utils::env);
#else
    jmpRes = sigsetjmp(ov::test::utils::env, 1);
#endif
    if (jmpRes == ov::test::utils::JMP_STATUS::ok) {
        crashHandler.StartTimer();
        summary.setDeviceName(targetDevice);
        try {
            auto queryNetworkResult = core->query_model(function, targetDevice);
            {
                std::set<std::string> expected;
                for (auto &&node : function->get_ops()) {
                    expected.insert(node->get_friendly_name());
                }

                std::set<std::string> actual;
                for (auto &&res : queryNetworkResult) {
                    actual.insert(res.first);
                }

                if (expected == actual) {
                    summary.updateOPsImplStatus(function, true);
                }
            }
        } catch (const std::exception &e) {
            GTEST_FAIL() << "Exception in the Core::compile_model() method call: " << e.what();
        } catch (...) {
            GTEST_FAIL() << "Error in the Core::compile_model() method call!";
        }
    } else if (jmpRes == ov::test::utils::JMP_STATUS::anyError) {
        GTEST_FAIL() << "Crash happens";
    } else if (jmpRes == ov::test::utils::JMP_STATUS::alarmErr) {
        GTEST_FAIL() << "Hang happens";
    }
}

namespace {
INSTANTIATE_TEST_SUITE_P(conformance,
                         OpImplCheckTest,
                         ::testing::ValuesIn(createFunctions()),
                         OpImplCheckTest::getTestCaseName);
}   // namespace
}   // namespace op_conformance
}   // namespace test
}   // namespace ov
