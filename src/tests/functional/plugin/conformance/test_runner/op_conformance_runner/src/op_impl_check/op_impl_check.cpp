// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <signal.h>
#ifdef _WIN32
#include <process.h>
#endif

#include "op_impl_check/op_impl_check.hpp"
#include "functional_test_utils/crash_handler.hpp"

#include "common_test_utils/postgres_link.hpp"

namespace ov {
namespace test {
namespace subgraph {

void OpImplCheckTest::SetUp() {
    std::pair<ov::DiscreteTypeInfo, std::shared_ptr<ov::Model>> funcInfo;
    std::tie(funcInfo, targetDevice, configuration) = this->GetParam();
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
