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
        if (this->targetDevice == "CPU") {
            pgLink->set_custom_field("targetDevice", this->targetDevice, true);
#    if defined(OPENVINO_ARCH_X86_64)
            pgLink->set_custom_field("targetDeviceArch", "x64", true);
#    elif defined(OPENVINO_ARCH_X86)
            pgLink->set_custom_field("targetDeviceArch", "x86", true);
#    elif defined(OPENVINO_ARCH_ARM)
            pgLink->set_custom_field("targetDeviceArch", "arm", true);
#    elif defined(OPENVINO_ARCH_ARM64)
            pgLink->set_custom_field("targetDeviceArch", "arm64", true);
#    elif defined(OPENVINO_ARCH_RISCV64)
            pgLink->set_custom_field("targetDeviceArch", "riskv64", true);
#    endif
        } else if (this->targetDevice == "GPU") {
            auto devName = core->get_property("GPU", "FULL_DEVICE_NAME").as<std::string>();
            std::cerr << "GPU Device: " << devName << std::endl;
            if (devName.find("dGPU") != std::string::npos) {
                pgLink->set_custom_field("targetDevice", "DGPU", true);
            } else {
                pgLink->set_custom_field("targetDevice", this->targetDevice, true);
            }
        } else {
            pgLink->set_custom_field("targetDevice", this->targetDevice, true);
        }
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
