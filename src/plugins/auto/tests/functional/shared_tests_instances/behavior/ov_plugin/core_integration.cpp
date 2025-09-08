// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/properties.hpp>

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

// defined in plugin_name.cpp
extern const char* cpu_plugin_file_name;

namespace {
//
// OV Class Common tests with <pluginName, deviceName params>
//

const std::vector<ov::AnyMap> configsWithEmpty = {{}};
const std::vector<ov::AnyMap> configsWithMetaPlugin = {{ov::device::priorities("AUTO")},
                                                       {ov::device::priorities("MULTI")},
                                                       {ov::device::priorities("AUTO", "MULTI")},
                                                       {ov::device::priorities("AUTO", "TEMPLATE")},
                                                       {ov::device::priorities("MULTI", "TEMPLATE")}};

INSTANTIATE_TEST_SUITE_P(
    smoke_MULTI_AUTO_DoNotSupportMetaPluginLoadingItselfRepeatedlyWithEmptyConfigTest,
    OVClassCompileModelWithCondidateDeviceListContainedMetaPluginTest,
    ::testing::Combine(::testing::Values("MULTI:AUTO", "AUTO:MULTI", "MULTI:AUTO,TEMPLATE", "AUTO:TEMPLATE,MULTI"),
                       ::testing::ValuesIn(configsWithEmpty)),
    ::testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(smoke_MULTI_AUTO_DoNotSupportMetaPluginLoadingItselfRepeatedlyTest,
                         OVClassCompileModelWithCondidateDeviceListContainedMetaPluginTest,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(configsWithMetaPlugin)),
                         ::testing::PrintToStringParamName());

// Several devices case
/* enable below in nightly tests*/
/*
INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestCompileModel,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestQueryModel,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));

INSTANTIATE_TEST_SUITE_P(nightly_OVClassSeveralDevicesTest,
                         OVClassSeveralDevicesTestDefaultCore,
                         ::testing::Values(std::vector<std::string>({"GPU.0", "GPU.1"})));
*/
}  // namespace
