// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"

#include <openvino/runtime/properties.hpp>

#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

// defined in plugin_name.cpp
extern const char * cpu_plugin_file_name;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, OVClassImportExportTestP,
        ::testing::Values("HETERO:CPU"));

// IE Class Query model
INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest, OVClassQueryModelTest, ::testing::Values("CPU"));
}  // namespace
