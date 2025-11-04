// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ze_api.h>  // not redundant, needed for `ze_structure_type_t` structure
#include <ze_mem_import_system_memory_ext.h>

#include <behavior/ov_plugin/caching_tests.hpp>

#include "intel_npu/utils/zero/zero_init.hpp"
#include "openvino/core/log_util.hpp"

namespace ov {
namespace test {
namespace behavior {

using OVCompileModelLoadFromFileTestBaseNPU = CompileModelLoadFromFileTestBase;

TEST_P(OVCompileModelLoadFromFileTestBaseNPU, BlobWithOVHeaderAligmentCanBeImported) {
    core->set_property(ov::cache_dir(m_cacheFolderName));

    ze_device_external_memory_properties_t externalMemorydDesc = {};
    externalMemorydDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;

    auto res =
        intel_npu::zeDeviceGetExternalMemoryProperties(intel_npu::ZeroInitStructsHolder::getInstance()->getDevice(),
                                                       &externalMemorydDesc);
    if ((res != ZE_RESULT_SUCCESS) ||
        ((externalMemorydDesc.memoryAllocationImportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_STANDARD_ALLOCATION) == 0)) {
        GTEST_SKIP() << "Standard allocation is not supported by the current configuration.";
    }

    std::stringstream custom_logger;
    ov::util::LogCallback custom_log_callback =
        [&](std::string_view s) {  // switch to query allocation info for import flag when possible
            custom_logger << s << std::endl;
        };
    ov::util::set_log_callback(custom_log_callback);

    for (size_t i = 0; i < 2; ++i) {
        if (i != 0) {
            configuration.emplace(ov::log::level(ov::log::Level::DEBUG));
        }
        std::ignore = core->compile_model(m_modelName, targetDevice, configuration);
        configuration.erase(ov::log::level.name());
    }
    ov::util::reset_log_callback();
    EXPECT_THAT(custom_logger.str(),
                ::testing::HasSubstr("getGraphDescriptor - set ZE_GRAPH_FLAG_INPUT_GRAPH_PERSISTENT"));
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
