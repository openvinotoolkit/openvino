// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/utils/zero/zero_types.hpp"

namespace intel_npu {

namespace test_constants {

inline constexpr ze_api_version_t TARGET_ZE_API_VERSION = ZE_API_VERSION_1_18;
inline constexpr uint32_t TARGET_ZE_DRIVER_NPU_EXT_VERSION = ZE_DRIVER_NPU_EXT_VERSION_1_0;
inline constexpr uint32_t TARGET_ZE_GRAPH_NPU_EXT_VERSION = ZE_GRAPH_EXT_VERSION_1_19;
inline constexpr uint32_t TARGET_ZE_COMMAND_QUEUE_NPU_EXT_VERSION = ZE_COMMAND_QUEUE_NPU_EXT_VERSION_1_1;
inline constexpr uint32_t TARGET_ZE_PROFILING_NPU_EXT_VERSION = ZE_PROFILING_DATA_EXT_VERSION_1_0;
inline constexpr uint32_t TARGET_ZE_CONTEXT_NPU_EXT_VERSION = ZE_CONTEXT_NPU_EXT_VERSION_1_0;
inline constexpr uint32_t TARGET_ZE_MUTABLE_COMMAND_LIST_EXT_VERSION = ZE_MUTABLE_COMMAND_LIST_EXP_VERSION_1_1;
inline constexpr uint32_t TARGET_ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION = ZE_EXTERNAL_MEMMAP_SYSMEM_EXT_VERSION_1_0;

inline constexpr std::string_view HostCompile_Interpreter = "HostCompile_Interpreter";

}  // namespace test_constants

}  // namespace intel_npu
