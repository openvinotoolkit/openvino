// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <openvino/core/type/element_type.hpp>
#include "cpu_memory.h"
#include "executor_pa_common.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

std::shared_ptr<PagedAttentionExecutor> make_pa_executor(ov::element::Type data_type, ov::element::Type kvcache_type);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov