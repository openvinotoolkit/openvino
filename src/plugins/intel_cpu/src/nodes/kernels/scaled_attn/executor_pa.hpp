// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <openvino/core/type/element_type.hpp>

#include "executor_pa_common.hpp"

namespace ov::Extensions::Cpu::XARCH {

std::shared_ptr<PagedAttentionExecutor> make_pa_executor(ov::element::Type data_type,
                                                         ov::element::Type key_cache_type,
                                                         ov::element::Type value_cache_type,
                                                         const PagedAttnQuantParams& params);

}  // namespace ov::Extensions::Cpu::XARCH
