// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>

#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {

OPENVINO_API
Tensor create_mmaped_tensor(const Tensor& tensor, const std::string& file_name);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
OPENVINO_API
Tensor create_mmaped_tensor(const Tensor& tensor, const std::wstring& file_name);
#endif

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
OPENVINO_API
Tensor create_mmaped_tensor(const Tensor& tensor, const std::filesystem::path& file_name);
#endif

}  // namespace ov