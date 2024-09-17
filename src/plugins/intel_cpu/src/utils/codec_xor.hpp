// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>

namespace ov {
namespace intel_cpu {

void codec_xor(char* dst_str, const char* src_str, size_t len);

std::string codec_xor_str(const std::string& source_str);

}  // namespace intel_cpu
}  // namespace ov
