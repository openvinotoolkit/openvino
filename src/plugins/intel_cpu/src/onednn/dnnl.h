// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <dnnl.hpp>

namespace dnnl {

using primitive_desc_iterator = dnnl::primitive_desc;

namespace utils {

unsigned get_cache_size(int level, bool per_core);

const char* fmt2str(memory::format_tag fmt);
dnnl::memory::format_tag str2fmt(const char* str);

}  // namespace utils
}  // namespace dnnl
