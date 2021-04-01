// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn.hpp"

namespace mkldnn {

using primitive_desc_iterator = mkldnn::primitive_desc;

namespace utils {

int get_cache_size(int level, bool per_core);

const char* fmt2str(memory::format_tag fmt);
mkldnn::memory::format_tag str2fmt(const char *str);

}  // namespace utils
}  // namespace mkldnn
