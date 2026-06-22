// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "openvino/core/except.hpp"

namespace cldnn {

inline void check_boundaries(size_t src_size,
                                size_t src_offset,
                                size_t dst_size,
                                size_t dst_offset,
                                size_t copy_size,
                                const std::string& func_str = "") {
    OPENVINO_ASSERT(src_offset + copy_size <= src_size && dst_offset + copy_size <= dst_size,
                    "[GPU] Incorrect buffer sizes for ",
                    func_str,
                    " call. ",
                    "Parameters provided are",
                    ": src_size=",
                    src_size,
                    ", src_offset=",
                    src_offset,
                    ", dst_size=",
                    dst_size,
                    ", dst_offset=",
                    dst_offset,
                    ", copy_size=",
                    copy_size,
                    ".");
}
}  // namespace cldnn
