// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/io.hpp"

namespace ov::util {

bool io_populate_mmap(void* /*ptr*/, size_t /*size*/, size_t /*offset*/, size_t /*queue_depth*/) noexcept {
    // CVS-186700
    return false;
}

std::error_code io_read_into(FileHandle /*handle*/, void* /*dst*/, size_t /*file_offset*/, size_t /*size*/, size_t /*queue_depth*/) noexcept {
    // CVS-186707
    return std::make_error_code(std::errc::function_not_supported);
}
}  // namespace ov::util
