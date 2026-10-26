//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/except.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

[[noreturn]] void intel_npu::vm::detail::throwRuntimeError([[maybe_unused]] const char* file, [[maybe_unused]] int line,
                                                           const std::string& message) {
    std::ostringstream stream;
    stream
#ifdef VPUX_DEVELOPER_BUILD
            << file << ":" << line << ": "
#endif
            << message;
    throw std::runtime_error(stream.str());
}
