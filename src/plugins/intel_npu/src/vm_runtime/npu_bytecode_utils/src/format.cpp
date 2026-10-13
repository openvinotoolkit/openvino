//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/format.hpp"

#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

std::string intel_npu::vm::formatString(std::string_view format, const std::vector<std::string>& args) {
    std::ostringstream stream;
    size_t formatOffset = 0;
    size_t argIndex = 0;

    while (true) {
        const auto placeholderPos = format.find("{}", formatOffset);
        if (placeholderPos == std::string_view::npos) {
            stream << format.substr(formatOffset);
            break;
        }

        stream << format.substr(formatOffset, placeholderPos - formatOffset);
        if (argIndex < args.size()) {
            stream << args.at(argIndex++);
        } else {
            stream << "{}";
        }
        formatOffset = placeholderPos + 2;
    }

    while (argIndex < args.size()) {
        stream << ' ' << args.at(argIndex++);
    }

    return stream.str();
}
