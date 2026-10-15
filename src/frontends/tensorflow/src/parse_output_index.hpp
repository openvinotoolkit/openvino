// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>

namespace ov {
namespace frontend {
namespace tensorflow {

// Strict integer parser for the output-index suffix of a TF node reference
// like "RestoreV2:3". Returns nullopt for empty / non-numeric / trailing
// garbage / overflow / out-of-int-range tokens; the caller decides how to
// react (e.g. throw a frontend exception). Uses strtoll + std::int64_t so
// the int-range check is meaningful on platforms where long aliases int.
inline std::optional<int> parse_output_index(const std::string& token) {
    if (token.empty()) {
        return std::nullopt;
    }
    // strtoll otherwise silently skips leading whitespace; well-formed TF node
    // references never contain it, so reject it as malformed up front.
    if (std::isspace(static_cast<unsigned char>(token.front()))) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    constexpr auto base = 10;
    const std::int64_t v = std::strtoll(token.c_str(), &end, base);
    if (end == token.c_str() || *end != '\0' || errno == ERANGE) {
        return std::nullopt;
    }
    if (v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        return std::nullopt;
    }
    return static_cast<int>(v);
}

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
