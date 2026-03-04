// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <filesystem>

#include "openvino/core/any.hpp"
#include "openvino/util/file_util.hpp"

namespace ov::frontend {

/// @brief Extracts std::filesystem::path from ov::Any containing path, string, or wstring.
/// @return Path if conversion succeeded, empty path otherwise.
inline std::optional<std::filesystem::path> get_path_from_any(const ov::Any& param) {
    if (param.is<std::filesystem::path>()) {
        return std::make_optional(param.as<std::filesystem::path>());
    } else if (param.is<std::string>()) {
        return std::make_optional(ov::util::make_path(param.as<std::string>()));
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
    } else if (param.is<std::wstring>()) {
        return std::make_optional(ov::util::make_path(param.as<std::wstring>()));
#endif
    } else {
        return std::nullopt;
    }
}

}  // namespace ov::frontend
