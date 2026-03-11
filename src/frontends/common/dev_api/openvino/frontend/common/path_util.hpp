// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <optional>

#include "openvino/core/any.hpp"
#include "openvino/util/file_util.hpp"

namespace ov::frontend {

/// @brief Extracts std::filesystem::path from ov::Any containing path, string, or wstring.
/// @return Path if conversion succeeded, std::nullopt otherwise.
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

/// @brief Extracts std::vector<std::filesystem::path> from ov::Any containing
/// std::vector<std::string>, std::vector<std::wstring>, or std::vector<std::filesystem::path>.
/// @return Vector of paths if conversion succeeded, std::nullopt otherwise.
inline std::optional<std::vector<std::filesystem::path>> get_path_vec_from_any(const ov::Any& param) {
    if (param.is<std::vector<std::filesystem::path>>()) {
        return std::make_optional(param.as<std::vector<std::filesystem::path>>());
    } else if (param.is<std::vector<std::string>>()) {
        const auto& string_vec = param.as<std::vector<std::string>>();
        std::vector<std::filesystem::path> path_vec;
        path_vec.reserve(string_vec.size());
        for (const auto& str : string_vec) {
            path_vec.emplace_back(ov::util::make_path(str));
        }
        return std::make_optional(path_vec);
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
    } else if (param.is<std::vector<std::wstring>>()) {
        const auto& wstring_vec = param.as<std::vector<std::wstring>>();
        std::vector<std::filesystem::path> path_vec;
        path_vec.reserve(wstring_vec.size());
        for (const auto& wstr : wstring_vec) {
            path_vec.emplace_back(ov::util::make_path(wstr));
        }
        return std::make_optional(path_vec);
#endif
    } else {
        return std::nullopt;
    }
}

}  // namespace ov::frontend
