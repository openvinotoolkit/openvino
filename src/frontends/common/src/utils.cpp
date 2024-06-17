// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "openvino/util/file_util.hpp"

namespace ov {
namespace frontend {

std::vector<ov::Any> to_wstring_if_needed(const std::vector<ov::Any>& variants) {
    if (variants[0].is<std::string>()) {
        auto model_path = variants[0].as<std::string>();
// Fix unicode name
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        std::wstring model_path_wstr = ov::util::string_to_wstring(model_path.c_str());
#else
        std::string model_path_wstr = model_path;
#endif
        ov::AnyVector params{model_path_wstr};
        return params;
    } else {
        return variants;
    }
}

}  // namespace frontend
}  // namespace ov