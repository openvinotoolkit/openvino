// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace ov {
namespace detail {

inline std::vector<Extension::Ptr> load_extensions(const std::string& path) {
    auto so = ov::util::load_shared_object(path.c_str());
    using CreateFunction = void(std::vector<Extension::Ptr>&);
    std::vector<Extension::Ptr> extensions;
    reinterpret_cast<CreateFunction*>(ov::util::get_symbol(so, "create_extensions"))(extensions);

    for (auto&& ex : extensions) {
        ex->so = so;
    }
    return extensions;
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
inline std::vector<Extension::Ptr> load_extensions(const std::wstring& path) {
    return load_extensions(ov::util::wstring_to_string(path).c_str());
}
#endif

inline void unload_extensions(std::vector<Extension::Ptr>& extensions) {
    std::vector<std::shared_ptr<void>> shared_objects;
    shared_objects.reserve(extensions.size());

    for (auto&& ex : extensions) {
        shared_objects.emplace_back(ex->so);
    }
    for (auto&& ex : extensions) {
        // TODO: Should we check here that there is no other references to ex left? Hard to debug otherwise.
        ex.reset();
    }
}

}  // namespace detail
}  // namespace ov
