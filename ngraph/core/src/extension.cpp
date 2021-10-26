// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <iostream>

#include "openvino/core/except.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

using namespace ov;

std::vector<Extension> ov::load_extension(const std::string& path) {
    auto so = ov::util::load_shared_object(path.c_str());
    using CreateFunction = void(std::vector<BaseExtension::Ptr>&);
    std::vector<BaseExtension::Ptr> extensions;
    reinterpret_cast<CreateFunction*>(ov::util::get_symbol(so, "create_extensions"))(extensions);

    std::vector<Extension> result;
    result.reserve(extensions.size());
    for (auto&& ex : extensions) {
        result.emplace_back(Extension(ex, so));
    }
    return result;
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::vector<Extension> ov::load_extension(const std::wstring& path) {
    return load_extension(ov::util::wstring_to_string(path).c_str());
}
#endif

ov::BaseExtension::~BaseExtension() = default;

ov::Extension::Extension(ov::BaseExtension::Ptr ext, std::shared_ptr<void> so) : so(so), ext(std::move(ext)) {}

const ov::BaseExtension::Ptr& ov::Extension::get() const {
    OPENVINO_ASSERT(ext != nullptr, "Extension doesn't contain pointer to base extension.");
    return ext;
}
