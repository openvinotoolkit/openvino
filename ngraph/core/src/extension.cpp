// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <iostream>

#include "so_loader.hpp"

using namespace ov;

namespace {
template <class C>
inline std::vector<Extension::Ptr> load_extension_impl(const std::basic_string<C>& path) {
    auto so = std::make_shared<SOLoader>(path);
    using CreateFunction = void(std::vector<Extension::Ptr>&);
    std::vector<Extension::Ptr> extensions;
    reinterpret_cast<CreateFunction*>(so->get_symbol("create_extensions"))(extensions);

    for (auto&& ex : extensions) {
        set_extension_shared_object(ex, so);
    }
    return extensions;
}
}  // namespace

ov::Extension::~Extension() {
    std::cout << "AAAA " << std::endl;
    so.reset();
}

void ov::set_extension_shared_object(const Extension::Ptr& extension, const std::shared_ptr<SOLoader>& so) {
    extension->so = so;
}

std::vector<Extension::Ptr> ov::load_extension(const std::string& path) {
    return load_extension_impl(path);
}

std::vector<Extension::Ptr> ov::load_extension(const std::wstring& path) {
    return load_extension_impl(path);
}
