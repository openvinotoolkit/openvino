// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/extension.hpp"

#include <iostream>

#include "openvino/core/except.hpp"
#include "openvino/util/shared_object.hpp"

using namespace ov;

namespace {

class SOExtension : public ov::Extension {
public:
    SOExtension(const std::shared_ptr<void>& lib, const ov::Extension::Ptr& ext) : so(lib), extension(ext) {}
    const Extension::Ptr& getExtension() const {
        return extension;
    }

private:
    std::shared_ptr<void> so;
    ov::Extension::Ptr extension;
};

template <class C>
inline std::vector<Extension::Ptr> load_extension_impl(const std::basic_string<C>& path) {
    auto so = ov::util::load_shared_object(path.c_str());
    using CreateFunction = void(std::vector<Extension::Ptr>&);
    std::vector<Extension::Ptr> extensions;
    reinterpret_cast<CreateFunction*>(ov::util::get_symbol(so, "create_extensions"))(extensions);

    std::vector<Extension::Ptr> result;
    result.reserve(extensions.size());
    for (auto&& ex : extensions) {
        result.emplace_back(std::make_shared<SOExtension>(so, ex));
    }
    return result;
}
}  // namespace

Extension* ov::_get_extension(Extension* extension) {
    if (const auto* so_extension = dynamic_cast<SOExtension*>(extension))
        return ov::_get_extension(so_extension->getExtension().get());
    return extension;
}

Extension::Ptr ov::_get_extension(Extension::Ptr extension) {
    if (const auto so_extension = std::dynamic_pointer_cast<SOExtension>(extension))
        return ov::_get_extension(so_extension->getExtension());
    return extension;
}

ov::Extension::~Extension() = default;

std::vector<Extension::Ptr> ov::load_extension(const std::string& path) {
    return load_extension_impl(path);
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT
std::vector<Extension::Ptr> ov::load_extension(const std::wstring& path) {
    return load_extension_impl(path);
}
#endif
