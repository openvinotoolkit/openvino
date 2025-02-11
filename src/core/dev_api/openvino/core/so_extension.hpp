// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/extension.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"

namespace ov {
namespace detail {

class OPENVINO_API SOExtension : public Extension {
public:
    virtual ~SOExtension() override;

    SOExtension(const Extension::Ptr& ext, const std::shared_ptr<void>& so) : m_ext(ext), m_so(so) {}

    const Extension::Ptr& extension() const;

    const std::shared_ptr<void> shared_object() const;

private:
    Extension::Ptr m_ext;
    std::shared_ptr<void> m_so;
};

inline std::string resolve_extension_path(const std::string& path) {
    std::string retvalue;
    try {
        const std::string absolute_path = ov::util::get_absolute_file_path(path);
        retvalue = ov::util::file_exists(absolute_path) ? absolute_path : path;
    } catch (const std::runtime_error&) {
        retvalue = path;
    }
    return retvalue;
}

inline std::vector<Extension::Ptr> load_extensions(const std::string& path) {
    const std::string resolved_path = resolve_extension_path(path);
    auto so = ov::util::load_shared_object(resolved_path.c_str());
    using CreateFunction = void(std::vector<Extension::Ptr>&);
    std::vector<Extension::Ptr> extensions;
    reinterpret_cast<CreateFunction*>(ov::util::get_symbol(so, "create_extensions"))(extensions);

    std::vector<Extension::Ptr> so_extensions;
    so_extensions.reserve(extensions.size());

    for (auto&& ex : extensions) {
        so_extensions.emplace_back(std::make_shared<SOExtension>(ex, so));
    }
    return so_extensions;
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
inline std::vector<Extension::Ptr> load_extensions(const std::wstring& path) {
    return load_extensions(ov::util::wstring_to_string(path).c_str());
}
#endif

}  // namespace detail
}  // namespace ov
