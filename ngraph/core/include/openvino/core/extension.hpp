// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/so_loader.hpp"

// Use extern "C""Catching" Failures in order to avoid issues with mangling
#if defined(_WIN32) && defined(IMPLEMENT_OPENVINO_EXTENSION_API)
#    define OPENVINO_EXTENSION_API extern "C" __declspec(dllexport)
#else
#    define OPENVINO_EXTENSION_API extern "C" OPENVINO_API
#endif

namespace ov {

class Extension;

OPENVINO_API
void setExtensionSharedObject(const std::shared_ptr<Extension>&, const SOLoader&);

class OPENVINO_API Extension {
public:
    using Ptr = std::shared_ptr<Extension>;
    virtual ~Extension();

private:
    SOLoader so;

    friend void setExtensionSharedObject(const Extension::Ptr&, const SOLoader&);
};

template <typename C,
          typename = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type>
OPENVINO_API inline std::vector<Extension::Ptr> load_extension(const std::basic_string<C>& path) {
    SOLoader so(path);
    using CreateFunction = void(std::vector<Extension::Ptr>&);
    std::vector<Extension::Ptr> extensions;
    reinterpret_cast<CreateFunction*>(so.get_symbol("create_extensions"))(extensions);

    for (auto&& ex : extensions) {
        setExtensionSharedObject(ex, so);
    }
    return extensions;
}

OPENVINO_EXTENSION_API
void create_extensions(std::vector<Extension::Ptr>&);

#define OPENVINO_CREATE_EXTENSIONS(extensions)                             \
    OPENVINO_EXTENSION_API                                                 \
    void ::ov::create_extensions(std::vector<::ov::Extension::Ptr>& ext) { \
        ext = extensions;                                                  \
    }

}  // namespace ov
