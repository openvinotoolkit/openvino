// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/type.hpp"

// Use extern "C" in order to avoid issues with mangling
#if defined(_WIN32) && defined(IMPLEMENT_OPENVINO_EXTENSION_API)
#    define OPENVINO_EXTENSION_API extern "C" __declspec(dllexport)
#else
#    define OPENVINO_EXTENSION_API extern "C" OPENVINO_API
#endif

namespace ov {

class OPENVINO_API Extension {
public:
    using Ptr = std::shared_ptr<Extension>;
    virtual ~Extension();
};

OPENVINO_API std::vector<Extension::Ptr> load_extension(const std::string& path);
#ifdef ENABLE_UNICODE_PATH_SUPPORT
OPENVINO_API std::vector<Extension::Ptr> load_extension(const std::wstring& path);
#endif

OPENVINO_EXTENSION_API
void create_extensions(std::vector<Extension::Ptr>&);

OPENVINO_API Extension* _get_extension(Extension*);
OPENVINO_API Extension::Ptr _get_extension(Extension::Ptr);

template <class T,
          typename std::enable_if<std::is_base_of<Extension, typename std::remove_pointer<T>::type>::value,
                                  bool>::type = true>
typename std::remove_pointer<T>::type* as_type(Extension* ext) {
    auto* extension = _get_extension(ext);

    return dynamic_cast<typename std::remove_pointer<T>::type*>(extension);
}
template <class T,
          typename std::enable_if<std::is_base_of<Extension, typename std::remove_pointer<T>::type>::value,
                                  bool>::type = true>
typename std::remove_pointer<T>::type* as_type(Extension& ext) {
    return as_type<T>(&ext);
}
template <class T,
          typename std::enable_if<std::is_base_of<Extension, typename std::remove_pointer<T>::type>::value,
                                  bool>::type = true>
typename std::remove_pointer<T>::type* as_type(const Extension::Ptr& ext) {
    return as_type<T>(ext.get());
}

template <class T, typename std::enable_if<std::is_base_of<Extension, T>::value, bool>::type = true>
typename std::shared_ptr<T> as_type_ptr(const Extension::Ptr& ext) {
    return std::dynamic_pointer_cast<T>(_get_extension(ext));
}
}  // namespace ov

#define OPENVINO_CREATE_EXTENSIONS(extensions)                             \
    OPENVINO_EXTENSION_API                                                 \
    void ::ov::create_extensions(std::vector<::ov::Extension::Ptr>& ext) { \
        ext = extensions;                                                  \
    }
