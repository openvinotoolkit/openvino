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
#    define OPENVINO_EXTENSION_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    define OPENVINO_EXTENSION_API   OPENVINO_CORE_EXPORTS
#else
#    define OPENVINO_EXTENSION_C_API OPENVINO_EXTERN_C OPENVINO_API
#    define OPENVINO_EXTENSION_API   OPENVINO_API
#endif

namespace ov {

class Extension;

namespace detail {

std::vector<std::shared_ptr<Extension>> load_extensions(const std::string& path);
void unload_extensions(std::vector<std::shared_ptr<Extension>>& path);

}  // namespace detail

class OPENVINO_API Extension : public std::enable_shared_from_this<Extension> {
public:
    using Ptr = std::shared_ptr<Extension>;

    virtual ~Extension();

private:
    friend std::vector<Extension::Ptr> ov::detail::load_extensions(const std::string& path);
    friend void ov::detail::unload_extensions(std::vector<std::shared_ptr<Extension>>& path);
    std::shared_ptr<void> so;
};

OPENVINO_EXTENSION_C_API
void create_extensions(std::vector<Extension::Ptr>&);

}  // namespace ov

#define OPENVINO_CREATE_EXTENSIONS(extensions)                             \
    OPENVINO_EXTENSION_C_API                                               \
    void ::ov::create_extensions(std::vector<::ov::Extension::Ptr>& ext) { \
        ext = extensions;                                                  \
    }
