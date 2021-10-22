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
#    define OPENVINO_EXTENSION_C_API extern "C" __declspec(dllexport)
#    define OPENVINO_EXTENSION_API   __declspec(dllexport)
#else
#    define OPENVINO_EXTENSION_C_API extern "C" OPENVINO_API
#    define OPENVINO_EXTENSION_API   OPENVINO_API
#endif

namespace ov {

class OPENVINO_API BaseExtension : public std::enable_shared_from_this<BaseExtension> {
public:
    using Ptr = std::shared_ptr<BaseExtension>;

protected:
    virtual ~BaseExtension();
};

class OPENVINO_API Extension final {
private:
    std::shared_ptr<void> so;
    BaseExtension::Ptr ext;

public:
    Extension(BaseExtension::Ptr ext, std::shared_ptr<void> so = {});

    const BaseExtension::Ptr& get() const;
};

OPENVINO_API std::vector<Extension> load_extension(const std::string& path);
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
OPENVINO_API std::vector<Extension> load_extension(const std::wstring& path);
#endif

OPENVINO_EXTENSION_C_API
void create_extensions(std::vector<BaseExtension::Ptr>&);

}  // namespace ov

#define OPENVINO_CREATE_EXTENSIONS(extensions)                                 \
    OPENVINO_EXTENSION_C_API                                                   \
    void ::ov::create_extensions(std::vector<::ov::BaseExtension::Ptr>& ext) { \
        ext = extensions;                                                      \
    }
