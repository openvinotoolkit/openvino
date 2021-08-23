// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/core/core_visibility.hpp"

// Use extern "C""Catching" Failures in order to avoid issues with mangling
#if defined(_WIN32) && defined(IMPLEMENT_OPENVINO_EXTENSION_API)
#    define OPENVINO_EXTENSION_API extern "C" __declspec(dllexport)
#else
#    define OPENVINO_EXTENSION_API extern "C" OPENVINO_API
#endif

namespace ov {

class SOLoader;
class Extension;

OPENVINO_API
void set_extension_shared_object(const std::shared_ptr<Extension>&, const std::shared_ptr<SOLoader>&);

class OPENVINO_API Extension {
public:
    using Ptr = std::shared_ptr<Extension>;
    virtual ~Extension();
    friend void set_extension_shared_object(const Extension::Ptr&, const std::shared_ptr<SOLoader>&);

private:
    std::shared_ptr<SOLoader> so;
};

OPENVINO_API std::vector<Extension::Ptr> load_extension(const std::string& path);
OPENVINO_API std::vector<Extension::Ptr> load_extension(const std::wstring& path);

OPENVINO_EXTENSION_API
void create_extensions(std::vector<Extension::Ptr>&);
}  // namespace ov

#define OPENVINO_CREATE_EXTENSIONS(extensions)                             \
    OPENVINO_EXTENSION_API                                                 \
    void ::ov::create_extensions(std::vector<::ov::Extension::Ptr>& ext) { \
        ext = extensions;                                                  \
    }
