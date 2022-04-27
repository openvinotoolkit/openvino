// Copyright (C) 2018-2022 Intel Corporation
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
#    define OPENVINO_EXTENSION_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#    define OPENVINO_EXTENSION_API   OPENVINO_CORE_EXPORTS
#endif

namespace ov {

class Extension;

/**
 * @brief The class provides the base interface for OpenVINO extensions
 * @ingroup ov_model_cpp_api
 */
class OPENVINO_API Extension {
public:
    using Ptr = std::shared_ptr<Extension>;

    virtual ~Extension();
};

/**
 * @brief The entry point for library with OpenVINO extensions
 *
 * @param vector of extensions
 */
OPENVINO_EXTENSION_C_API
void create_extensions(std::vector<Extension::Ptr>&);

}  // namespace ov

/**
 * @brief Macro generates the entry point for the library
 *
 * @param vector of extensions
 */
#define OPENVINO_CREATE_EXTENSIONS(extensions)                             \
    OPENVINO_EXTENSION_C_API                                               \
    void ::ov::create_extensions(std::vector<::ov::Extension::Ptr>& ext) { \
        ext = extensions;                                                  \
    }
