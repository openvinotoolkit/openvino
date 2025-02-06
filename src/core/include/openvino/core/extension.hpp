// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "openvino/core/core_visibility.hpp"
#include "openvino/core/type.hpp"

#define OPENVINO_EXTENSION_C_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#define OPENVINO_EXTENSION_API   OPENVINO_CORE_EXPORTS

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

#ifndef OV_CREATE_EXTENSION
#    define OV_CREATE_EXTENSION create_extensions
#endif

struct OPENVINO_API FeExtension {
    virtual ~FeExtension();
};

template <class T, class... FEs>
class OPENVINO_API FeExtensionBase : public T, public FeExtension, public FEs... {
public:
    using T::T;
};

// All FEs should be added here as IsFeName structures
struct OPENVINO_API IsIr {
    virtual ~IsIr();
};

struct OPENVINO_API IsOnnx {
    virtual ~IsOnnx();
};

struct OPENVINO_API IsTf {
    virtual ~IsTf();
};

// These classes can be defined in customer code
template <class T>
class OPENVINO_API IrFeExtension : public FeExtensionBase<T, IsIr> {
public:
    using FeExtensionBase<T, IsIr>::FeExtensionBase;
};

template <class T>
class OPENVINO_API OnnxFeExtension : public FeExtensionBase<T, IsOnnx> {
public:
    using FeExtensionBase<T, IsOnnx>::FeExtensionBase;
};

template <class T>
class OPENVINO_API TfFeExtension : public FeExtensionBase<T, IsTf> {
public:
    using FeExtensionBase<T, IsTf>::FeExtensionBase;
};

// As example. It is possible to create extensions for several FEs
template <class T>
class OPENVINO_API IrAndOnnxFeExtension : public FeExtensionBase<T, IsIr, IsOnnx> {
public:
    using FeExtensionBase<T, IsIr, IsOnnx>::FeExtensionBase;
};

}  // namespace ov
/**
 * @brief The entry point for library with OpenVINO extensions
 *
 * @param vector of extensions
 */
OPENVINO_EXTENSION_C_API
void OV_CREATE_EXTENSION(std::vector<ov::Extension::Ptr>&);

/**
 * @brief Macro generates the entry point for the library
 *
 * @param vector of extensions
 */
#define OPENVINO_CREATE_EXTENSIONS(extensions)                                               \
    OPENVINO_EXTENSION_C_API void OV_CREATE_EXTENSION(std::vector<ov::Extension::Ptr>& ext); \
    OPENVINO_EXTENSION_C_API void OV_CREATE_EXTENSION(std::vector<ov::Extension::Ptr>& ext) { ext = extensions; }
