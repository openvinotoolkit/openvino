// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <ostream>

#include "openvino/core/except.hpp"
#include "intel_gpu/primitives/primitive.hpp"

namespace cldnn {

/// @brief Primitives implementation type.
enum class impl_types : uint8_t {
    cpu = 1 << 0,
    common = 1 << 1,
    ocl = 1 << 2,
    onednn = 1 << 3,
    sycl = 1 << 4,
    cm = 1 << 5,
    any = 0xFF,
};

inline impl_types operator&(impl_types a, impl_types b) {
    typedef std::underlying_type<impl_types>::type underlying_type;
    return static_cast<impl_types>(static_cast<underlying_type>(a) & static_cast<underlying_type>(b));
}

inline impl_types operator|(impl_types a, impl_types b) {
    typedef std::underlying_type<impl_types>::type underlying_type;
    return static_cast<impl_types>(static_cast<underlying_type>(a) | static_cast<underlying_type>(b));
}

inline impl_types operator~(impl_types a) {
    typedef std::underlying_type<impl_types>::type underlying_type;
    return static_cast<impl_types>(~static_cast<underlying_type>(a));
}

inline std::ostream& operator<<(std::ostream& out, const impl_types& impl_type) {
    switch (impl_type) {
        case impl_types::cpu: out << "cpu"; break;
        case impl_types::common: out << "common"; break;
        case impl_types::ocl: out << "ocl"; break;
        case impl_types::onednn: out << "onednn"; break;
        case impl_types::cm: out << "cm"; break;
        case impl_types::any: out << "any"; break;
        default: out << "unknown"; break;
    }

    return out;
}

inline std::istream& operator>>(std::istream& is, impl_types& impl_type) {
    std::string str;
    is >> str;
    if (str == "cpu") {
        impl_type = impl_types::cpu;
    } else if (str == "common") {
        impl_type = impl_types::common;
    } else if (str == "ocl") {
        impl_type = impl_types::ocl;
    } else if (str == "onednn") {
        impl_type = impl_types::onednn;
    } else if (str == "cm") {
        impl_type = impl_types::cm;
    } else if (str == "any") {
        impl_type = impl_types::any;
    } else {
        OPENVINO_THROW("Unsupported impl type: ", str);
    }
    return is;
}

/// @brief Possible supported shape types.
enum class shape_types : uint8_t {
    static_shape = 1 << 0,
    dynamic_shape = 1 << 1,
    any = 0xFF,
};

inline shape_types operator&(shape_types a, shape_types b) {
    typedef std::underlying_type<shape_types>::type underlying_type;
    return static_cast<shape_types>(static_cast<underlying_type>(a) & static_cast<underlying_type>(b));
}

inline shape_types operator|(shape_types a, shape_types b) {
    typedef std::underlying_type<shape_types>::type underlying_type;
    return static_cast<shape_types>(static_cast<underlying_type>(a) | static_cast<underlying_type>(b));
}

inline shape_types operator~(shape_types a) {
    typedef std::underlying_type<shape_types>::type underlying_type;
    return static_cast<shape_types>(~static_cast<underlying_type>(a));
}

inline std::ostream& operator<<(std::ostream& out, const shape_types& shape_type) {
    switch (shape_type) {
        case shape_types::static_shape: out << "static_shape"; break;
        case shape_types::dynamic_shape: out << "dynamic_shape"; break;
        case shape_types::any: out << "any"; break;
        default: out << "unknown"; break;
    }

    return out;
}

}  // namespace cldnn

namespace ov::intel_gpu {

struct ImplementationDesc {
    cldnn::format::type output_format;  ///< Output format.
    std::string kernel_name;            ///< GPU kernel name.
    cldnn::impl_types impl_type;        ///< GPU implementation type.

    ImplementationDesc() :
        output_format(cldnn::format::any),
        kernel_name(""),
        impl_type(cldnn::impl_types::any) {}

    ImplementationDesc(cldnn::format::type output_format,
                        std::string kernel_name,
                        cldnn::impl_types impl_type = cldnn::impl_types::any) :
        output_format(output_format),
        kernel_name(kernel_name),
        impl_type(impl_type) {}
};

inline std::ostream& operator<<(std::ostream& out, const ImplementationDesc& desc) {
    out << desc.impl_type << ":" << desc.kernel_name << ":" << desc.output_format;
    return out;
}

using ImplForcingMap = std::map<cldnn::primitive_id, ImplementationDesc>;

}  // namespace ov::intel_gpu
