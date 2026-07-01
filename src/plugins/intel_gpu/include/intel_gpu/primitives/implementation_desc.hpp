// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <map>
#include <ostream>
#include <sstream>

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
        case impl_types::sycl: out << "sycl"; break;
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
    } else if (str == "sycl") {
        impl_type = impl_types::sycl;
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

inline std::istream& operator>>(std::istream& is, ImplementationDesc& desc) {
    std::string str;
    is >> str;

    const auto first_sep = str.find(':');
    const auto second_sep = str.find(':', first_sep == std::string::npos ? first_sep : first_sep + 1);
    if (first_sep == std::string::npos || second_sep == std::string::npos) {
        OPENVINO_THROW("Invalid ImplementationDesc value: ", str,
                       ". Expected format: impl_type:kernel_name:output_format");
    }

    const auto impl_type_str = str.substr(0, first_sep);
    const auto kernel_name_str = str.substr(first_sep + 1, second_sep - first_sep - 1);
    const auto output_format_str = str.substr(second_sep + 1);

    {
        std::istringstream impl_type_ss(impl_type_str);
        impl_type_ss >> desc.impl_type;
    }
    desc.kernel_name = kernel_name_str;

    if (output_format_str == "any") {
        desc.output_format = cldnn::format::any;
        return is;
    }

    for (int i = 0; i < static_cast<int>(cldnn::format::type::format_num); i++) {
        const auto format_type = static_cast<cldnn::format::type>(i);
        if (cldnn::format(format_type).to_string() == output_format_str) {
            desc.output_format = format_type;
            return is;
        }
    }

    OPENVINO_THROW("Unsupported output format: ", output_format_str);
}

using ImplForcingMap = std::map<cldnn::primitive_id, ImplementationDesc>;

}  // namespace ov::intel_gpu
