// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <tuple>
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_gpu {

enum class TensorType {
    BT_EMPTY,
    BT_BUF_INTERNAL,
    BT_BUF_SHARED,
    BT_USM_SHARED,
    BT_USM_HOST_INTERNAL,
    BT_USM_DEVICE_INTERNAL,
    BT_IMG_SHARED,
    BT_SURF_SHARED,
    BT_DX_BUF_SHARED,
};

#define TensorValue(val) static_cast<cldnn::tensor::value_type>(val)

inline cldnn::tensor tensor_from_dims(const ov::Shape& dims, int def = 1) {
    switch (dims.size()) {
    case 0: return cldnn::tensor(cldnn::batch(def), cldnn::feature(def), cldnn::spatial(def, def));
    case 1: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(def), cldnn::spatial(def, def));
    case 2: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, def));
    case 3: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, dims[2]));
    case 4: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[3], dims[2]));
    case 5: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[4], dims[3], dims[2]));
    case 6: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[5], dims[4], dims[3], dims[2]));
    default: OPENVINO_THROW("Invalid dimensions size(", dims.size(), ") for gpu tensor");
    }
}

template<typename T, typename V>
std::tuple<V, V, V> get_xyz(const T data, V def) {
    switch (data.size()) {
        case 1:  return std::make_tuple(def,                     static_cast<V>(data[0]), def);
        case 2:  return std::make_tuple(static_cast<V>(data[1]), static_cast<V>(data[0]), def);
        case 3:  return std::make_tuple(static_cast<V>(data[2]), static_cast<V>(data[1]), static_cast<V>(data[0]));
        default: return std::make_tuple(def,                     def,                     def);
    }
}

inline cldnn::layout make_layout(const ov::element::Type type, const ov::Shape& shape) {
    return cldnn::layout{ov::PartialShape{shape},
                         cldnn::element_type_to_data_type(type),
                         cldnn::format::get_default_format(shape.size())};
}

inline ov::element::Type convert_to_supported_device_type(ov::element::Type et) {
    switch (et) {
        case ov::element::f64:
        case ov::element::i16:
        case ov::element::u16:
            return ov::element::f32;
        case ov::element::u64:
        case ov::element::u32:
            return ov::element::i32;
        default: return et;
    }
}

/// WA: Force exit. Any opencl api call can be hang after CL_OUT_OF_RESOURCES.
inline void ForceExit() {
    std::cerr << "[GPU] force exit.\n"
              << "\tDue to the driver bug any subsequent OpenCL API call will cause application hang, "
              << "so GPU plugin can't finish correctly.\n"
              << "\tPlease try to update the driver or reduce memory consumption "
              << "(use smaller batch size, less streams, lower precision, etc)"
              << "to avoid CL_OUT_OF_RESOURCES exception" << std::endl;
    std::_Exit(-1);
}

}  // namespace intel_gpu

inline std::ostream& operator<<(std::ostream& os, const ov::AnyMap& params) {
    for (const auto& p : params) {
        os << p.first << " : " << p.second.as<std::string>() << std::endl;
    }
    return os;
}
}  // namespace ov
