// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>
#include <tuple>
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include "intel_gpu/runtime/shape_predictor.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/core/node.hpp"
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

inline bool can_use_usm_host(cldnn::engine& engine, const uint64_t total_output_bytes) {
    GPU_DEBUG_IF(ExecutionConfig::get_usm_policy() == 1) { return true; }
    GPU_DEBUG_IF(ExecutionConfig::get_usm_policy() == 2) { return false; }

    auto can_use_usm = engine.use_unified_shared_memory();
    // When output size is large, it is better not to write to usm_host directly
    const uint64_t LARGE_OUTPUT_BYTES_THRESHOLD = 4 * 1048576;

    const auto& device_info = engine.get_device_info();
    if ((device_info.gfx_ver.major == 12 && device_info.gfx_ver.minor == 60) ||
        (device_info.gfx_ver.major >= 20 && device_info.dev_type == cldnn::device_type::discrete_gpu) ||
        (device_info.dev_type == cldnn::device_type::discrete_gpu && total_output_bytes > LARGE_OUTPUT_BYTES_THRESHOLD)) {
        // WA: Disable USM host memory for infer request`s tensors for PVC and subsequent dGPUs, as kernel access
        // to system memory is slower than using an explicit memcpy (Host <-> Device) call with the copy engine
        // Driver tickets with additional details: 6155, 10054
        GPU_DEBUG_TRACE << "Do not use usm_host for performance issue" << std::endl;
        can_use_usm = false;
    }

    return can_use_usm;
}
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
            return ov::element::f32;
        case ov::element::u64:
            return ov::element::i32;
        default: return et;
    }
}

bool is_supported(ov::element::Type_t et);
bool data_types_are_supported(const ov::Node* node);

using PrecisionMap = std::map<ov::element::Type_t, ov::element::Type>;

std::vector<cldnn::optional_data_type> get_output_data_types(const ov::Node* op, PrecisionMap precision_map = {});

inline std::vector<cldnn::optional_data_type> get_output_data_types(const std::shared_ptr<ov::Node>& op, PrecisionMap precision_map = {}) {
    return get_output_data_types(op.get(), precision_map);
}

inline ov::Shape get_tensor_shape(const ov::PartialShape& pshape) {
    ov::Shape res(pshape.size());
    for (size_t i = 0; i < pshape.size(); i++) {
        res[i] = pshape[i].is_dynamic() ? 0 : pshape[i].get_length();
    }

    return res;
}

inline ov::Shape predict_shape(const std::string& name, const cldnn::layout layout, cldnn::ShapePredictor& shape_predictor) {
    auto prealloc_info = shape_predictor.predict_preallocation_shape(name, layout, false);
    const auto& preallocation_shape = prealloc_info.second;
    auto can_preallocate_buffer = prealloc_info.first &&
                                    shape_predictor.can_preallocate(cldnn::ceil_div(ov::shape_size(preallocation_shape) *
                                    ov::element::Type(layout.data_type).bitwidth(), 8));
    if (can_preallocate_buffer) {
        return preallocation_shape;
    }

    return layout.get_shape();
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

void convert_and_copy(
    const ov::ITensor* src,
    cldnn::memory::ptr dst,
    cldnn::stream& stream,
    const cldnn::layout& src_layout = cldnn::layout({}, ov::element::dynamic, cldnn::format::bfyx, cldnn::padding()));
void convert_and_copy(const cldnn::memory::ptr src, ov::ITensor const* dst, const cldnn::stream& stream);
void convert_and_copy(const ov::ITensor* src, ov::ITensor* dst, const cldnn::stream& stream);
void convert_and_copy(const cldnn::memory::ptr src, cldnn::memory::ptr dst, cldnn::stream& stream);

}  // namespace intel_gpu

inline std::ostream& operator<<(std::ostream& os, const ov::AnyMap& params) {
    for (const auto& p : params) {
        os << p.first << " : " << p.second.as<std::string>() << std::endl;
    }
    return os;
}
}  // namespace ov
