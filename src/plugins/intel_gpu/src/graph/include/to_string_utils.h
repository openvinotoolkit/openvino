// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/tensor.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/device.hpp"
#include "intel_gpu/primitives/primitive.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include "program_node.h"
#endif


#include <string>
#include <sstream>
#include <memory>

namespace cldnn {

inline std::string to_string_hex(int val) {
    std::stringstream stream;
    stream << "0x" << std::uppercase << std::hex << val;
    return stream.str();
}

inline std::string bool_to_str(bool cond) { return cond ? "true" : "false"; }

inline std::string get_extr_type(const std::string& str) {
    auto begin = str.find('<');
    auto end = str.find('>');

    if (begin == std::string::npos || end == std::string::npos)
        return {};

    return str.substr(begin + 1, (end - begin) - 1);
}

inline std::string dt_to_str(data_types dt) {
    switch (dt) {
        case data_types::bin:
            return "bin";
        case data_types::i8:
            return "i8";
        case data_types::u8:
            return "u8";
        case data_types::i32:
            return "i32";
        case data_types::i64:
            return "i64";
        case data_types::f16:
            return "f16";
        case data_types::f32:
            return "f32";
        default:
            return "unknown (" + std::to_string(typename std::underlying_type<data_types>::type(dt)) + ")";
    }
}

inline std::string fmt_to_str(format fmt) {
    return fmt.to_string();
}

inline std::string type_to_str(std::shared_ptr<const primitive> primitive) { return primitive->type_string(); }

inline std::string allocation_type_to_str(allocation_type type) {
    switch (type) {
    case allocation_type::cl_mem: return "cl_mem";
    case allocation_type::usm_host: return "usm_host";
    case allocation_type::usm_shared: return "usm_shared";
    case allocation_type::usm_device: return "usm_device";
    default: return "unknown";
    }
}

#ifdef ENABLE_ONEDNN_FOR_GPU
inline std::string onednn_post_op_type_to_str(onednn_post_op_type type) {
    switch (type) {
    case onednn_post_op_type::eltwise_act: return "eltwise_act";
    case onednn_post_op_type::eltwise_clip: return "eltwise_clip";
    case onednn_post_op_type::eltwise_linear: return "eltwise_linear";
    case onednn_post_op_type::eltwise_round: return "eltwise_round";
    case onednn_post_op_type::binary_mul: return "binary_mul";
    case onednn_post_op_type::binary_add: return "binary_add";
    case onednn_post_op_type::binary_max: return "binary_max";
    case onednn_post_op_type::binary_min: return "binary_min";
    case onednn_post_op_type::binary_relu: return "binary_relu";
    case onednn_post_op_type::scale: return "scale";
    case onednn_post_op_type::sum: return "sum";
    case onednn_post_op_type::optimized: return "optimized";
    case onednn_post_op_type::optimized_eltwise_act: return "optimized_eltwise_act";
    case onednn_post_op_type::optimized_eltwise_linear: return "optimized_eltwise_linear";
    case onednn_post_op_type::optimized_eltwise_clip: return "optimized_eltwise_clip";
    case onednn_post_op_type::optimized_eltwise_round: return "optimized_eltwise_round";
    case onednn_post_op_type::optimized_sum: return "optimized_sum";
    default: return "unknown";
    }
}
#endif

}  // namespace cldnn
