// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "kernel_selector_params.h"
#include "meta_utils.h"

namespace cldnn {
struct fused_primitive_desc {
    explicit fused_primitive_desc(std::shared_ptr<const primitive> prim) : desc(prim) {}

    template <class PType>
    bool is_type() const {
        static_assert(meta::is_primitive<PType>::value,
            "Type argument fused_primitive_desc::is_type should be a non-const, non-volatile type derived from primitive");
        return desc->type == PType::type_id();
    }

    template <class PType>
    std::shared_ptr<const PType> typed_desc() const { return std::static_pointer_cast<const PType>(desc); }

    template<typename T>
    std::shared_ptr<T> get_typed_fuse_params() const {
        auto p = std::dynamic_pointer_cast<T>(f_param);
        if (!p)
            throw std::runtime_error("Invalid dynamic cast of fused parameters!");
        return p;
    }

    std::shared_ptr<const primitive> desc;

    layout input_layout = layout(data_types::f32, format::bfyx, tensor());
    layout output_layout = layout(data_types::f32, format::bfyx, tensor());

    std::shared_ptr<kernel_selector::fuse_params> f_param;

    std::vector<std::pair<primitive_id, size_t>> deps;
    std::map<primitive_id, size_t> fused_deps;
    size_t dep_start_idx;
    size_t total_num_deps = 0;
};

#ifdef ENABLE_ONEDNN_FOR_GPU
enum class onednn_post_op_type : uint32_t {
    eltwise_act,
    eltwise_clip,
    eltwise_linear,
    eltwise_round,
    eltwise_hardsigmoid,
    binary_mul,
    binary_add,
    binary_sub,
    binary_max,
    binary_min,
    binary_relu,
    scale,
    sum,
    optimized,
    optimized_eltwise_act,
    optimized_eltwise_clip,
    optimized_eltwise_linear,
    optimized_eltwise_round,
    optimized_sum
};

static inline std::ostream& operator<< (std::ostream& os, onednn_post_op_type& t) {
    switch (t) {
        case onednn_post_op_type::eltwise_act: os << "eltwise_act"; break;
        case onednn_post_op_type::eltwise_clip: os << "eltwise_clip"; break;
        case onednn_post_op_type::eltwise_linear: os << "eltwise_linear"; break;
        case onednn_post_op_type::eltwise_round: os << "eltwise_round"; break;
        case onednn_post_op_type::eltwise_hardsigmoid: os << "eltwise_hardsigmoid"; break;
        case onednn_post_op_type::binary_mul: os << "binary_mul"; break;
        case onednn_post_op_type::binary_add: os << "binary_add"; break;
        case onednn_post_op_type::binary_sub: os << "binary_sub"; break;
        case onednn_post_op_type::binary_max: os << "binary_max"; break;
        case onednn_post_op_type::binary_min: os << "binary_min"; break;
        case onednn_post_op_type::binary_relu: os << "binary_relu"; break;
        case onednn_post_op_type::scale: os << "scale"; break;
        case onednn_post_op_type::sum: os << "sum"; break;
        case onednn_post_op_type::optimized: os << "optimized"; break;
        case onednn_post_op_type::optimized_eltwise_act: os << "optimized_eltwise_act"; break;
        case onednn_post_op_type::optimized_eltwise_clip: os << "optimized_eltwise_clip"; break;
        case onednn_post_op_type::optimized_eltwise_linear: os << "optimized_eltwise_linear"; break;
        case onednn_post_op_type::optimized_eltwise_round: os << "optimized_eltwise_round"; break;
        case onednn_post_op_type::optimized_sum: os << "optimized_sum"; break;
        default: os << "invalid";
    }
    return os;
}

struct fused_primitive_desc_onednn {
    onednn_post_op_type op_type; // onednn post-operation type
    size_t mem_offset;           // index of a memory buffer for current post-operation
    size_t mem_dep;              // memory dependency for working with fused node
    dnnl::memory::format_tag tag;
    bool flatten;
};
#endif // ENABLE_ONEDNN_FOR_GPU
} // namespace cldnn
