// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/runtime/utils.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "dnnl.hpp"
#endif
namespace cldnn {

class NodeFuseParams {
public:
    explicit NodeFuseParams(primitive_type_id type) : _type(type) {}
    virtual ~NodeFuseParams() = default;
    virtual primitive_type_id type() const { return _type; }
    virtual size_t ops_count() const { return 0; }

private:
    const primitive_type_id _type;
};

// Dependency(Input) type of fusing operation in fused node.
// There are different ways to generate input var name and type by the dependency(input) type in MakeOpJitConstants in jitter
// - ORIGINAL: The input of the operation is the fused node such as Conv
// - EXTERNAL: The input of the operation is the external node outside the fused node
// - INTERNAL: The input of the operation is the another fused operation in the fused node
enum class FusedInputType {
    UNDEFINED  = -1,
    ORIGINAL   = 0,
    EXTERNAL   = 1,
    INTERNAL   = 2
};

struct fused_primitive_desc {
    explicit fused_primitive_desc(const std::shared_ptr<const primitive>& prim) : desc(prim) {}

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

    bool operator==(const fused_primitive_desc& rhs) const {
        if (total_num_deps != rhs.total_num_deps)
            return false;
        if (outer_dep_start_idx != rhs.outer_dep_start_idx)
            return false;

        return *desc == *rhs.desc;
    }

    bool operator!=(const fused_primitive_desc& rhs) const { return !(*this == rhs); }

    bool has_outer_dep() const { return outer_dep_start_idx >= 0; }

    std::shared_ptr<const primitive> desc;
    std::shared_ptr<NodeFuseParams> f_param;

    layout input_layout;
    layout output_layout;

    struct InputDescriptor {
        InputDescriptor(FusedInputType type, size_t idx, ov::element::Type_t element_type) : m_type(type), m_idx(idx), m_element_type(element_type) {};

        FusedInputType m_type;
        size_t m_idx;
        ov::element::Type_t m_element_type;
    };

    std::vector<InputDescriptor> inputs;


    std::vector<std::pair<primitive_id, size_t>> deps;
    std::map<primitive_id, size_t> fused_deps;
    // TODO:
    // Currently, it assumes very simple case where dep 0 is the fused node and no input sharing b/w fused node and peer node
    // To cover such cases where some of the peer node uses input of fused node, we need to maintain actual indexes of the dependencies
    // not only the "starting index".
    int32_t outer_dep_start_idx = -1; // if -1, no external dep after fusing
    size_t total_num_deps = 0;
};

template<typename... SupportedTypes>
bool fused_ops_are_one_of(const std::vector<fused_primitive_desc>& fused_ops) {
    std::array supported_type_ids = {(SupportedTypes::type_id())...};
    for (const auto& fd : fused_ops) {
        if (!one_of(fd.desc->type, supported_type_ids)) {
            return false;
        }
    }
    return true;
}

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
    dnnl::memory::dims dims;
    dnnl::memory::data_type dt;
};
#endif // ENABLE_ONEDNN_FOR_GPU
} // namespace cldnn


namespace ov::intel_gpu {
    using FusedPrimitiveDesc = cldnn::fused_primitive_desc;
}  // namespace ov::intel_gpu
