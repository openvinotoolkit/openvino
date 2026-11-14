// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <unordered_map>
#include <utility>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "thread_pool_imp.hpp"

namespace ov::intel_cpu::utils {

// Key identifying a single oneDNN inner_product primitive: the (M, K) source, the (N, K) weights,
// the destination data type, the optional bias and the decompression scale / zero point shapes.
struct InnerProductKey {
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc bias_md;
    // The destination may differ from the source, e.g. a bf16 enforced node feeding an f32 consumer
    dnnl::memory::data_type dst_data_type = dnnl::memory::data_type::undef;
    VectorDims scale_shape;
    VectorDims zp_shape;
    dnnl::memory::data_type zp_dt = dnnl::memory::data_type::f32;
    size_t src_dyn_quant_group_size = 0;

    [[nodiscard]] size_t hash() const;
    bool operator==(const InnerProductKey& rhs) const;
};

// A single oneDNN inner_product with detached memory handles, so it can be re-executed against
// arbitrary src / weights / dst pointers. Used by the executors that loop a matmul over a batch of
// independent weight matrices (GatherMatmul, GroupedMatMul).
class InnerProduct {
public:
    InnerProduct() = delete;

    InnerProduct(const InnerProduct&) = delete;

    InnerProduct(InnerProduct&&) = delete;

    InnerProduct(const dnnl::engine& eng, const std::shared_ptr<ThreadPool>& thread_pool, const InnerProductKey& key);

    ~InnerProduct() = default;

    InnerProduct& operator=(const InnerProduct&) = delete;

    InnerProduct& operator=(InnerProduct&&) = delete;

    void exec(void* src, void* dst, void* weight, void* bias = nullptr, void* scale = nullptr, void* zp = nullptr);

    [[nodiscard]] dnnl::memory::desc get_weights_md() const {
        return m_wei_md;
    }
    [[nodiscard]] dnnl::memory::desc get_scale_md() const {
        return m_scale_md;
    }
    [[nodiscard]] dnnl::memory::desc get_zp_md() const {
        return m_zp_md;
    }
    [[nodiscard]] impl_desc_type get_impl_type() const {
        return m_impl_type;
    }

private:
    void init_w_scales(const VectorDims& scale_shape);
    void init_w_zp(const VectorDims& zp_shape, dnnl::memory::data_type data_type);

    static std::unordered_map<int, dnnl::memory> make_args(dnnl::memory& src,
                                                           dnnl::memory& dst,
                                                           dnnl::memory& weight,
                                                           dnnl::memory& bias,
                                                           dnnl::memory& scale,
                                                           dnnl::memory& zp);

    dnnl::stream m_stream;
    dnnl::primitive m_prim;
    dnnl::memory::desc m_input_md;
    dnnl::memory::desc m_output_md;
    dnnl::memory::desc m_wei_md;
    dnnl::memory::desc m_scale_md;
    dnnl::memory::desc m_zp_md;
    dnnl::primitive_attr m_attr;
    std::unordered_map<int, dnnl::memory> m_args;
    impl_desc_type m_impl_type = impl_desc_type::unknown;
};

using InnerProductPtr = std::shared_ptr<InnerProduct>;

// Round M up to a bucket boundary to limit the amount of distinct inner_product primitives created
// for a dynamic M.
Dim normalize_m(Dim M);

// Descriptor of an optional 1D bias of N elements. Returns an empty descriptor if there is no bias.
dnnl::memory::desc make_bias_md(dnnl::memory::dim N, const MemoryPtr& bias_mem);

// Prepend a batch dimension to a blocked descriptor, keeping the inner blocking untouched. Used to
// describe a batch of per-group weights / scales / zero points which oneDNN itself is unaware of.
DnnlMemoryDescPtr add_batch_dim(const BlockedMemoryDescPtr& desc, size_t batch_dim);

// Whether the oneDNN inner_product based executors can handle the given source / weights descriptors:
// f32 / bf16 source, and AVX2 for compressed (u3 / u4 / i4 / u8 / i8) weights. Empty (dynamic)
// descriptors are accepted, they are resolved at createPrimitive time.
bool supports_inner_product(const MemoryDescArgs& descs);

// oneDNN inner_product limits for weights decompression, shared by the nodes which use it (GatherMatmul,
// GroupedMatMul): AVX2, AMX shape limits (bf16 inference), IC % G == 0, IC / G >= 4, OC != 1.
// Keep all oneDNN related restrictions here. Always false on non-x64.
bool is_supported_compressed_inner_product(size_t IC,
                                           size_t OC,
                                           size_t G,
                                           const ov::element::Type& inference_precision);

// Per-group scale / zero point shape: the leading group dimension is stripped, since oneDNN describes a
// single group's weights and the group axis is walked by the executor itself. A broadcast scalar ([1])
// is kept as is. Returns an empty shape if there is no such tensor.
VectorDims per_group_shape(const MemoryPtr& mem);

// Prepares a scale / zero point tensor for the batched primitive: takes the per-group descriptor chosen
// by oneDNN (prim_md), prepends the group dimension, and returns the tensor as is if it is already
// compatible or reordered into that layout otherwise. A broadcast scalar ([1]) is first expanded to
// [1, 1, ...] since it has no group dimension.
MemoryPtr prepare_decompression_memory(const MemoryPtr& mem,
                                       const dnnl::memory::desc& prim_md,
                                       const dnnl::engine& eng);

// Resolves dynamic quantization of the source for compressed u3 (and optionally u4) weights: returns the
// source group size (0 if disabled) and the zero point data type to use. u3 weights come with i8 zero
// points; they are converted to u8 which is lossless only when none of them is negative
// (zp_non_negative_i8). u4 zero points are mapped to u8.
//   dq_group_size    - requested dynamic quantization group size (must be a non-zero multiple of 32)
//   wei_precision    - weights precision
//   src_precision    - source (activation) precision
//   zp_precision     - zero point precision
//   zp_non_negative_i8 - true if all i8 zero point values are non-negative
//   K                - input channels count
//   scale_shape / zp_shape - shapes of the scales / zero points tensors (used to check group alignment)
//   allow_u4         - if true, u4 weights are eligible for dynamic quantization in addition to u3;
//                      if false (default) only u3 weights are
std::pair<size_t, dnnl::memory::data_type> dyn_quant_params(uint64_t dq_group_size,
                                                            ov::element::Type wei_precision,
                                                            ov::element::Type src_precision,
                                                            ov::element::Type zp_precision,
                                                            bool zp_non_negative_i8,
                                                            size_t K,
                                                            const VectorDims& scale_shape,
                                                            const VectorDims& zp_shape,
                                                            bool allow_u4 = false);

}  // namespace ov::intel_cpu::utils
