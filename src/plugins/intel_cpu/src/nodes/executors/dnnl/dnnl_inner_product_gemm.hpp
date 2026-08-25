// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <unordered_map>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "onednn/iml_type_mapper.h"
#include "thread_pool_imp.hpp"

namespace ov::intel_cpu::dnnl_utils {

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
    InnerProduct& operator=(const InnerProduct&) = delete;
    InnerProduct& operator=(InnerProduct&&) = delete;
    ~InnerProduct() = default;

    InnerProduct(const dnnl::engine& eng, const std::shared_ptr<ThreadPool>& threadPool, const InnerProductKey& key);

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
    void init_w_zp(const VectorDims& zp_shape);

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
Dim normalizeM(Dim M);

// Descriptor of an optional 1D bias of N elements. Returns an empty descriptor if there is no bias.
dnnl::memory::desc makeBiasMd(dnnl::memory::dim N, const MemoryPtr& biasMem);

// Prepend a batch dimension to a blocked descriptor, keeping the inner blocking untouched. Used to
// describe a batch of per-group weights / scales / zero points which oneDNN itself is unaware of.
DnnlMemoryDescPtr addBatchDim(const BlockedMemoryDescPtr& desc, size_t batchDim);

}  // namespace ov::intel_cpu::dnnl_utils
