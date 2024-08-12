// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <oneapi/dnnl/dnnl.hpp>

#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/matmul_config.hpp"

namespace ov {
namespace intel_cpu {

class DnnlMatMulPrimitive {
    struct Key {
        DnnlMemoryDescCPtr src;
        DnnlMemoryDescCPtr wei;
        DnnlMemoryDescCPtr bias;
        DnnlMemoryDescCPtr dst;
        dnnl::primitive_attr attr;

        size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

public:
    DnnlMatMulPrimitive(const Key& key, const dnnl::engine& engine, const std::vector<impl_desc_type>& implPriorities);

    void execute(const dnnl_primitive_args& primArgs) const;

    const DnnlMemoryDescPtr srcDesc() const {
        return m_srcDesc;
    }

    const DnnlMemoryDescPtr dstDesc() const {
        return m_dstDesc;
    }

    const DnnlMemoryDescPtr weightsDesc() const {
        return m_weiDesc;
    }

    const DnnlMemoryDescPtr scratchPadDesc() const {
        return m_scratchPadDesc;
    }

    impl_desc_type implType() const {
        return m_implType;
    }

    static DnnlShapeAgnosticDataPtr createShapeAgnosticData(const FCAttrs& attrs,
                                                            const PostOps& postOps,
                                                            const MemoryArgs& memory,
                                                            const ExecutorContext::CPtr context,
                                                            const bool cacheWeights);

    static DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr srcDesc,
                                                            const DnnlMemoryDescPtr dstDesc,
                                                            bool weightsNonTransposed);

    static std::shared_ptr<DnnlMatMulPrimitive> create(const MemoryArgs& memory,
                                                       const MatMulAttrs& attrs,
                                                       const ExecutorContext::CPtr context,
                                                       const DnnlShapeAgnosticDataPtr& shapeAgnosticData);

private:
    dnnl::stream m_stream;
    dnnl::primitive_desc m_primDesc;
    impl_desc_type m_implType;
    DnnlMemoryDescPtr m_srcDesc;
    DnnlMemoryDescPtr m_weiDesc;
    DnnlMemoryDescPtr m_dstDesc;
    DnnlMemoryDescPtr m_scratchPadDesc;
    dnnl::primitive m_prim;
};

using DnnlMatMulPrimitivePtr = std::shared_ptr<DnnlMatMulPrimitive>;

}  // namespace intel_cpu
}  // namespace ov
