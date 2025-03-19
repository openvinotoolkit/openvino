// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov::intel_cpu {

// @todo executor is not complete and covers only 1x1 fallback case for fullyconnected node
class DnnlConvolutionPrimitive {
    // @todo generalize caching for dnnl backend
    struct Key {
        // @todo shouldn't we have a key representing onednn specific data types only?
        const DnnlMemoryDescCPtr src;
        const DnnlMemoryDescCPtr wei;
        const DnnlMemoryDescCPtr bias;
        const DnnlMemoryDescCPtr dst;

        const dnnl::primitive_attr attr;

        [[nodiscard]] size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

public:
    DnnlConvolutionPrimitive(const Key& key,
                             const dnnl::engine& engine,
                             const std::vector<impl_desc_type>& implPriorities);

    void execute(const dnnl_primitive_args& primArgs) const;

    [[nodiscard]] const DnnlMemoryDescPtr srcDesc() const {
        return m_srcDesc;
    }

    [[nodiscard]] const DnnlMemoryDescPtr dstDesc() const {
        return m_dstDesc;
    }

    [[nodiscard]] const DnnlMemoryDescPtr weightsDesc() const {
        return m_weiDesc;
    }

    [[nodiscard]] const DnnlMemoryDescPtr scratchPadDesc() const {
        return m_scratchPadDesc;
    }

    [[nodiscard]] impl_desc_type implType() const {
        return m_implType;
    }

    static DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                            const DnnlMemoryDescPtr& dstDesc,
                                                            bool weightsNonTransposed);

    // create shape agnostic data using FC attributes (1x1 Convolution as FC executor)
    static DnnlShapeAgnosticDataPtr createShapeAgnosticData(const FCAttrs& attrs,
                                                            const PostOps& postOps,
                                                            const MemoryArgs& memory,
                                                            const ExecutorContext::CPtr& context,
                                                            const bool cacheWeights);

    static std::shared_ptr<DnnlConvolutionPrimitive> create(const MemoryArgs& memory,
                                                            const ConvAttrs& attrs,
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

using DnnlConvExecutorPtr = std::shared_ptr<DnnlConvolutionPrimitive>;

}  // namespace ov::intel_cpu
