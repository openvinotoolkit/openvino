// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov::intel_cpu {

class DnnlFCPrimitive {
    struct Key {
        DnnlMemoryDescCPtr src;
        DnnlMemoryDescCPtr wei;
        DnnlMemoryDescCPtr bias;
        DnnlMemoryDescCPtr dst;
        dnnl::primitive_attr attr;
        bool sparseWeights;
        Config::ModelType modelType;

        [[nodiscard]] size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

public:
    DnnlFCPrimitive(const Key& key, const dnnl::engine& engine, const std::vector<impl_desc_type>& implPriorities);

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

    static DnnlShapeAgnosticDataPtr createShapeAgnosticData(const FCAttrs& attrs,
                                                            const PostOps& postOps,
                                                            const MemoryArgs& memory,
                                                            const ExecutorContext::CPtr& context,
                                                            const bool cacheWeights);

    static bool useWeightsDecompressionImpl(const ov::element::Type inputType,
                                            const ov::element::Type weightsType,
                                            const Config::ModelType modelType);

    static DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                            const DnnlMemoryDescPtr& dstDesc,
                                                            bool weightsNonTransposed);

    static std::shared_ptr<DnnlFCPrimitive> create(const MemoryArgs& memory,
                                                   const FCAttrs& attrs,
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

using DnnlFCPrimitivePtr = std::shared_ptr<DnnlFCPrimitive>;

}  // namespace ov::intel_cpu
