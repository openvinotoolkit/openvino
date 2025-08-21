// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <vector>

#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

class DnnlMatMulPrimitive {
    struct Key {
        DnnlMemoryDescCPtr src0;
        DnnlMemoryDescCPtr src1;
        DnnlMemoryDescCPtr bias;
        DnnlMemoryDescCPtr dst;
        dnnl::primitive_attr attr;
        impl_desc_type implType;
        bool transposeA = false;
        bool transposeB = false;
        bool fcSemantic = false;

        [[nodiscard]] size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

public:
    DnnlMatMulPrimitive(const Key& key,
                        const dnnl::engine& engine,
                        const std::vector<impl_desc_type>& implPriorities,
                        impl_desc_type defaultImplType);

    void execute(const dnnl_primitive_args& primArgs) const;

    [[nodiscard]] DnnlMemoryDescPtr srcDesc() const {
        return m_srcDesc;
    }

    [[nodiscard]] DnnlMemoryDescPtr dstDesc() const {
        return m_dstDesc;
    }

    [[nodiscard]] DnnlMemoryDescPtr weightsDesc() const {
        return m_weiDesc;
    }

    [[nodiscard]] DnnlMemoryDescPtr scratchPadDesc() const {
        return m_scratchPadDesc;
    }

    [[nodiscard]] impl_desc_type implType() const {
        return m_implType;
    }

    static bool useWeightsDecompressionImpl(ov::element::Type inputType, ov::element::Type weightsType);

    static DnnlShapeAgnosticDataPtr createShapeAgnosticData(const MatMulAttrs& attrs,
                                                            const MemoryArgs& memory,
                                                            const ExecutorContext::CPtr& context,
                                                            bool cacheWeights);

    static DnnlShapeAgnosticDataPtr createShapeAgnosticData(const FCAttrs& fcAttrs,
                                                            const MemoryArgs& memory,
                                                            const ExecutorContext::CPtr& context,
                                                            bool cacheWeights);

    static DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                            const DnnlMemoryDescPtr& dstDesc,
                                                            const MatMulAttrs& attrs);

    static std::shared_ptr<DnnlMatMulPrimitive> create(const MemoryArgs& memory,
                                                       const MatMulAttrs& attrs,
                                                       ExecutorContext::CPtr context,
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

}  // namespace ov::intel_cpu
