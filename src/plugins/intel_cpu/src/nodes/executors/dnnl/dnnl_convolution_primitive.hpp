// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

// @todo executor is not complete and covers only 1x1 fallback case for fullyconnected node
class DnnlConvolutionPrimitive {
    // @todo generalize caching for dnnl backend
    struct Key {
        // @todo shouldn't we have a key representing onednn specific data types only?
        DnnlMemoryDescCPtr src;
        DnnlMemoryDescCPtr wei;
        DnnlMemoryDescCPtr bias;
        DnnlMemoryDescCPtr dst;

        std::vector<size_t> stride;
        std::vector<size_t> dilation;
        std::vector<ptrdiff_t> paddingL;
        std::vector<ptrdiff_t> paddingR;

        dnnl::primitive_attr attr;

        bool fcSemantic = false;
        bool constantWeights = true;

        [[nodiscard]] size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

    struct IntermediateReorder {
        dnnl::reorder m_reorder;
        dnnl::memory::desc m_memory_desc;
    };

    class IntermediateReorders {
    public:
        IntermediateReorders(const Key& key, const dnnl::primitive_desc& primDesc, const dnnl::engine& engine);

        bool empty() const;

        std::unordered_map<int, IntermediateReorder> m_inputReorders;
        std::unordered_map<int, IntermediateReorder> m_outputReorders;
    };

public:
    DnnlConvolutionPrimitive(const Key& key,
                             const dnnl::engine& engine,
                             const std::vector<impl_desc_type>& implPriorities,
                             impl_desc_type defaultImplType);

    void execute(dnnl_primitive_args& primArgs);

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

    static DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                            const DnnlMemoryDescPtr& dstDesc,
                                                            const ConvAttrs& attrs);

    static DnnlMemoryDescPtr makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                            const DnnlMemoryDescPtr& dstDesc,
                                                            const FCAttrs& attrs);

    static DnnlShapeAgnosticDataPtr createShapeAgnosticData(const ConvAttrs& attrs,
                                                            const MemoryArgs& memory,
                                                            const ExecutorContext::CPtr& context,
                                                            bool cacheWeights);

    // create shape agnostic data using FC attributes (1x1 Convolution as FC executor)
    static DnnlShapeAgnosticDataPtr createShapeAgnosticData(const FCAttrs& fcAttrs,
                                                            const MemoryArgs& memory,
                                                            const ExecutorContext::CPtr& context,
                                                            bool cacheWeights);

    static std::shared_ptr<DnnlConvolutionPrimitive> create(const MemoryArgs& memory,
                                                            const ConvAttrs& attrs,
                                                            const ExecutorContext::CPtr& context,
                                                            const DnnlShapeAgnosticDataPtr& shapeAgnosticData);

    static bool isJitPlanarAvailable(const ConvConfig& config);

    static bool isBrgConvAvailable(const ConvConfig& config);

    static bool isNspcAvailable(const ConvConfig& config);

    static std::tuple<size_t, size_t, size_t, size_t> getChannelParams(const ConvConfig& config);

private:
    dnnl::stream m_stream;
    dnnl::primitive_desc m_primDesc;
    impl_desc_type m_implType;
    DnnlMemoryDescPtr m_srcDesc;
    DnnlMemoryDescPtr m_weiDesc;
    DnnlMemoryDescPtr m_dstDesc;
    DnnlMemoryDescPtr m_scratchPadDesc;
    dnnl::primitive m_prim;
    IntermediateReorders m_intermediateReorders;
};

using DnnlConvExecutorPtr = std::shared_ptr<DnnlConvolutionPrimitive>;

}  // namespace ov::intel_cpu
