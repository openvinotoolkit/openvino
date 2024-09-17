// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/executor.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/memory_arguments.hpp"

namespace ov {
namespace intel_cpu {

template <typename ExecutorT, typename Attrs, typename ShapeAgnosticData>
class DefaultInstantiator {
public:
    std::shared_ptr<ExecutorT> operator()(const MemoryArgs& memory,
                                          const Attrs& attrs,
                                          const ExecutorContext::CPtr context,
                                          const std::shared_ptr<ShapeAgnosticData> shapeAgnosticData) {
        return ExecutorT::create(memory, attrs, context, shapeAgnosticData);
    }
};

template <typename Primitive,
          typename Attrs,
          typename ShapeAgnosticData,
          typename Instantiator = DefaultInstantiator<Primitive, Attrs, ShapeAgnosticData>>
class DnnlFCExecutor : public Executor {
public:
    using PrimitivePtr = std::shared_ptr<Primitive>;
    DnnlFCExecutor(const Attrs& attrs,
                   const PostOps& postOps,
                   const MemoryArgs& memory,
                   const ExecutorContext::CPtr context,
                   const bool cacheWeights)
        : m_attrs(attrs),
          m_context(context),
          m_shapeAgnosticData(Primitive::createShapeAgnosticData(m_attrs, postOps, memory, m_context, cacheWeights)),
          m_primArgs(m_shapeAgnosticData->primAttrs.dnnlArgs) {}
    bool update(const MemoryArgs& memory) override {
        const auto primitive = createPrimitive(memory);
        if (!primitive) {
            return false;
        }
        updateMemory(m_primitive, primitive, memory);
        m_primitive = primitive;
        return true;
    }

    void execute(const MemoryArgs& memory) override {
        if (resetSrcMemoryDataHandle)
            m_primArgs[DNNL_ARG_SRC].set_data_handle(memory.at(ARG_SRC)->getData());
        if (resetDstMemoryDataHandle)
            m_primArgs[DNNL_ARG_DST].set_data_handle(memory.at(ARG_DST)->getData());

        m_primitive->execute(m_primArgs);
    }

    impl_desc_type implType() const override {
        return m_primitive ? m_primitive->implType() : undef;
    }

    void moveMemToNumaNode(int numaNodeID) override {
        if (curNumaNode == numaNodeID) {
            return;
        }
        const auto newPrimMemDesc = m_primitive->scratchPadDesc();
        m_scratchPadMemory = m_context->getScratchPad(numaNodeID)->createScratchPadMem(newPrimMemDesc);
        m_primArgs[DNNL_ARG_SCRATCHPAD] = m_scratchPadMemory->getPrimitive();

        if (m_primArgs.count(DNNL_ARG_WEIGHTS)) {
            if (!mbind_move(m_primArgs[DNNL_ARG_WEIGHTS], numaNodeID)) {
                DEBUG_LOG("[FullyConnected] move DNNL_ARG_WEIGHTS to node ", numaNodeID, " failed");
            }
        }

        if (m_primArgs.count(DNNL_ARG_BIAS)) {
            if (!mbind_move(m_primArgs[DNNL_ARG_BIAS], numaNodeID)) {
                DEBUG_LOG("[FullyConnected] move DNNL_ARG_BIAS to node ", numaNodeID, " failed");
            }
        }
        curNumaNode = numaNodeID;
    }

private:
    void updateSrcMemory(const DnnlMemoryDescPtr& memDesc, const PrimitivePtr primitive, const MemoryPtr memory) {
        const auto& primMemDesc = primitive->srcDesc();
        if (memDesc->isCompatible(*primMemDesc)) {
            m_primArgs[DNNL_ARG_SRC] = memory->getPrimitive();
        } else {
            resetSrcMemoryDataHandle = true;
            // create 2D memory without underlying buffer and reset to the actual memory in scope of 'execute' call
            m_primArgs[DNNL_ARG_SRC] =
                dnnl::memory(primMemDesc->getDnnlDesc(), m_context->getEngine(), DNNL_MEMORY_NONE);
        }
    }

    void updateDstMemory(const DnnlMemoryDescPtr& memDesc, const PrimitivePtr primitive, const MemoryPtr memory) {
        const auto& primMemDesc = primitive->dstDesc();
        if (memDesc->isCompatible(*primMemDesc)) {
            m_primArgs[DNNL_ARG_DST] = memory->getPrimitive();
        } else {
            resetDstMemoryDataHandle = true;
            // create 2D memory without underlying buffer and reset to the actual memory in scope of 'execute' call
            m_primArgs[DNNL_ARG_DST] =
                dnnl::memory(primMemDesc->getDnnlDesc(), m_context->getEngine(), DNNL_MEMORY_NONE);
        }
    }

    void updateWeightsMemory(DnnlMemoryDescPtr originalMemDesc,
                             const PrimitivePtr currentPrimitive,
                             const PrimitivePtr newPrimitive,
                             const MemoryPtr memory) {
        const auto newPrimMemDesc = newPrimitive->weightsDesc();
        if (currentPrimitive && currentPrimitive->weightsDesc()->isCompatible(*newPrimMemDesc))
            return;

        originalMemDesc = Primitive::makeTransposedWeightDescriptor(originalMemDesc, newPrimMemDesc, m_attrs.weightsNonTransposed);

        const auto weiMemory = utils::prepareWeightsMemory(originalMemDesc, newPrimMemDesc, memory, m_context, true);
        m_primArgs[DNNL_ARG_WEIGHTS] = weiMemory->getPrimitive();
    }

    void updateBiasMemory(const MemoryPtr memory) {
        m_primArgs[DNNL_ARG_BIAS] = memory->getPrimitive();
    }

    void updateScratchPadMem(const PrimitivePtr currentPrimitive, const PrimitivePtr newPrimitive) {
        const auto newPrimMemDesc = newPrimitive->scratchPadDesc();
        // @todo should we compare dnnl::memory::desc directly to avoid any overhead?
        if (currentPrimitive && currentPrimitive->scratchPadDesc()->isCompatible(*newPrimMemDesc))
            return;

        m_scratchPadMemory = m_context->getScratchPad(curNumaNode)->createScratchPadMem(newPrimMemDesc);
        m_primArgs[DNNL_ARG_SCRATCHPAD] = m_scratchPadMemory->getPrimitive();
    }

    void updateMemory(const PrimitivePtr currentPrimitive,
                      const PrimitivePtr newPrimitive,
                      const MemoryArgs& memory) {
        const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_SRC)->getDescPtr());
        const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
        const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_DST)->getDescPtr());

        updateSrcMemory(srcDesc, newPrimitive, memory.at(ARG_SRC));
        updateDstMemory(dstDesc, newPrimitive, memory.at(ARG_DST));
        updateWeightsMemory(weiDesc, currentPrimitive, newPrimitive, memory.at(ARG_WEI));
        updateBiasMemory(memory.at(ARG_BIAS));
        updateScratchPadMem(currentPrimitive, newPrimitive);
    }

    PrimitivePtr createPrimitive(const MemoryArgs& memory) {
        return Instantiator{}(memory, m_attrs, m_context, m_shapeAgnosticData);
    }

    const Attrs& m_attrs;
    const ExecutorContext::CPtr m_context;
    const std::shared_ptr<ShapeAgnosticData> m_shapeAgnosticData;
    dnnl_primitive_args& m_primArgs;
    bool resetSrcMemoryDataHandle = false;
    bool resetDstMemoryDataHandle = false;
    MemoryPtr m_scratchPadMemory;
    PrimitivePtr m_primitive;
    int curNumaNode = -1;
};

}  // namespace intel_cpu
}  // namespace ov
