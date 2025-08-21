// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl_common_types.h>
#include <oneapi/dnnl/dnnl_types.h>

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

template <typename Primitive, typename Attrs, typename ShapeAgnosticData, typename Instantiator>
class DnnlExecutor : public Executor {
public:
    using PrimitivePtr = std::shared_ptr<Primitive>;
    DnnlExecutor(Attrs attrs,
                 const MemoryArgs& memory,
                 ExecutorContext::CPtr context,
                 const bool cacheWeights,
                 const bool fc3Das2D = false)
        : m_attrs(std::move(attrs)),
          m_context(std::move(context)),
          m_shapeAgnosticData(Primitive::createShapeAgnosticData(m_attrs, memory, m_context, cacheWeights)),
          m_primArgs(m_shapeAgnosticData->m_primAttrs.dnnlArgs),
          m_fc3Das2D(fc3Das2D) {}
    bool update(const MemoryArgs& memory) override {
        const auto primitive = createPrimitive(memory, m_attrs);
        if (!primitive) {
            return false;
        }
        updateMemory(m_primitive, primitive, memory);
        m_primitive = primitive;
        return true;
    }

    void execute(const MemoryArgs& memory) override {
        if (resetSrcMemoryDataHandle) {
            m_primArgs[DNNL_ARG_SRC].set_data_handle(memory.at(ARG_SRC)->getData());
        }
        if (resetDstMemoryDataHandle) {
            m_primArgs[DNNL_ARG_DST].set_data_handle(memory.at(ARG_DST)->getData());
        }

        m_primitive->execute(m_primArgs);
    }

    void execute() override {
        m_primitive->execute(m_primArgs);
    }

    void execute() const override {
        m_primitive->execute(m_primArgs);
    }

    [[nodiscard]] impl_desc_type implType() const override {
        // to satisfy functional tests logic, implementation type should be shape agnostic
        if (m_shapeAgnosticData->m_implType != impl_desc_type::undef) {
            return m_shapeAgnosticData->m_implType;
        }

        return m_primitive ? m_primitive->implType() : impl_desc_type::undef;
    }

    void moveMemToNumaNode(int numaNodeID) override {
        if (curNumaNode == numaNodeID) {
            return;
        }
        const auto newPrimMemDesc = m_primitive->scratchPadDesc();
        m_scratchPadMemory = m_context->getScratchPad()->createScratchPadMem(newPrimMemDesc);
        m_primArgs[DNNL_ARG_SCRATCHPAD] = m_scratchPadMemory->getPrimitive();

        if (auto it = m_primArgs.find(DNNL_ARG_WEIGHTS); it != m_primArgs.end()) {
            if (!mbind_move(it->second, numaNodeID)) {
                DEBUG_LOG("[FullyConnected] move DNNL_ARG_WEIGHTS to node ", numaNodeID, " failed");
            }
        }

        if (auto it = m_primArgs.find(DNNL_ARG_BIAS); it != m_primArgs.end()) {
            if (!mbind_move(it->second, numaNodeID)) {
                DEBUG_LOG("[FullyConnected] move DNNL_ARG_BIAS to node ", numaNodeID, " failed");
            }
        }
        curNumaNode = numaNodeID;
    }

private:
    void updateSrcMemory(const DnnlMemoryDescPtr& memDesc, const PrimitivePtr primitive, const MemoryPtr& memory) {
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

    void updateDstMemory(const DnnlMemoryDescPtr& memDesc, const PrimitivePtr primitive, const MemoryPtr& memory) {
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
                             const MemoryPtr& memory) {
        if (m_attrs.nonConstantWeights) {  // non constant weights are handled by the primitive
            m_primArgs[DNNL_ARG_WEIGHTS] = memory->getPrimitive();
            return;
        }

        const auto newPrimMemDesc = newPrimitive->weightsDesc();

        if (currentPrimitive && currentPrimitive->weightsDesc()->isCompatible(*newPrimMemDesc)) {
            return;
        }

        originalMemDesc = Primitive::makeTransposedWeightDescriptor(originalMemDesc, newPrimMemDesc, m_attrs);

        const auto weiMemory = utils::prepareWeightsMemory(originalMemDesc, newPrimMemDesc, memory, m_context, true);
        m_primArgs[DNNL_ARG_WEIGHTS] = weiMemory->getPrimitive();
    }

    void updateBiasMemory(const MemoryPtr& memory) {
        m_primArgs[DNNL_ARG_BIAS] = memory->getPrimitive();
    }

    void updatePostOpsMemory(const MemoryArgs& memory) {
        auto update = [&memory, this](int cpuMemoryArg, int dnnlMemoryArg) {
            if (const auto arg = memory.find(cpuMemoryArg); arg != memory.end()) {
                const auto& memory = arg->second;
                m_primArgs[dnnlMemoryArg] = memory->getPrimitive();
            }
        };

        update(ARG_ATTR_POST_OP_DW | ARG_WEI, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
        update(ARG_ATTR_POST_OP_DW | ARG_BIAS, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

        if (m_shapeAgnosticData->m_primAttrs.legacyZeroPoints) {
            update(ARG_ATTR_ZERO_POINTS | ARG_SRC, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
            update(ARG_ATTR_ZERO_POINTS | ARG_WEI, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
            update(ARG_ATTR_ZERO_POINTS | ARG_DST, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);
        } else {
            update(ARG_ATTR_ZERO_POINTS | ARG_SRC_3, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
        }
    }

    void updateScratchPadMem(const PrimitivePtr currentPrimitive, const PrimitivePtr newPrimitive) {
        const auto newPrimMemDesc = newPrimitive->scratchPadDesc();
        // @todo should we compare dnnl::memory::desc directly to avoid any overhead?
        if (currentPrimitive && currentPrimitive->scratchPadDesc()->isCompatible(*newPrimMemDesc)) {
            return;
        }

        m_scratchPadMemory = m_context->getScratchPad()->createScratchPadMem(newPrimMemDesc);
        m_primArgs[DNNL_ARG_SCRATCHPAD] = m_scratchPadMemory->getPrimitive();
    }

    void updateMemory(const PrimitivePtr currentPrimitive, const PrimitivePtr newPrimitive, const MemoryArgs& memory) {
        const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_SRC)->getDescPtr());
        const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
        const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_DST)->getDescPtr());

        if (m_fc3Das2D) {
            updateSrcMemory(srcDesc, newPrimitive, memory.at(ARG_SRC));
            updateDstMemory(dstDesc, newPrimitive, memory.at(ARG_DST));
        } else {
            m_primArgs[DNNL_ARG_SRC] = memory.at(ARG_SRC)->getPrimitive();
            m_primArgs[DNNL_ARG_DST] = memory.at(ARG_DST)->getPrimitive();
        }

        updateWeightsMemory(weiDesc, currentPrimitive, newPrimitive, memory.at(ARG_WEI));
        updateBiasMemory(memory.at(ARG_BIAS));
        updatePostOpsMemory(memory);
        updateScratchPadMem(currentPrimitive, newPrimitive);
    }

    PrimitivePtr createPrimitive(const MemoryArgs& memory, const Attrs& attrs) {
        return Instantiator{}(memory, attrs, m_context, m_shapeAgnosticData);
    }
    // @todo there is no real reason to store attrs. Better to just pass as api argument
    Attrs m_attrs;
    const ExecutorContext::CPtr m_context;
    std::shared_ptr<ShapeAgnosticData> m_shapeAgnosticData;
    dnnl_primitive_args& m_primArgs;
    bool resetSrcMemoryDataHandle = false;
    bool resetDstMemoryDataHandle = false;
    MemoryPtr m_scratchPadMemory;
    PrimitivePtr m_primitive;
    int curNumaNode = -1;
    bool m_fc3Das2D = false;
};

}  // namespace ov::intel_cpu
