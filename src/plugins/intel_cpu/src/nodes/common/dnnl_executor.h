// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu_memory.h>
#include <onednn/iml_type_mapper.h>

#include "memory_desc/dnnl_memory_desc.h"

namespace ov {
namespace intel_cpu {

class DnnlExecutor {
protected:
    class IntermReorder {
    public:
        IntermReorder(const dnnl::memory::desc& descSrc, const dnnl::memory::desc& descDst, const dnnl::engine& engine);
        void exec(dnnl::memory& memSrc, dnnl::memory& memDst, const dnnl::stream& strm);
        const dnnl::memory::desc& getSrcDesc() const {
            return m_descSrc;
        }
        const dnnl::memory::desc& getDstDesc() const {
            return m_descDst;
        }

    private:
        dnnl::reorder m_reorder;
        dnnl::memory::desc m_descSrc;
        dnnl::memory::desc m_descDst;
    };

public:
    explicit DnnlExecutor(const dnnl::primitive_desc& pd);
    void exec(const std::unordered_map<int, dnnl::memory>& primArgs, const dnnl::stream& strm);
    bool needReordering() const;
    virtual ~DnnlExecutor() = default;
    dnnl::primitive getExecPrim() const;
    const_dnnl_primitive_desc_t getPrimitiveDesc() const;
    impl_desc_type getImplementationType() const;

    DnnlMemoryDescPtr getSrcDesc() const {
        return src_md;
    }
    DnnlMemoryDescPtr getWeightDesc() const {
        return wghts_md;
    }
    DnnlMemoryDescPtr getDstDesc() const {
        return dst_md;
    }
    DnnlMemoryDescPtr getScratchPadDesc() const {
        return scrch_md;
    }

    const dnnl::memory::desc& getDnnlSrcDesc() const {
        return src_md->getDnnlDesc();
    }
    const dnnl::memory::desc& getDnnlWeightDesc() const {
        return wghts_md->getDnnlDesc();
    }
    const dnnl::memory::desc& getDnnlDstDesc() const {
        return dst_md->getDnnlDesc();
    }
    const dnnl::memory::desc& getDnnlScratchPadDesc() const {
        return scrch_md->getDnnlDesc();
    }

protected:
    virtual void reorder_exec(std::unordered_map<int, dnnl::memory> primArgs, const dnnl::stream& strm);

protected:
    dnnl::primitive execPrim;
    // key is the port number for the primitive that needs memory reordering
    std::unordered_map<int, IntermReorder> inputReorders;
    std::unordered_map<int, IntermReorder> outputReorders;
    DnnlMemoryDescPtr src_md;
    DnnlMemoryDescPtr wghts_md;
    DnnlMemoryDescPtr dst_md;
    DnnlMemoryDescPtr scrch_md;
};

}  // namespace intel_cpu
}  // namespace ov
