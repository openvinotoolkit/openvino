// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu_memory.h>
#include <onednn/iml_type_mapper.h>

namespace ov {
namespace intel_cpu {

class DnnlExecutor {
    protected:
        class IntermReorder {
            public:
                IntermReorder(const dnnl::memory::desc& descSrc, const dnnl::memory::desc& descDst, const dnnl::engine& engine);
                void exec(dnnl::memory& memSrc, dnnl::memory& memDst, dnnl::stream strm);
                const dnnl::memory::desc& getSrcDesc() const { return m_descSrc; }
                const dnnl::memory::desc& getDstDesc() const { return m_descDst; }

            private:
                dnnl::reorder m_reorder;
                dnnl::memory::desc m_descSrc;
                dnnl::memory::desc m_descDst;
        };

    public:
        void exec(std::unordered_map<int, dnnl::memory> primArgs, dnnl::stream strm);
        bool needReordering() const;
        virtual ~DnnlExecutor() = default;
        dnnl::primitive getExecPrim() const;
        const_dnnl_primitive_desc_t getPrimitiveDesc() const;
        dnnl::memory::desc getSrcDesc() const;
        dnnl::memory::desc getWeightDesc() const;
        dnnl::memory::desc getDstDesc() const;
        impl_desc_type getImplementationType() const;

    protected:
        DnnlExecutor() = default;
        dnnl::primitive execPrim;
        // key is the port number for the primitive that needs memory reordering
        std::unordered_map<int, IntermReorder> inputReorders;
        std::unordered_map<int, IntermReorder> outputReorders;
};

}   // namespace intel_cpu
}   // namespace ov
