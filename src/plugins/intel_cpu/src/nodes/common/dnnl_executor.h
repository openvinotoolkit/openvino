// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu_memory.h>
#include <primitive.h>
#include <onednn/iml_type_mapper.h>

#include "graph_context.h"
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
        Primitive getExecPrim() const;
        const_dnnl_primitive_desc_t getPrimitiveDesc() const;
        dnnl::memory::desc getSrcDesc() const;
        dnnl::memory::desc getWeightDesc() const;
        dnnl::memory::desc getDstDesc() const;
        impl_desc_type getImplementationType() const;

    protected:
        DnnlExecutor() = default;
        Primitive execPrim;
        // key is the port number for the primitive that needs memory reordering
        std::unordered_map<int, IntermReorder> inputReorders;
        std::unordered_map<int, IntermReorder> outputReorders;
};

class DnnlExecutor2 {
    protected:
        using PrimArgs = std::unordered_map<int, dnnl::memory>;
        struct IntermReorder {
            dnnl::reorder prim;
            PrimArgs args;
            IntermReorder() = default;
            IntermReorder(IntermReorder&& s) : prim(std::move(s.prim)), args(std::move(s.args)) {}
            IntermReorder(dnnl::reorder prim, PrimArgs args)
                : prim(std::move(prim)),
                args(std::move(args)) {}
        };

    public:
        DnnlExecutor2(const GraphContext::CPtr context, const std::string& name)
            : context(context),
              name(name) {}

        void reset(dnnl::primitive p) {
            prim = p;
            pd = prim.get_primitive_desc();
            args.clear();
            inputReorders.clear();
            outputReorders.clear();
            constFoldings.clear();
            constFolded = false;
            // privateWeightCache was kept to always reference weights of different format
        }

        operator bool() {
            return static_cast<bool>(prim);
        }
        void exec(dnnl::stream strm);
        bool needReordering() const;
        virtual ~DnnlExecutor2() = default;
        const_dnnl_primitive_desc_t getPrimitiveDesc() const;
        dnnl::memory::desc queryMD(const dnnl::query& what, int idx);
        impl_desc_type getImplementationType() const;

        void setSrc(MemoryPtr memPtr, bool isConst, int idx = 0);
        void setWeight(MemoryPtr memPtr, bool isConst, int idx = 0);
        void setOutput(MemoryPtr memPtr, int idx = 0);
        void setArg(int arg_id, dnnl::memory arg) {
            args[arg_id] = arg;
        }
        void setScratchPad();

    protected:
        const GraphContext::CPtr context;
        std::string name;
        dnnl::primitive prim;
        const_dnnl_primitive_desc_t pd;
        PrimArgs args;
        struct ConstFolding {
            MemoryPtr src;
            // before const folding dst_mem is referencing a memory with expected desc but no handle(pointer)
            // and all primitives need this memory as argument have been setup with reference exactly to this memory.
            // after const folding, it's handle will be set to valid pointer with data filled.
            dnnl::memory dst_mem;
            std::string key;
        };

        bool constFolded;
        std::vector<ConstFolding> constFoldings;
        std::vector<IntermReorder> inputReorders;
        std::vector<IntermReorder> outputReorders;
        // when weightCache is not enabled (such as stream=1), brgconv weights may change due to
        // different shapes. Weights will be cached in privateWeightCache.
        // When weightCache is enabled, it holds weight ptr reference since weightCache does not hold the
        // reference
        std::unordered_map<std::string, MemoryPtr> privateWeightCache;

        // holding scratch pad memory object with callbacks responsible for change memory handle
        // when underlying scratch pad memory is resized.
        MemoryPtr scratchpadMem;

        dnnl::memory addConstFolding(MemoryPtr src, std::string privateSrcKey, DnnlMemoryDescPtr expectedWeightDesc);
        void doConstFolding(ConstFolding& cf);
};

}   // namespace intel_cpu
}   // namespace ov
