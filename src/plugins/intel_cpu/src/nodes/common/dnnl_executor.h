// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mkldnn_memory.h"
#include "mkldnn_primitive.h"

namespace MKLDNNPlugin {

class DnnlExecutor {
    protected:
        class IntermReorder {
            public:
                IntermReorder(const mkldnn::memory::desc& descSrc, const mkldnn::memory::desc& descDst, const mkldnn::engine& engine);
                void exec(mkldnn::memory& memSrc, mkldnn::memory& memDst, mkldnn::stream strm);
                const mkldnn::memory::desc& getSrcDesc() const { return m_descSrc; }
                const mkldnn::memory::desc& getDstDesc() const { return m_descDst; }

            private:
                mkldnn::reorder m_reorder;
                mkldnn::memory::desc m_descSrc;
                mkldnn::memory::desc m_descDst;
        };

    public:
        void exec(std::unordered_map<int, mkldnn::memory> primArgs, mkldnn::stream strm);
        virtual ~DnnlExecutor() = default;

    protected:
        DnnlExecutor() = default;
        MKLDNNPrimitive execPrim;
        // key is the port number for the primitive that needs memory reordering
        std::unordered_map<int, IntermReorder> inputReorders;
        std::unordered_map<int, IntermReorder> outputReorders;
};

}  // namespace MKLDNNPlugin
