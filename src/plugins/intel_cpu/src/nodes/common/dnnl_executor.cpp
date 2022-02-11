// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_executor.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;

DnnlExecutor::IntermReorder::IntermReorder(const mkldnn::memory::desc& descSrc,
                                           const mkldnn::memory::desc& descDst,
                                           const mkldnn::engine& engine) : m_descSrc(descSrc), m_descDst(descDst) {
    auto reorderPd = mkldnn::reorder::primitive_desc(engine, descSrc, engine, descDst);
    m_reorder = mkldnn::reorder(reorderPd);
}

void DnnlExecutor::IntermReorder::exec(mkldnn::memory& memSrc, mkldnn::memory& memDst, mkldnn::stream strm) {
    m_reorder.execute(strm, memSrc, memDst);
}

void DnnlExecutor::exec(std::unordered_map<int, mkldnn::memory> primArgs, mkldnn::stream strm) {
    for (auto &inReorder : inputReorders) {
        if (primArgs.count(inReorder.first)) {
            mkldnn::memory memDst(inReorder.second.getDstDesc(), strm.get_engine());
            inReorder.second.exec(primArgs[inReorder.first], memDst, strm);
            primArgs[inReorder.first] = memDst;
        } else {
            IE_THROW() << "DnnlExecutor has reorder for input " << inReorder.first << ", but doesn't have source memory";
        }
    }
    std::unordered_map<int, mkldnn::memory> outputMem;
    for (auto &outReorder : outputReorders) {
        if (primArgs.count(outReorder.first)) {
            mkldnn::memory memSrc(outReorder.second.getSrcDesc(), strm.get_engine());
            outputMem[outReorder.first] = primArgs[outReorder.first];
            primArgs[outReorder.first] = memSrc;
        } else {
            IE_THROW() << "DnnlExecutor has reorder for output " << outReorder.first << ", but doesn't have destination memory";
        }
    }
    (*execPrim).execute(strm, primArgs);
    for (auto &outReorder : outputReorders) {
        outReorder.second.exec(primArgs[outReorder.first], outputMem[outReorder.first], strm);
    }
}

bool DnnlExecutor::needReordering() const {
    return !inputReorders.empty() || !outputReorders.empty();
}
