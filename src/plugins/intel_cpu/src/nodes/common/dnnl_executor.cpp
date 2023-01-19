// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_executor.h"

using namespace dnnl;

namespace ov {
namespace intel_cpu {

DnnlExecutor::IntermReorder::IntermReorder(const dnnl::memory::desc& descSrc,
                                           const dnnl::memory::desc& descDst,
                                           const dnnl::engine& engine) : m_descSrc(descSrc), m_descDst(descDst) {
    auto reorderPd = dnnl::reorder::primitive_desc(engine, descSrc, engine, descDst);
    m_reorder = dnnl::reorder(reorderPd);
}

void DnnlExecutor::IntermReorder::exec(dnnl::memory& memSrc, dnnl::memory& memDst, dnnl::stream strm) {
    m_reorder.execute(strm, memSrc, memDst);
}

void DnnlExecutor::exec(std::unordered_map<int, dnnl::memory> primArgs, dnnl::stream strm) {
    for (auto &inReorder : inputReorders) {
        if (primArgs.count(inReorder.first)) {
            dnnl::memory memDst(inReorder.second.getDstDesc(), strm.get_engine());
            inReorder.second.exec(primArgs[inReorder.first], memDst, strm);
            primArgs[inReorder.first] = memDst;
        } else {
            IE_THROW() << "DnnlExecutor has reorder for input " << inReorder.first << ", but doesn't have source memory";
        }
    }
    std::unordered_map<int, dnnl::memory> outputMem;
    for (auto &outReorder : outputReorders) {
        if (primArgs.count(outReorder.first)) {
            dnnl::memory memSrc(outReorder.second.getSrcDesc(), strm.get_engine());
            outputMem[outReorder.first] = primArgs[outReorder.first];
            primArgs[outReorder.first] = memSrc;
        } else {
            IE_THROW() << "DnnlExecutor has reorder for output " << outReorder.first << ", but doesn't have destination memory";
        }
    }
    execPrim.execute(strm, primArgs);
    for (auto &outReorder : outputReorders) {
        outReorder.second.exec(primArgs[outReorder.first], outputMem[outReorder.first], strm);
    }
}

bool DnnlExecutor::needReordering() const {
    return !inputReorders.empty() || !outputReorders.empty();
}

dnnl::primitive DnnlExecutor::getExecPrim() const {
    return execPrim;
}

const_dnnl_primitive_desc_t DnnlExecutor::getPrimitiveDesc() const {
    return execPrim.get_primitive_desc();
}

dnnl::memory::desc DnnlExecutor::getSrcDesc() const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::src_md);

    return md->getDnnlDesc();
}

dnnl::memory::desc DnnlExecutor::getWeightDesc() const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::weights_md);

    return md->getDnnlDesc();
}

dnnl::memory::desc DnnlExecutor::getDstDesc() const {
    auto pd = getPrimitiveDesc();
    auto md = DnnlExtensionUtils::query_md(pd, dnnl::query::dst_md);

    return md->getDnnlDesc();
}

impl_desc_type DnnlExecutor::getImplementationType() const {
    auto pd = getPrimitiveDesc();
    return parse_impl_name(DnnlExtensionUtils::query_impl_info_str(pd));
}

}  // namespace intel_cpu
}   // namespace ov
