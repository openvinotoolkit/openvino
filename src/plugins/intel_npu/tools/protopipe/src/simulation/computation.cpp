//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation/computation.hpp"

Computation::Computation(cv::GComputation&& comp, cv::GCompileArgs&& args, std::vector<Meta>&& metas, GraphDesc&& desc)
        : m_comp(std::move(comp)),
          m_compile_args(std::move(args)),
          m_out_meta(std::move(metas)),
          m_desc(std::move(desc)) {
}

uint32_t Computation::getMaxParallelBranches() const {
    return m_desc.max_parallel_branches;
}

const std::vector<Meta>& Computation::getOutMeta() const {
    return m_out_meta;
}

cv::GCompiled Computation::compile(cv::GMetaArgs&& in_meta, cv::GCompileArgs&& args) {
    auto compile_args = m_compile_args;
    compile_args += std::move(args);
    return m_comp.compile(std::move(in_meta), std::move(compile_args));
}

cv::GStreamingCompiled Computation::compileStreaming(cv::GMetaArgs&& in_meta, cv::GCompileArgs&& args) {
    auto compile_args = m_compile_args;
    compile_args += std::move(args);
    return m_comp.compileStreaming(std::move(in_meta), std::move(compile_args));
}

cv::GMetaArgs descr_of(const std::vector<DummySource::Ptr>& sources) {
    cv::GMetaArgs meta;
    meta.reserve(sources.size());
    for (auto src : sources) {
        meta.push_back(src->descr_of());
    }
    return meta;
}
