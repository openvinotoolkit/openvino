//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph.hpp"
#include "simulation/dummy_source.hpp"

#include <opencv2/gapi/core.hpp>
#include <vector>

class Computation {
public:
    // NB: Holds information about Graph structure
    struct GraphDesc {
        const uint32_t max_parallel_branches;
    };

    Computation(cv::GComputation&& comp, cv::GCompileArgs&& args, std::vector<Meta>&& metas, GraphDesc&& desc);

    uint32_t getMaxParallelBranches() const;
    const std::vector<Meta>& getOutMeta() const;

    cv::GCompiled compile(cv::GMetaArgs&& in_meta, cv::GCompileArgs&& args = {});
    cv::GStreamingCompiled compileStreaming(cv::GMetaArgs&& in_meta, cv::GCompileArgs&& args = {});

private:
    cv::GComputation m_comp;
    cv::GCompileArgs m_compile_args;
    std::vector<Meta> m_out_meta;
    GraphDesc m_desc;
};

cv::GMetaArgs descr_of(const std::vector<DummySource::Ptr>& sources);
