//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "result.hpp"
#include "scenario/criterion.hpp"
#include "scenario/inference.hpp"
#include "scenario/scenario_graph.hpp"
#include "simulation/dummy_source.hpp"

#include <opencv2/gapi/infer.hpp>  // cv::gapi::GNetPackage

struct ICompiled {
    using Ptr = std::shared_ptr<ICompiled>;
    virtual Result run(ITermCriterion::Ptr) = 0;
};

struct PipelinedCompiled : public ICompiled {};
struct SyncCompiled : public ICompiled {};

using DummySources = std::vector<DummySource::Ptr>;

class Simulation {
public:
    using Ptr = std::shared_ptr<Simulation>;

    struct Config {
        std::string stream_name;
        uint64_t frames_interval_in_us;
        bool disable_high_resolution_timer;
        ScenarioGraph graph;
        InferenceParamsMap params;
    };

    explicit Simulation(Config&& cfg);

    virtual std::shared_ptr<PipelinedCompiled> compilePipelined(const bool drop_frames);
    virtual std::shared_ptr<SyncCompiled> compileSync(const bool drop_frames);

    virtual ~Simulation() = default;

protected:
    virtual std::shared_ptr<PipelinedCompiled> compilePipelined(DummySources&& sources,
                                                                cv::GCompileArgs&& compile_args);
    virtual std::shared_ptr<SyncCompiled> compileSync(DummySources&& sources, cv::GCompileArgs&& compile_args);

    std::vector<DummySource::Ptr> createSources(const bool drop_frames);
    cv::gapi::GNetPackage getNetworksPackage() const;

protected:
    Config m_cfg;
};
