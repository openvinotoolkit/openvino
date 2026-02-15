//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "simulation/computation.hpp"
#include "simulation/computation_builder.hpp"
#include "simulation/simulation.hpp"

struct PerformanceStrategy;
class PerformanceSimulation : public Simulation {
public:
    struct Options {
        IRandomGenerator::Ptr global_initializer;
        ModelsAttrMap<IRandomGenerator::Ptr> initializers_map;
        ModelsAttrMap<std::string> input_data_map;
        const bool inference_only;
        std::optional<double> target_latency;
    };
    explicit PerformanceSimulation(Simulation::Config&& cfg, Options&& opts);

    std::shared_ptr<PipelinedCompiled> compilePipelined(DummySources&& sources,
                                                        cv::GCompileArgs&& compiler_args) override;
    std::shared_ptr<SyncCompiled> compileSync(const bool drop_frames) override;

private:
    Options m_opts;
    std::shared_ptr<PerformanceStrategy> m_strategy;
    Computation m_comp;
};

struct PerformanceStrategy : public IBuildStrategy {
    explicit PerformanceStrategy(const PerformanceSimulation::Options& opts);
    IBuildStrategy::InferBuildInfo build(const InferDesc& infer) override;

    const PerformanceSimulation::Options& opts;
};
