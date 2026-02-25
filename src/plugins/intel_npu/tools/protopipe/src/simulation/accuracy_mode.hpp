//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "simulation/computation.hpp"
#include "simulation/simulation.hpp"
#include "scenario/inference.hpp"

class AccuracyStrategy;
class AccuracySimulation : public Simulation {
public:
    struct Options {
        std::string ref_device;
        std::string tgt_device;
        std::string npu_compiler_type;
        IRandomGenerator::Ptr global_initializer;
        ModelsAttrMap<IRandomGenerator::Ptr> initializers_map;
        ModelsAttrMap<std::string> input_data_map;
        ModelsAttrMap<std::string> output_data_map;
        IAccuracyMetric::Ptr global_metric;
        ModelsAttrMap<IAccuracyMetric::Ptr> metrics_map;
    };

    explicit AccuracySimulation(Simulation::Config&& cfg, Options&& opts);

    std::shared_ptr<SyncCompiled> compileSync(const bool drop_frames) override;
    std::shared_ptr<PipelinedCompiled> compilePipelined(const bool drop_frames) override;

    std::shared_ptr<SyncCompiled> compileSync(DummySources&& ref_sources, DummySources&& tgt_sources,
                                              cv::GCompileArgs&& ref_compiler_args, cv::GCompileArgs&& tgt_compiler_args);
    std::shared_ptr<PipelinedCompiled> compilePipelined(DummySources&& ref_sources, DummySources&& tgt_sources,
                                                        cv::GCompileArgs&& ref_compile_args, cv::GCompileArgs&& tgt_compile_args);

private:
    Options m_opts;
    std::shared_ptr<AccuracyStrategy> m_strategy;
    Computation m_comp;
};
