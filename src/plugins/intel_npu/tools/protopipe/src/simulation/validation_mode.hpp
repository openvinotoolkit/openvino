//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <memory>

#include "simulation/computation.hpp"
#include "simulation/simulation.hpp"

class ValidationStrategy;
class ValSimulation : public Simulation {
public:
    struct Options {
        IAccuracyMetric::Ptr global_metric;
        ModelsAttrMap<IAccuracyMetric::Ptr> metrics_map;
        ModelsAttrMap<std::string> input_data_map;
        ModelsAttrMap<std::string> output_data_map;
        std::optional<std::filesystem::path> per_iter_outputs_path;
    };
    explicit ValSimulation(Simulation::Config&& cfg, Options&& opts);

    std::shared_ptr<PipelinedCompiled> compilePipelined(DummySources&& sources,
                                                        cv::GCompileArgs&& compile_args) override;
    std::shared_ptr<SyncCompiled> compileSync(DummySources&& sources, cv::GCompileArgs&& compiler_args) override;

private:
    Options m_opts;
    std::shared_ptr<ValidationStrategy> m_strategy;
    Computation m_comp;
};
