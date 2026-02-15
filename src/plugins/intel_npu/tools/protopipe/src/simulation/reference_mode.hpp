//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "simulation/computation.hpp"
#include "simulation/simulation.hpp"

class ReferenceStrategy;
class CalcRefSimulation : public Simulation {
public:
    struct Options {
        // FIXME: In fact, there should be only input data initializers
        // and the path where to dump outputs
        IRandomGenerator::Ptr global_initializer;
        ModelsAttrMap<IRandomGenerator::Ptr> initializers_map;
        ModelsAttrMap<std::string> input_data_map;
        ModelsAttrMap<std::string> output_data_map;
    };

    explicit CalcRefSimulation(Simulation::Config&& cfg, Options&& opts);

    std::shared_ptr<PipelinedCompiled> compilePipelined(DummySources&& sources,
                                                        cv::GCompileArgs&& compile_args) override;
    std::shared_ptr<SyncCompiled> compileSync(DummySources&& sources, cv::GCompileArgs&& compiler_args) override;

private:
    Options m_opts;
    std::shared_ptr<ReferenceStrategy> m_strategy;
    Computation m_comp;
};
