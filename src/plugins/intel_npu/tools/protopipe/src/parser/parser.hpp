//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "scenario/criterion.hpp"
#include "scenario/inference.hpp"
#include "scenario/scenario_graph.hpp"

struct StreamDesc {
    // NB: Commons parameters for all modes
    std::string name;
    uint64_t frames_interval_in_us;
    ScenarioGraph graph;
    InferenceParamsMap infer_params_map;
    ITermCriterion::Ptr criterion;
    // Mode specific params
    ModelsAttrMap<IAccuracyMetric::Ptr> metrics_map;
    ModelsAttrMap<IRandomGenerator::Ptr> initializers_map;
    ModelsAttrMap<std::string> input_data_map;
    ModelsAttrMap<std::string> output_data_map;
    std::optional<double> target_latency;
    std::optional<std::filesystem::path> per_iter_outputs_path;
};

struct ScenarioDesc {
    std::string name;
    std::vector<StreamDesc> streams;
    bool disable_high_resolution_timer;
};

struct Config {
    IRandomGenerator::Ptr initializer;
    IAccuracyMetric::Ptr metric;
    bool disable_high_resolution_timer;
    std::vector<ScenarioDesc> scenarios;
};

struct ReplaceBy {
    std::string device;
};

struct IScenarioParser {
    virtual Config parseScenarios(const ReplaceBy& replace_by) = 0;
    virtual ~IScenarioParser() = default;
};

class ScenarioParser : public IScenarioParser {
public:
    ScenarioParser(const std::string& filepath);
    Config parseScenarios(const ReplaceBy& replace_by) override;

private:
    std::string m_filepath;
};
