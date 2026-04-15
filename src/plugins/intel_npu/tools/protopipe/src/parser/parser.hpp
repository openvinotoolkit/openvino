//
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "scenario/inference.hpp"
#include "scenario/scenario_graph.hpp"

struct ITermCriterion;
using ITermCriterionPtr = std::shared_ptr<ITermCriterion>;

struct WorkloadTypeDesc {
    std::string initial_value;
    std::vector<std::string> change_to;
    uint64_t change_interval;
    bool repeat;
};
struct StreamDesc {
    // NB: Commons parameters for all modes
    std::string name;
    uint64_t frames_interval_in_us;
    ScenarioGraph graph;
    InferenceParamsMap infer_params_map;
    ITermCriterionPtr criterion;
    // Mode specific params
    ModelsAttrMap<IAccuracyMetric::Ptr> metrics_map;
    ModelsAttrMap<IRandomGenerator::Ptr> initializers_map;
    ModelsAttrMap<std::string> input_data_map;
    ModelsAttrMap<std::string> output_data_map;
    std::optional<double> target_latency;
    std::optional<std::filesystem::path> per_iter_outputs_path;
    std::optional<WorkloadTypeDesc> workload_type;
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
    std::string npu_compiler_type;
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
