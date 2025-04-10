//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "result.hpp"
#include "scenario/inference.hpp"
#include "scenario/scenario_graph.hpp"
#include "simulation/computation.hpp"
#include "utils/data_providers.hpp"

#include <filesystem>
#include <functional>
#include <memory>

struct InputIdx {
    uint32_t idx;
};

struct OutputIdx {
    uint32_t idx;
};

struct GraphInput {};
struct GraphOutput {};
struct GData {};
struct GOperation {
    using F = std::function<cv::GProtoArgs(const cv::GProtoArgs&)>;
    F on;
};

struct Dump {
    std::filesystem::path path;
};

struct Validate {
    using F = std::function<Result(const cv::Mat& lhs, const cv::Mat& rhs)>;
    F validator;
    std::vector<cv::Mat> reference;
};

struct InferDesc {
    std::string tag;
    LayersInfo input_layers;
    LayersInfo output_layers;
};

struct IBuildStrategy {
    using Ptr = std::shared_ptr<IBuildStrategy>;
    struct InferBuildInfo {
        std::vector<IDataProvider::Ptr> providers;
        std::vector<Meta> inputs_meta;
        std::vector<Meta> outputs_meta;
        const bool disable_copy;
    };
    // NB: Extend for any further node types needed
    virtual InferBuildInfo build(const InferDesc& infer) = 0;
};

class ComputationBuilder {
public:
    explicit ComputationBuilder(IBuildStrategy::Ptr strategy);

    struct Options {
        bool add_perf_meta;
    };

    Computation build(ScenarioGraph& graph, const InferenceParamsMap& infer_params, const Options& opts);

private:
    IBuildStrategy::Ptr m_strategy;
};
