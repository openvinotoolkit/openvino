// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/runtime_graph.hpp"

#include <legacy/ie_util_internal.hpp>
#include <ie_ngraph_utils.hpp>
#include <exec_graph_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/ngraph.hpp>

#include <vector>
#include <map>
#include <utility>
#include <memory>

using namespace InferenceEngine;

namespace vpu {

namespace {

std::map<std::string, std::string> extractMeta(const StageMetaInfo&);

}  // namespace

InferenceEngine::CNNNetwork buildRuntimeGraph(GraphMetaInfo& graphMetaInfo, const std::vector<float>& perfInfo) {
    std::map<size_t, std::shared_ptr<ngraph::Node>> stageMetaIndexToNode;
    std::function<void(size_t)> createNodeFromMeta;

    ngraph::ResultVector results;
    ngraph::ParameterVector params;

    //
    // Write performance counts
    //

    const auto deviceTimings = perfInfo.data();
    const auto deviceTimingsCount = perfInfo.size();

    if (deviceTimingsCount > 0) {
        std::size_t timeIndex = 0;

        for (auto &stageMeta : graphMetaInfo.stagesMeta) {
            if (stageMeta.status == ie::InferenceEngineProfileInfo::EXECUTED &&
                timeIndex < deviceTimingsCount) {
                stageMeta.execTime += deviceTimings[timeIndex];
                timeIndex++;
            }
        }
    }

    auto getInputs = [&](const StageMetaInfo& stageMeta) {
        ngraph::OutputVector inputs;

        for (int i = 0; i < stageMeta.parentIndices.size(); i++) {
            const auto prIndex = stageMeta.parentIndices[i];
            const auto dims = stageMeta.inputDims[i];
            const auto precision = stageMeta.inputPrecisions[i];

            if (stageMetaIndexToNode.count(prIndex) == 0) {
                // Create parent node if it doesn't exist
                createNodeFromMeta(prIndex);
            }
            const auto outSize = stageMetaIndexToNode[prIndex]->get_output_size();
            stageMetaIndexToNode[prIndex]->set_output_size(outSize + 1);
            stageMetaIndexToNode[prIndex]->set_output_type(outSize, InferenceEngine::details::convertPrecision(precision), ngraph::PartialShape(dims));
            inputs.push_back(stageMetaIndexToNode[prIndex]->output(outSize));
        }
        return inputs;
    };

    createNodeFromMeta = [&](size_t index) {
        const auto stageMeta = graphMetaInfo.stagesMeta[index];

        const auto inputs = getInputs(stageMeta);

        std::shared_ptr<ngraph::Node> node;
        if (stageMeta.stageType == "Input") {
            params.emplace_back(std::make_shared<ngraph::op::Parameter>());
            node = params.back();
        } else if (stageMeta.childrenNum == 0) {
            results.emplace_back(std::make_shared<ngraph::op::Result>(inputs.back()));
            node = results.back();
        } else {
            node = std::make_shared<ExecGraphInfoSerialization::ExecutionNode>(inputs, 0);
        }

        node->set_friendly_name(stageMeta.stageName);
        const auto metaData = extractMeta(stageMeta);
        for (const auto& meta : metaData) {
            node->get_rt_info()[meta.first] = std::make_shared<::ngraph::VariantWrapper<std::string>>(meta.second);
        }

        stageMetaIndexToNode[index] = node;
    };

    //
    // Add stages to graph
    //

    for (std::size_t i = 0; i < graphMetaInfo.stagesMeta.size(); i++) {
        const auto stageMeta = graphMetaInfo.stagesMeta[i];

        if (stageMeta.status == ie::InferenceEngineProfileInfo::LayerStatus::OPTIMIZED_OUT ||
            stageMetaIndexToNode.count(i) != 0 ||
            stageMeta.stageName == "<Receive-Tensor>" ||
            stageMeta.stageName == "<none>") {
            continue;
        }
        createNodeFromMeta(i);
    }

    auto ngraph = std::make_shared<ngraph::Function>(results, params, graphMetaInfo.graphName);
    InferenceEngine::CNNNetwork net(ngraph);
    return net;
}

namespace {

std::map<std::string, std::string> extractMeta(const StageMetaInfo& stageMeta) {
    std::map<std::string, std::string> serializationInfo;

    serializationInfo[ExecGraphInfoSerialization::ORIGINAL_NAMES] = stageMeta.layerName;
    serializationInfo[ExecGraphInfoSerialization::IMPL_TYPE] = stageMeta.stageType;
    serializationInfo[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(stageMeta.execOrder);
    serializationInfo[ExecGraphInfoSerialization::LAYER_TYPE] = stageMeta.layerType;
    if (stageMeta.execOrder < 0 || stageMeta.execTime == 0) {
        serializationInfo[ExecGraphInfoSerialization::PERF_COUNTER] = "not_executed";
    } else {
        serializationInfo[ExecGraphInfoSerialization::PERF_COUNTER] = std::to_string(stageMeta.execTime);
    }
    std::stringstream layoutStream;
    int ind = 0;
    for (auto &outLayout : stageMeta.outLayouts) {
        if (ind == 0) {
            layoutStream << outLayout;
            ind++;
            continue;
        }
        layoutStream << ',' << outLayout;
    }
    serializationInfo[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = layoutStream.str();

    std::string outPrecisionsStr;
    Precision runtimePrecision {Precision::I32};
    ind = 0;
    for (auto &outPrecision : stageMeta.outPrecisions) {
        // if we have any output precision not equal I32 -> we assume runtimePrecision is FP16
        if (outPrecision != Precision::I32) {
            runtimePrecision = Precision::FP16;
        }
        if (ind == 0) {
            outPrecisionsStr += outPrecision.name();
            ind++;
            continue;
        }
        outPrecisionsStr += ',' + std::string(outPrecision.name());
    }
    serializationInfo[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = outPrecisionsStr;
    serializationInfo[ExecGraphInfoSerialization::RUNTIME_PRECISION] = runtimePrecision.name();
    return serializationInfo;
}

}  // namespace

}  // namespace vpu
