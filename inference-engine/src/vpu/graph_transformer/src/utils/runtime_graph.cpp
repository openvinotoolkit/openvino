// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/runtime_graph.hpp"

#include <cnn_network_impl.hpp>
#include <exec_graph_info.hpp>

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <memory>

using namespace InferenceEngine;

namespace vpu {

    InferenceEngine::ICNNNetwork::Ptr buildRuntimeGraph(GraphMetaInfo &graphMetaInfo, const std::vector<float>& perfInfo) {
        auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>();
        net->setName(graphMetaInfo.graphName);

        std::map<size_t, CNNLayerPtr> stageMetaIndexToLayer;

        auto createLayerFromMeta = [&](const StageMetaInfo &stageMetaInfo) -> CNNLayer::Ptr {
            auto layer = std::make_shared<CNNLayer>(LayerParams{stageMetaInfo.stageName,
                                                                stageMetaInfo.layerType,
                                                                Precision::FP16});

            layer->params[ExecGraphInfoSerialization::ORIGINAL_NAMES] = stageMetaInfo.layerName;
            layer->params[ExecGraphInfoSerialization::IMPL_TYPE] = stageMetaInfo.stageType;
            layer->params[ExecGraphInfoSerialization::EXECUTION_ORDER] = std::to_string(stageMetaInfo.execOrder);

            std::stringstream layoutStream;
            int ind = 0;
            for (auto &outLayout : stageMetaInfo.outLayouts) {
                if (ind == 0) {
                    layoutStream << outLayout;
                    ind++;
                    continue;
                }
                layoutStream << ',' << outLayout;
            }
            layer->params[ExecGraphInfoSerialization::OUTPUT_LAYOUTS] = layoutStream.str();

            std::string outPrecisionsStr;
            ind = 0;
            for (auto &outPrecision : stageMetaInfo.outPrecisions) {
                if (ind == 0) {
                    outPrecisionsStr += outPrecision.name();
                    ind++;
                    continue;
                }
                outPrecisionsStr += ',' + std::string(outPrecision.name());
            }
            layer->params[ExecGraphInfoSerialization::OUTPUT_PRECISIONS] = outPrecisionsStr;

            if (stageMetaInfo.execOrder < 0) {
                layer->params[ExecGraphInfoSerialization::PERF_COUNTER] = "not_executed";
            } else {
                layer->params[ExecGraphInfoSerialization::PERF_COUNTER] = std::to_string(stageMetaInfo.execTime);
            }

            return layer;
        };

        //
        // Write performance counts
        //

        const auto deviceTimings = perfInfo.data();
        auto deviceTimingsCount = perfInfo.size();

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

        //
        // Add all stages to network
        //

        for (std::size_t i = 0; i < graphMetaInfo.stagesMeta.size(); i++) {
            const auto stageMetaData = graphMetaInfo.stagesMeta[i];

            if (stageMetaData.status == ie::InferenceEngineProfileInfo::LayerStatus::OPTIMIZED_OUT ||
                stageMetaData.stageName == "<Receive-Tensor>" ||
                stageMetaData.stageName == "<none>") {
                continue;
            }

            auto layer = createLayerFromMeta(stageMetaData);
            stageMetaIndexToLayer.insert(std::make_pair(i, layer));
            net->addLayer(layer);
        }

        //
        // Add all edges to network
        //

        for (const auto &dataMetaData : graphMetaInfo.datasMeta) {
            ::InferenceEngine::DataPtr data;

            auto parent = stageMetaIndexToLayer[dataMetaData.parentIndex];
            data = std::make_shared<::InferenceEngine::Data>(dataMetaData.name, dataMetaData.desc);
            parent->outData.push_back(data);
            data->getCreatorLayer() = parent;

            for (auto &childMetaIndex : dataMetaData.childrenIndices) {
                auto child = stageMetaIndexToLayer[childMetaIndex];
                data->getInputTo()[child->name] = child;
                child->insData.push_back(data);
            }
        }

        //
        // Specify inputs data
        //

        for (std::size_t i = 0; i < graphMetaInfo.stagesMeta.size(); i++) {
            const auto stageMetaData = graphMetaInfo.stagesMeta[i];

            if (stageMetaData.inputsNum != 0 ||
                stageMetaData.stageName == "<Receive-Tensor>" ||
                stageMetaData.stageName == "<none>") {
                continue;
            }

            auto input = stageMetaIndexToLayer[i];
            auto inputInfo = std::make_shared<InputInfo>();
            inputInfo->setInputData(input->outData[0]);
            net->setInputInfo(inputInfo);
        }

        return net;
    }
}  // namespace vpu
