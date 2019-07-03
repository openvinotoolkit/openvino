// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/backend/backend.hpp>

#include <climits>
#include <cstring>

#include <string>
#include <memory>
#include <list>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <algorithm>
#include <map>
#include <streambuf>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <atomic>

#include <precision_utils.h>
#include <details/caseless.hpp>
#include <graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>

#include <vpu/parsed_config.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/file_system.hpp>

namespace vpu {

void BackEnd::getMetaData(
        const Model::Ptr& model,
        const std::vector<ie::CNNLayerPtr>& allLayers,
        std::vector<StageMetaInfo>& metaData) {
    VPU_PROFILE(getMetaData);

    metaData.clear();
    metaData.reserve(3 * model->numStages() / 2 + 1);

    std::unordered_set<ie::CNNLayerPtr> visitedLayers;

    auto getStageMeta = [&visitedLayers](const Stage& stage) -> StageMetaInfo {
        StageMetaInfo meta;

        meta.stageName = stage->name();
        meta.stageType = toString(stage->type());

        if (stage->numInjectedStages() > 0) {
            meta.stageName += " + injected[";
            meta.stageType += " + injected[";

            int ind = 0;
            for (const auto& injectedStageEdge : stage->injectedStageEdges()) {
                if (ind != 0) {
                    meta.stageName += ", ";
                    meta.stageType += ", ";
                }

                meta.stageName += injectedStageEdge->child()->name();
                meta.stageType += toString(injectedStageEdge->child()->type());

                ++ind;
            }

            meta.stageName += "]";
            meta.stageType += "]";
        }

        if (stage->origLayer() == nullptr) {
            meta.layerName = "<Extra>";
            meta.layerType = "<Extra>";
        } else {
            meta.layerName = stage->origLayer()->name;
            meta.layerType = stage->origLayer()->type;
            visitedLayers.insert(stage->origLayer());
        }

        return meta;
    };

    //
    // Add real stages
    //

    for (const auto& stage : model->getStages()) {
        if (stage->category() == StageCategory::Special) {
            continue;
        }

        auto meta = getStageMeta(stage);
        meta.status = ie::InferenceEngineProfileInfo::EXECUTED;
        metaData.emplace_back(std::move(meta));
    }

    //
    // Receive-Tensor time
    //

    // TODO : support config to disable timings and not to add this meta if it is not required by user
    StageMetaInfo receiveTensorMeta;
    receiveTensorMeta.stageName = "<Receive-Tensor>";
    receiveTensorMeta.stageType = "<Receive-Tensor>";
    receiveTensorMeta.layerName = "<Receive-Tensor>";
    receiveTensorMeta.layerType = "<Receive-Tensor>";
    receiveTensorMeta.status = ie::InferenceEngineProfileInfo::EXECUTED;
    metaData.emplace_back(std::move(receiveTensorMeta));

    //
    // Add special stages
    //

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::Special) {
            continue;
        }

        auto meta = getStageMeta(stage);
        meta.status = ie::InferenceEngineProfileInfo::OPTIMIZED_OUT;
        metaData.emplace_back(std::move(meta));
    }

    //
    // Add optimized layers
    //

    for (const auto& layer : allLayers) {
        if (visitedLayers.count(layer) != 0) {
            continue;
        }

        StageMetaInfo meta;
        meta.stageName = "<none>";
        meta.stageType = "<none>";
        meta.layerName = layer->name;
        meta.layerType = layer->type;
        meta.status = ie::InferenceEngineProfileInfo::LayerStatus::OPTIMIZED_OUT;
        metaData.emplace_back(std::move(meta));
    }
}

}  // namespace vpu
