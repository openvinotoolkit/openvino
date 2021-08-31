// Copyright (C) 2018-2021 Intel Corporation
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
#include <legacy/graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>

#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/utils/profiling.hpp>

namespace vpu {

void BackEnd::getMetaData(
        const Model& model,
        const std::vector<ie::CNNLayerPtr>& allLayers,
        GraphMetaInfo& graphMeta) {
    VPU_PROFILE(getMetaData);

    std::vector<StageMetaInfo> stagesMeta;
    std::vector<DataMetaInfo> datasMeta;

    std::unordered_set<ie::CNNLayerPtr> visitedLayers;
    int execOrder{};
    StageMap<size_t> stageToMetaIndex;

    stagesMeta.reserve(3 * model->numStages() / 2 + 1);
    datasMeta.reserve(3 * model->numDatas() / 2 + 1);

    graphMeta.graphName = model->name();

    auto getStageMeta = [&](const Stage& stage) -> StageMetaInfo {
        StageMetaInfo stageMeta;

        stageMeta.displayStageName = stageMeta.stageName = stage->name();
        stageMeta.stageType = toString(stage->type());

        if (stage->category() != StageCategory::Special) {
            stageMeta.execOrder = execOrder++;
        } else {
            stageMeta.execOrder = -1;
        }

        if (const auto injectedStage = stage->injectedStage()) {
            stageMeta.displayStageName += " + injected[";
            stageMeta.stageType += " + injected[";

            stageMeta.displayStageName += injectedStage->name();
            stageMeta.stageType += toString(injectedStage->type());

            stageMeta.displayStageName += "]";
            stageMeta.stageType += "]";
        }

        if (stage->origLayer() == nullptr) {
            stageMeta.layerName = "";
            stageMeta.layerType = "<Extra>";
        } else {
            const auto& origLayer = stage->origLayer();
            stageMeta.layerName = origLayer->params.count("originalLayersNames") ? origLayer->params["originalLayersNames"] :
                                  origLayer->name;
            stageMeta.layerType = origLayer->type;
            visitedLayers.insert(origLayer);
        }

        return stageMeta;
    };

    auto getDataMeta = [&](const Data& data) -> DataMetaInfo {
        DataMetaInfo dataMeta;

        dataMeta.name = data->name();
        dataMeta.desc = data->desc().toTensorDesc();

        if (data->usage() == DataUsage::Input) {
            // Create fake input layer
            StageMetaInfo inputInfo;

            inputInfo.layerType = "Input";
            inputInfo.layerName = inputInfo.stageName = inputInfo.displayStageName = data->name();
            inputInfo.stageType = "NONE";
            inputInfo.outPrecisions.push_back(dataMeta.desc.getPrecision());
            inputInfo.outLayouts.push_back(dataMeta.desc.getLayout());
            stagesMeta.push_back(std::move(inputInfo));

            dataMeta.parentIndex = stagesMeta.size() - 1;
        }  else {
            auto it = stageToMetaIndex.find(data->producer());

            if (it != stageToMetaIndex.end()) {
                StageMetaInfo& meta = stagesMeta[it->second];

                meta.outPrecisions.push_back(dataMeta.desc.getPrecision());
                meta.outLayouts.push_back(dataMeta.desc.getLayout());

                dataMeta.parentIndex = it->second;
            }
        }

        if (data->usage() != DataUsage::Output) {
            size_t prIndex;
            if (data->usage() == DataUsage::Input) {
                prIndex = stagesMeta.size() - 1;
            } else {
                prIndex = stageToMetaIndex.find(data->producer())->second;
            }

            for (const auto &child : data->consumers()) {
                auto it = stageToMetaIndex.find(child);

                if (it != stageToMetaIndex.end()) {
                    StageMetaInfo& meta = stagesMeta[it->second];
                    stagesMeta[prIndex].childrenNum++;
                    meta.parentIndices.push_back(prIndex);
                    meta.inputDims.push_back(dataMeta.desc.getDims());
                    meta.inputPrecisions.push_back(dataMeta.desc.getPrecision());
                    dataMeta.childrenIndices.push_back(it->second);
                }
            }
        }

        return dataMeta;
    };

    //
    // Add real stages
    //

    for (const auto& stage : model->getStages()) {
        if (stage->category() == StageCategory::Special) {
            continue;
        }

        auto stageMeta = getStageMeta(stage);

        stageMeta.status = ie::InferenceEngineProfileInfo::EXECUTED;
        stagesMeta.emplace_back(std::move(stageMeta));
        stageToMetaIndex[stage] = stagesMeta.size() - 1;
    }

    //
    // Receive-Tensor time
    //

    // TODO : support config to disable timings and not to add this meta if it is not required by user
    StageMetaInfo receiveTensorMeta;
    receiveTensorMeta.displayStageName = receiveTensorMeta.stageName = "<Receive-Tensor>";
    receiveTensorMeta.stageType = "<Receive-Tensor>";
    receiveTensorMeta.layerName = "<Receive-Tensor>";
    receiveTensorMeta.layerType = "<Receive-Tensor>";
    receiveTensorMeta.status = ie::InferenceEngineProfileInfo::EXECUTED;
    stagesMeta.emplace_back(std::move(receiveTensorMeta));

    //
    // Add special stages
    //

    for (const auto& stage : model->getStages()) {
        if (stage->category() != StageCategory::Special) {
            continue;
        }

        auto stageMeta = getStageMeta(stage);
        stageMeta.status = ie::InferenceEngineProfileInfo::NOT_RUN;
        stagesMeta.emplace_back(std::move(stageMeta));
        stageToMetaIndex[stage] = stagesMeta.size() - 1;
    }

    //
    // Add optimized layers
    //

    for (const auto& layer : allLayers) {
        if (visitedLayers.count(layer) != 0) {
            continue;
        }

        StageMetaInfo stageMeta;
        stageMeta.stageName = "<none>";
        stageMeta.stageType = "<none>";
        stageMeta.layerName = layer->name;
        stageMeta.layerType = layer->type;
        stageMeta.status = ie::InferenceEngineProfileInfo::LayerStatus::OPTIMIZED_OUT;
        stagesMeta.emplace_back(std::move(stageMeta));
    }

    //
    // Add data info
    //

    for (const auto& data : model->datas()) {
        if (data->usage() != DataUsage::Input &&
            data->usage() != DataUsage::Intermediate &&
            data->usage() != DataUsage::Output) {
            continue;
        }

        auto dataMeta = getDataMeta(data);
        datasMeta.emplace_back(std::move(dataMeta));
    }

    graphMeta.stagesMeta = std::move(stagesMeta);
    graphMeta.datasMeta = std::move(datasMeta);
}

}  // namespace vpu
