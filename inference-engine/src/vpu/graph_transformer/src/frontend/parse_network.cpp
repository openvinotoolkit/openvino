// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <set>
#include <list>
#include <unordered_set>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <algorithm>
#include <utility>

#include <cpp/ie_cnn_network.h>
#include <graph_tools.hpp>
#include <details/caseless.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

namespace {

void runDFS(
        const std::string& networkName,
        const ie::CNNLayerPtr& layer,
        std::vector<ie::CNNLayerPtr>& out,
        std::unordered_map<ie::CNNLayerPtr, bool>& visitedMap) {
    const auto& env = CompileEnv::get();

    visitedMap[layer] = false;

    SmallVector<ie::CNNLayerPtr> nextLayers;
    for (const auto& output : layer->outData) {
        IE_ASSERT(output != nullptr);

        for (const auto& consumer : output->getInputTo()) {
            auto nextLayer = consumer.second;
            IE_ASSERT(nextLayer != nullptr);

            nextLayers.emplace_back(nextLayer);
        }
    }

    std::sort(nextLayers.begin(), nextLayers.end(),
              [](const ie::CNNLayerPtr& left, const ie::CNNLayerPtr& right) {
        ie::details::CaselessLess<std::string> cmp;
        return cmp(left->name, right->name);
    });

    for (const auto& nextLayer : nextLayers) {
        auto it = visitedMap.find(nextLayer);

        if (it != visitedMap.end()) {
            auto visited = it->second;

            if (!visited) {
                VPU_LOG_AND_THROW(env.log, "The network [%s] has a loop", networkName);
            }

            continue;
        }

        runDFS(networkName, nextLayer, out, visitedMap);
    }

    visitedMap[layer] = true;

    out.emplace_back(layer);
}

}  // namespace

void IeNetworkParser::clear() {
    networkInputs.clear();
    networkOutputs.clear();
    constDatas.clear();
    orderedLayers.clear();
}

void IeNetworkParser::checkNetwork(const ie::CNNNetwork& network) {
    const auto& env = CompileEnv::get();

    networkInputs = network.getInputsInfo();
    networkOutputs = network.getOutputsInfo();

    if (networkInputs.empty()) {
        VPU_LOG_AND_THROW(env.log, "No inputs detected in network %s", network.getName());
    }
    if (networkOutputs.empty()) {
        VPU_LOG_AND_THROW(env.log, "No outputs detected in network %s", network.getName());
    }

    for (const auto& netInput : networkInputs) {
        auto inputInfo = netInput.second;
        IE_ASSERT(inputInfo != nullptr);
    }

    for (const auto& netOutput : networkOutputs) {
        auto outputData = netOutput.second;
        IE_ASSERT(outputData != nullptr);
    }
}

void IeNetworkParser::parseNetworkDFS(const ie::CNNNetwork& network) {
    VPU_PROFILE(parseNetworkDFS);

    const auto& env = CompileEnv::get();

    ie::details::CaselessEq<std::string> cmp;

    env.log->debug("Parse network in DFS order");
    VPU_LOGGER_SECTION(env.log);

    //
    // Check network inputs and outputs.
    //

    checkNetwork(network);

    //
    // Collect all network input data.
    //

    std::unordered_set<ie::DataPtr> allInputDatas;

    for (const auto& netInput : networkInputs) {
        auto inputInfo = netInput.second;
        IE_ASSERT(inputInfo != nullptr);

        auto inputData = inputInfo->getInputData();
        IE_ASSERT(inputData != nullptr);

        allInputDatas.insert(inputData);
    }

    //
    // Collect all network const data.
    //

    for (const auto& layer : ie::CNNNetGetAllInputLayers(network)) {
        IE_ASSERT(layer != nullptr);

        if (!cmp(layer->type, "Const"))
            continue;

        if (layer->outData.size() != 1) {
            VPU_THROW_EXCEPTION
                    << "Const layer " << layer->name
                    << " has unsupported number of outputs "
                    << layer->outData.size();
        }

        if (layer->blobs.size() != 1) {
            VPU_THROW_EXCEPTION
                    << "Const layer " << layer->name
                    << " has unsupported number of blobs "
                    << layer->blobs.size();
        }

        auto constData = layer->outData[0];
        IE_ASSERT(constData != nullptr);

        auto constBlob = layer->blobs.begin()->second;
        IE_ASSERT(constBlob != nullptr);

        constDatas[constData] = constBlob;

        allInputDatas.insert(constData);
    }

    //
    // Collect initial layers.
    //

    std::unordered_set<ie::CNNLayerPtr> visitedInitialLayers;
    SmallVector<ie::CNNLayerPtr> initialLayers;

    for (const auto& inputData : allInputDatas) {
        for (const auto& consumer : inputData->getInputTo()) {
            auto initialLayer = consumer.second;
            IE_ASSERT(initialLayer != nullptr);

            if (visitedInitialLayers.count(initialLayer) > 0)
                continue;

            bool allInputsAvailable = true;
            for (const auto& in : initialLayer->insData) {
                auto input = in.lock();
                IE_ASSERT(input != nullptr);

                if (allInputDatas.count(input) == 0) {
                    allInputsAvailable = false;
                    break;
                }
            }

            if (allInputsAvailable) {
                visitedInitialLayers.insert(initialLayer);
                initialLayers.emplace_back(std::move(initialLayer));
            }
        }
    }

    IE_ASSERT(!initialLayers.empty());

    //
    // Run recursive DFS algorithm.
    //

    std::sort(initialLayers.begin(), initialLayers.end(),
              [](const ie::CNNLayerPtr& left, const ie::CNNLayerPtr& right) {
        ie::details::CaselessLess<std::string> cmp;
        return cmp(left->name, right->name);
    });

    std::unordered_map<ie::CNNLayerPtr, bool> visitedMap;
    for (const auto& layer : initialLayers) {
        runDFS(network.getName(), layer, orderedLayers, visitedMap);
    }

    //
    // Reverse the result.
    //

    std::reverse(orderedLayers.begin(), orderedLayers.end());
}

void IeNetworkParser::parseNetworkBFS(const ie::CNNNetwork& network) {
    VPU_PROFILE(parseNetworkBFS);

    const auto& env = CompileEnv::get();

    env.log->debug("Parse network in BFS order");
    VPU_LOGGER_SECTION(env.log);

    ie::details::CaselessEq<std::string> cmp;

    env.log->debug("parse network %s", network.getName());

    //
    // Check network inputs and outputs.
    //

    checkNetwork(network);

    //
    // Collect input datas.
    //

    std::unordered_set<ie::DataPtr> availableData;

    for (const auto& netInput : networkInputs) {
        auto inputInfo = netInput.second;
        IE_ASSERT(inputInfo != nullptr);

        auto inputData = inputInfo->getInputData();
        IE_ASSERT(inputData != nullptr);

        availableData.insert(inputData);
    }

    //
    // Collect all network const data.
    //

    for (const auto& layer : ie::CNNNetGetAllInputLayers(network)) {
        IE_ASSERT(layer != nullptr);

        if (!cmp(layer->type, "Const"))
            continue;

        if (layer->outData.size() != 1) {
            VPU_THROW_EXCEPTION
                    << "Const layer " << layer->name
                    << " has unsupported number of outputs "
                    << layer->outData.size();
        }

        if (layer->blobs.size() != 1) {
            VPU_THROW_EXCEPTION
                    << "Const layer " << layer->name
                    << " has unsupported number of blobs "
                    << layer->blobs.size();
        }

        auto constData = layer->outData[0];
        IE_ASSERT(constData != nullptr);

        auto constBlob = layer->blobs.begin()->second;
        IE_ASSERT(constBlob != nullptr);

        constDatas[constData] = constBlob;

        availableData.insert(constData);
    }

    //
    // Collect initial layers.
    //

    std::unordered_set<ie::CNNLayerPtr> visitedInitialLayers;
    std::list<ie::CNNLayerPtr> layersToHandle;

    for (const auto& inputData : availableData) {
        for (const auto& consumer : inputData->getInputTo()) {
            auto initialLayer = consumer.second;
            IE_ASSERT(initialLayer != nullptr);

            if (visitedInitialLayers.count(initialLayer) > 0)
                continue;

            bool allInputsAvailable = true;
            for (const auto& in : initialLayer->insData) {
                auto input = in.lock();
                IE_ASSERT(input != nullptr);

                if (availableData.count(input) == 0) {
                    allInputsAvailable = false;
                    break;
                }
            }

            if (allInputsAvailable) {
                visitedInitialLayers.insert(initialLayer);
                layersToHandle.emplace_back(std::move(initialLayer));
            }
        }
    }

    IE_ASSERT(!layersToHandle.empty());

    //
    // Traversing the topology (BFS).
    //

    std::unordered_set<ie::CNNLayerPtr> parsedLayers;

    size_t loopTracker = 0;

    while (!layersToHandle.empty()) {
        auto layer = layersToHandle.front();

        if (layersToHandle.size() == loopTracker) {
            VPU_THROW_EXCEPTION
                    << "Inputs for layer " << layer->name
                    << "(and " << loopTracker - 1 << " more layers) can not be computed";
        }

        layersToHandle.pop_front();

        bool allInputsAvailable = true;
        for (const auto& in : layer->insData) {
            auto inData = in.lock();
            IE_ASSERT(inData != nullptr);

            if (availableData.find(inData) == availableData.end()) {
                allInputsAvailable = false;
                break;
            }
        }

        if (!allInputsAvailable) {
            layersToHandle.emplace_back(std::move(layer));
            loopTracker++;
            continue;
        }

        if (parsedLayers.find(layer) == parsedLayers.end()) {
            orderedLayers.emplace_back(layer);
            parsedLayers.insert(layer);
        }

        // Add children to the list to verify.
        for (const auto& out : layer->outData) {
            IE_ASSERT(out != nullptr);
            availableData.insert(out);

            // New data added -> have to reset loop tracking.
            loopTracker = 0;

            for (const auto& layerInfo : out->getInputTo()) {
                auto consumer = layerInfo.second;
                IE_ASSERT(consumer != nullptr);

                auto it = std::find(layersToHandle.begin(), layersToHandle.end(), consumer);
                if (it == layersToHandle.end()) {
                    layersToHandle.emplace_back(std::move(consumer));
                }
            }
        }
    }
}

}  // namespace vpu
