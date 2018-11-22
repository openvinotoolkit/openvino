// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/graph_tools.hpp>
#include <gmock/gmock-generated-function-mockers.h>
#include <gmock/gmock-generated-matchers.h>
#include <gmock/gmock-more-actions.h>
#include "mock_icnn_network.hpp"
#include "cpp/ie_cnn_network.h"
#include "details/ie_cnn_network_tools.h"

namespace GraphTest {

using namespace InferenceEngine;
using namespace std;

/**
 * @class all layers are input without connection.
 * Input layers are defined by absence of ins data.
 */
class GraphTestsBase : public ::testing::Test {

 protected:

    MOCK_METHOD2(visited, void(size_t, int));
    MOCK_METHOD2(visited2, void(size_t, int));

    std::vector<CNNLayerPtr> layers;
    std::vector<std::vector<DataPtr>> datas;

    MockICNNNetwork mockNet;
    InferenceEngine::CNNNetwork wrap = InferenceEngine::CNNNetwork(&mockNet);

    /**
     * layers that are used as lhs in oriented connect operations
     */
    std::unordered_set<CNNLayerPtr> lhsLayers;
    std::unordered_set<CNNLayerPtr> rhsLayers;

    virtual void prepareInputs(InputsDataMap &inputsMap, int batchSize = 1) {
        auto prepareInputsInternal = [&inputsMap, &batchSize](std::unordered_set<CNNLayerPtr> & layersSet) {
            for (auto layer = layersSet.begin(); layer != layersSet.end(); layer++) {
                if ((*layer)->insData.empty()) {
                    auto info = make_shared<InputInfo>();
                    auto data = make_shared<Data>((*layer)->name, Precision::FP32, Layout::NC);
                    SizeVector dims = data->getDims();
                    dims.push_back(batchSize);
                    dims.push_back(batchSize);
                    data->setDims(dims);
                    for (auto output : (*layer)->outData) {
                        data->getInputTo() = output->inputTo;
                    }
                    data->creatorLayer = (*layer);
                    info->setInputData(data);
                    inputsMap[(*layer)->name] = info;
                }
            }
        };
        prepareInputsInternal(lhsLayers);
        prepareInputsInternal(rhsLayers);
    }

    CNNLayerPtr layerByName(std::string name) {
        auto sorted = InferenceEngine::details::CNNNetSortTopologically(mockNet);

        auto i = std::find_if(sorted.begin(), sorted.end(), [&](CNNLayerPtr l){
            return l->name == name;
        });
        if (i != sorted.end()) {
            return *i;
        }
        return nullptr;
    }
    #define ASSERT_CONNECTION(a, b) \
        ASSERT_TRUE(assertConnection(#a, #b));

    void ASSERT_DIMS(int x, const SizeVector & dims) {

        ASSERT_EQ(datas[x].front()->getDims().size(), dims.size());
        for(size_t i = 0; i != dims.size(); i++) {
            ASSERT_EQ(datas[x].front()->getDims()[i], dims[i]);
        }
    }

    bool assertConnection(std::string a, std::string b) {

        bool bForward = false;
        for (auto && outData : wrap.getLayerByName(a.c_str())->outData) {
            auto &inputMap = outData->inputTo;
            auto i =
                std::find_if(inputMap.begin(), inputMap.end(), [&](std::map<std::string, CNNLayerPtr>::value_type &vt) {
                    return vt.second->name == b;
                });
            if (i != inputMap.end()) {
                bForward = true;
                break;
            }
        }
        if (!bForward) {
            return false;
        }

        auto prevData = wrap.getLayerByName(b.c_str())->insData;

        auto j = std::find_if(prevData.begin(), prevData.end(), [&](DataWeakPtr wp) {
            return wp.lock()->getCreatorLayer().lock()->name == a;
        });
        return  j != prevData.end();
    }

    int numCreated = 0;
    CNNLayerPtr createGenericLayer (std::string name) {
        auto newData = std::make_shared<Data>(name,
                                              SizeVector({1, 1}),
                                              Precision::FP32,
                                              Layout::NC);

        CNNLayerPtr newLayer = make_shared<GenericLayer>(LayerParams({name, "Generic_" + std::to_string(numCreated++), Precision::FP32}));
        newData->creatorLayer = newLayer;
        newLayer->outData.push_back(newData);

        return newLayer;
    }


    void prepareSomeInputs(InputsDataMap &inputsMap, std::initializer_list<int> inputLayers, int batchSize = 1) {
        for (auto layer = lhsLayers.begin(); layer != lhsLayers.end(); layer++) {
            if ((*layer)->insData.empty()) {
                auto isMarked = std::find_if(begin(inputLayers), end(inputLayers), [&](int value) {
                    return  std::to_string(value) ==(*layer)->name;
                });
                if (isMarked == end(inputLayers))
                    continue;
                auto info = make_shared<InputInfo>();
                auto data = make_shared<Data>((*layer)->name, Precision::FP32, Layout::NC);
                SizeVector dims = data->getDims();
                dims.push_back(batchSize);
                dims.push_back(batchSize);
                data->setDims(dims);
                for (auto output : (*layer)->outData) {
                    data->getInputTo() = output->inputTo;
                }
                info->setInputData(data);
                inputsMap[(*layer)->name] = info;
            }
        }
    }

    /**
     * @brief output layers considered only leafs here
     * @param outputMap
     */
    void prepareOutputs(OutputsDataMap & outputMap) {
        for (auto layer = rhsLayers.begin(); layer != rhsLayers.end(); layer++) {
            bool notLast = false;
            for (auto && outData : (*layer)->outData) {
                if (!outData->getInputTo().empty()) {
                    notLast = true;
                    break;
                }
            }
            if (notLast) continue;
            for (auto && outData : (*layer)->outData) {
                outputMap[outData->getName()] = outData;
            }
        }
    }
    /**
     * @brief creates 10 independent layers without connections.
     * Data corresponding to each layer sets up in outData
     * Likewise creator layer sets up for data in getCreatorLayer
     */
    int _batchSize = 1;
    void SetUp() override {
        datas.resize(10);
        for (int i = 0; i < 10; i++) {
            layers.push_back(make_shared<CNNLayer>(LayerParams({std::to_string(i)})));
            datas[i].push_back(make_shared<Data>(std::to_string(i), Precision::FP32, Layout::NC));
            datas[i].back()->getCreatorLayer() = layers[i];

            SizeVector dims = datas[i].back()->getDims();
            dims.push_back(_batchSize);
            dims.push_back(_batchSize);
            datas[i].back()->setDims(dims);

            layers.back()->outData.push_back(datas[i].back());
        }
    }

    int ID(const CNNLayerPtr &ptr) {
        for (int i = 0; i < layers.size(); i++) {
            if (layers[i].get() == ptr.get())
                return i;
        }
        return -1;
    }

    void ADD_ATTR(int layer, std::string attr, std::string value) {
        layers[layer]->params[attr] = value;
    }
    /**
     * @brief add edges between layers x and y, creating connected graph
     * @param x output layer index
     * @param y input layer index
     */
    void CONNECT(int x, int y) {
        datas[x].front()->getInputTo()[std::to_string(y)] = layers[y];
        layers[y]->insData.push_back(datas[x].front());
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }

    void CONNECT_FROM_PORT(int x, int port, int y) {
        if (datas[x].size() <= port) {
            datas[x].push_back(make_shared<Data>(std::string("split_") + std::to_string(datas[x].size()), Precision::FP32, Layout::NC));
            datas[x].back()->getCreatorLayer() = layers[x];

            SizeVector dims = datas[x].back()->getDims();
            dims.push_back(_batchSize);
            dims.push_back(_batchSize);
            datas[x].back()->setDims(dims);
            layers[x]->outData.push_back(datas[x].back());
        }
        datas[x][port]->getInputTo()[std::to_string(y)] = layers[y];
        layers[y]->insData.push_back(datas[x][port]);
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }

    void SET_DIMS(int x, const SizeVector & dims) {
        datas[x].front()->setDims(dims);
    }

    void SET_TYPE(int x, std::string name) {
        layers[x]->type = name;
    }
};

class MockCopier {
 public:
    CNNLayerPtr operator()(CNNLayerPtr layer) const {
        return copyLayer(layer);
    }
    MOCK_CONST_METHOD1(copyLayer, CNNLayerPtr(CNNLayerPtr));
};

}

MATCHER_P2(IsBetween, a, b, std::string(negation ? "isn't" : "is") + " between " + ::testing::PrintToString(a) + " and " + ::testing::PrintToString(b)) { return a <= arg && arg <= b; }

