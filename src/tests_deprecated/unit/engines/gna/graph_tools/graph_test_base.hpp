// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <legacy/graph_tools.hpp>
#include <gmock/gmock-generated-function-mockers.h>
#include <gmock/gmock-generated-matchers.h>
#include <gmock/gmock-more-actions.h>
#include "cpp/ie_cnn_network.h"
#include <legacy/details/ie_cnn_network_tools.h>

#include "unit_test_utils/mocks/mock_icnn_network.hpp"
#include "common_test_utils/common_utils.hpp"

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

    std::shared_ptr<MockICNNNetwork> mockNet;
    InferenceEngine::CNNNetwork wrap;

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
                        getInputTo(data) = getInputTo(output);
                    }
                    getCreatorLayer(data) = (*layer);
                    info->setInputData(data);
                    inputsMap[(*layer)->name] = info;
                }
            }
        };
        prepareInputsInternal(lhsLayers);
        prepareInputsInternal(rhsLayers);
    }

    CNNLayerPtr layerByName(std::string name) {
        auto sorted = InferenceEngine::details::CNNNetSortTopologically(
            InferenceEngine::CNNNetwork(mockNet));

        auto i = std::find_if(sorted.begin(), sorted.end(), [&](CNNLayerPtr l){
            return l->name == name;
        });
        if (i != sorted.end()) {
            return *i;
        }
        return nullptr;
    }


    #define ASSERT_N_CONNECTIONS(a, b, n) \
        ASSERT_EQ(countForwardConnections(#a, #b), n);\
        ASSERT_EQ(countBackwardConnections(#a, #b), n)

    #define ASSERT_MN_CONNECTIONS(a, b, m, n) \
        ASSERT_EQ(countForwardConnections(#a, #b), m);\
        ASSERT_EQ(countBackwardConnections(#a, #b), n)

    #define ASSERT_CONNECTION(a, b) \
        ASSERT_N_CONNECTIONS(a,b,1)

    #define ASSERT_PORT_CONNECTION(a, from_port, b, to_port) \
        ASSERT_EQ(countForwardConnections(#a, #b, from_port), 1);\
        ASSERT_EQ(countBackwardConnections(#a, #b, to_port), 1)

    #define ASSERT_2_CONNECTIONS(a, b) \
        ASSERT_N_CONNECTIONS(a,b,2)

    #define ASSERT_3_CONNECTIONS(a, b) \
        ASSERT_N_CONNECTIONS(a,b,3)

    /**
     * @brief check connection without direction
     */
    #define ASSERT_NO_CONNECTION(a, b) \
        ASSERT_EQ(countConnections(#a, #b), 0);\
        ASSERT_EQ(countConnections(#b, #a), 0)

    void ASSERT_DIMS(int x, const SizeVector & dims) {

        ASSERT_EQ(datas[x].front()->getDims().size(), dims.size());
        for(size_t i = 0; i != dims.size(); i++) {
            ASSERT_EQ(datas[x].front()->getDims()[i], dims[i]);
        }
    }

    int countForwardConnections(std::string a, std::string b, int from_port_id=-1) {
        long int nForward = 0;
        CNNLayerPtr layerExist;
        try {
            layerExist = CommonTestUtils::getLayerByName(wrap, a.c_str());
            if (!layerExist) {
                return 0;
            }
        } catch(...) {
            return 0;
        }

        for (auto && outData : layerExist->outData) {
            if (from_port_id != -1) {
                if (from_port_id > 0) {
                    from_port_id--;
                    continue;
                }
            }
            auto &inputMap = getInputTo(outData);
            nForward +=
                std::count_if(inputMap.begin(), inputMap.end(), [&](std::map<std::string, CNNLayerPtr>::value_type &vt) {
                    return vt.second->name == b;
                });
        }

        return nForward;
    }

    int countBackwardConnections(std::string a, std::string b, int from_port_id=-1) {
        CNNLayerPtr layerExist;
        try {
            layerExist = CommonTestUtils::getLayerByName(wrap, b.c_str());
            if (!layerExist) {
                return 0;
            }
        } catch(...) {
            return 0;
        }

        auto countRef = [&](DataWeakPtr wp) {
            return getCreatorLayer(wp.lock()).lock()->name == a;
        };

        if (from_port_id == -1) {
            auto prevData = layerExist->insData;

            return std::count_if(prevData.begin(), prevData.end(), countRef);
        } else {
            return countRef(layerExist->insData[from_port_id]);
        }
    }

    int countConnections(std::string a, std::string b) {
        return  countForwardConnections(a, b) + countBackwardConnections(a, b);
    }

    int numCreated = 0;
    CNNLayerPtr createGenericLayer (std::string name) {
        auto newData = std::make_shared<Data>(name, TensorDesc(Precision::FP32, SizeVector({ 1, 1 }), Layout::NC));

        CNNLayerPtr newLayer = make_shared<GenericLayer>(LayerParams({name, "Generic_" + std::to_string(numCreated++), Precision::FP32}));
        getCreatorLayer(newData) = newLayer;
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
                auto data = make_shared<Data>((*layer)->name, TensorDesc(Precision::FP32, Layout::NC));
                SizeVector dims = data->getDims();
                dims.push_back(batchSize);
                dims.push_back(batchSize);
                data->setDims(dims);
                for (auto output : (*layer)->outData) {
                    getInputTo(data) = getInputTo(output);
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
                if (!getInputTo(outData).empty()) {
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
       mockNet = std::make_shared<MockICNNNetwork>();
       wrap = InferenceEngine::CNNNetwork(std::dynamic_pointer_cast<ICNNNetwork>(mockNet));

        datas.resize(10);
        for (int i = 0; i < 10; i++) {
            layers.push_back(make_shared<CNNLayer>(LayerParams({std::to_string(i)}, "", Precision::UNSPECIFIED)));
            datas[i].push_back(make_shared<Data>(std::to_string(i), Precision::FP32, Layout::NC));
            getCreatorLayer(datas[i].back()) = layers[i];

            SizeVector dims = datas[i].back()->getDims();
            dims.push_back(_batchSize);
            dims.push_back(_batchSize);
            datas[i].back()->setDims(dims);

            layers.back()->outData.push_back(datas[i].back());
        }
    }

    void TearDown() override {
        // Reset shared_pointer circular dependencies to mitigate memory leaks.
        for (auto& items : datas) {
            for (auto& data : items) {
                for (auto& input : getInputTo(data)) {
                    input.second.reset();
                }
            }
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
        getInputTo(datas[x].front())[std::to_string(y)] = layers[y];
        layers[y]->insData.push_back(datas[x].front());
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }

    void CONNECT_FROM_PORT(int x, int port, int y) {
        if (datas[x].size() <= port) {
            datas[x].push_back(make_shared<Data>(std::string("split_") + std::to_string(datas[x].size()), Precision::FP32, Layout::NC));
            getCreatorLayer(datas[x].back()) = layers[x];

            SizeVector dims = datas[x].back()->getDims();
            dims.push_back(_batchSize);
            dims.push_back(_batchSize);
            datas[x].back()->setDims(dims);
            layers[x]->outData.push_back(datas[x].back());
        }
        getInputTo(datas[x][port])[std::to_string(y)] = layers[y];
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
