// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "cnn_network_impl.hpp"
#include <../graph_tools/graph_test_base.hpp>

using namespace testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace std;
using namespace GraphTest;

class CNNNetworkImplTest : public GraphTestsBase {
public:
    StatusCode sts = OK;
/**
 * @brief connect layers with wrong input data
 * @param x - output layer index
 * @param y - input layer index
 * @param wrongID - data index, which falsely displayed among inputs in y
 */
    void CONNECT_WRONGLY(int x, int y, int wrongID) {
        datas[x].front()->getInputTo()[std::to_string(y)] = layers[y];
        layers[y]->insData.push_back(datas[wrongID].front());
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }
/**
 * @brief connect layers where no input data in layer
 * @param x - output layer index
 * @param y - input layer index
 */
    void CONNECT_WITHOUT_INS_DATA(int x, int y) {
        datas[x].front()->getInputTo()[std::to_string(y)] = layers[y];
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }
/**
 * @brief connect layers with wrong data name
 * @param x - output layer index
 * @param y - input layer index
 * @param wrongName - wrong data name, which displayed between x and y
 */
    void CONNECT_WITH_DATA_NAME(int x, int y, int name) {
        datas[x].front()->name = std::to_string(name);
        datas[x].front()->getInputTo()[std::to_string(y)] = layers[y];
        layers[y]->insData.push_back(datas[x].front());
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }
/**
 * @brief connect layers with wrong layer name
 * @param x - output layer index
 * @param y - input layer index
 * @param name - wrong layer name, which displayed instead of y
 */
    void CONNECT_WITH_LAYER_NAME(int x, int y, int name) {
        layers[y]->name = std::to_string(name);
        datas[x].front()->getInputTo()[std::to_string(y)] = layers[y];
        layers[y]->insData.push_back(datas[x].front());
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }
/**
 * @brief insert data, which has no creator layer, but it is, into layer
 * @param x - data input to y index
 * @param y - input layer index
 */
    void CONNECT_WITHOUT_CREATOR_LAYER_WHICH_EXIST(int x, int y) {
        datas[x].front()->getInputTo()[std::to_string(y)] = layers[y];
        datas[x].front()->getCreatorLayer() = std::weak_ptr<CNNLayer>();
        layers[y]->insData.push_back(datas[x].front());
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }
/**
 * @brief insert data, which has no creator layer, into layer
 * @param x - data input to y index
 * @param y - input layer index
 */
    void CONNECT_WITHOUT_CREATOR_LAYER(int x, int y) {
        datas[x].front()->getInputTo()[std::to_string(y)] = layers[y];
        datas[x].front()->getCreatorLayer() = std::weak_ptr<CNNLayer>();
        layers[x] = nullptr;
        layers[y]->insData.push_back(datas[x].front());
        rhsLayers.insert(layers[y]);
    }
/**
 * @brief connect and specify input layer type
 * @param x - output  layer index
 * @param y - input layer index
 */
    void CONNECT_WITH_INPUT_TYPE(int x, int y, std::string type) {
        datas[x].front()->getInputTo()[std::to_string(y)] = layers[y];
        layers[y]->insData.push_back(datas[x].front());
        layers[x]->type = type;
        lhsLayers.insert(layers[x]);
        rhsLayers.insert(layers[y]);
    }
};

TEST_F(CNNNetworkImplTest, throwOnWrongInputType) {
    MockCNNNetworkImpl network;
    CONNECT_WITH_INPUT_TYPE(1, 2, "const");

    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}

TEST_F(CNNNetworkImplTest, severalRightInputTypes) {
    MockCNNNetworkImpl network;

    CONNECT(1, 2);
    CONNECT_WITH_INPUT_TYPE(3, 1, "input");
    CONNECT_WITH_INPUT_TYPE(0, 1, "input");

    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_NO_THROW(network.validateNetwork());
}

TEST_F(CNNNetworkImplTest, noCreatorLayers) {
    MockCNNNetworkImpl network;

    CONNECT(1, 2);
    CONNECT(3, 1);
    CONNECT_WITHOUT_CREATOR_LAYER(0, 1);
    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}

TEST_F(CNNNetworkImplTest, dataHasNoCreatorLayerButItIs) {
    MockCNNNetworkImpl network;

    CONNECT(1, 2);
    CONNECT(3, 1);
    CONNECT_WITHOUT_CREATOR_LAYER_WHICH_EXIST(0, 1);
    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}

TEST_F(CNNNetworkImplTest, layerNameIsNotUnique) {
    MockCNNNetworkImpl network;

    CONNECT(1, 2);
    CONNECT_WITH_LAYER_NAME(2, 3, 1);

    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}

TEST_F(CNNNetworkImplTest, dataNameIsNotUnique) {
    MockCNNNetworkImpl network;

    CONNECT(1, 2);
    CONNECT(1, 4);
    CONNECT_WITH_DATA_NAME(2, 3, 1);

    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}

TEST_F(CNNNetworkImplTest, layerDoesNotHaveInputData) {

    MockCNNNetworkImpl network;

    CONNECT(1, 2);
    CONNECT(9, 3);
    CONNECT(2, 9);
    CONNECT_WITHOUT_INS_DATA(1, 3);

    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}

TEST_F(CNNNetworkImplTest, layerDataNotCoresspondEachOtherOneInput) {

    MockCNNNetworkImpl network;

    CONNECT(1, 2);
    CONNECT(2, 3);
    CONNECT_WRONGLY(3, 4, 2);

    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}

TEST_F(CNNNetworkImplTest, layerDataNotCoresspondEachOtherTwoInputs) {

    MockCNNNetworkImpl network;

    CONNECT(1, 2);
    CONNECT(2, 3);
    CONNECT(7, 4);
    CONNECT_WRONGLY(3, 4, 2);

    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));
    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}

TEST_F(CNNNetworkImplTest, canGetName) {
    InferenceEngine::details::CNNNetworkImpl net;
    net.setName("myName");
    const char* p = "33333333333";
    char name[20];
    net.getName(name, sizeof(name));
    ASSERT_STREQ(name, "myName");
}

TEST_F(CNNNetworkImplTest, canGetNameStr) {
    InferenceEngine::details::CNNNetworkImpl net;
    net.setName("myName");
    auto name = net.getName();
    ASSERT_STREQ(name.c_str(), "myName");
}

TEST_F(CNNNetworkImplTest, cycleIsDetectedInNetwork) {

    MockCNNNetworkImpl network;

    // 1->2->3-> 4->5->6->7-> 8
    //       ^  |^        |  ^|
    //       |  |└--------┘  ||
    //       |  └------------┘|
    //       └----------------┘

    CONNECT(1, 2);
    CONNECT(2, 3);
    CONNECT(3, 4);
    CONNECT(4, 5);
    CONNECT(5, 6);
    CONNECT(6, 7);
    CONNECT(7, 8);
    CONNECT(7, 4);
    CONNECT(4, 8);
    CONNECT(8, 3);

    EXPECT_CALL(network, getInputsInfo(_)).Times(2).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));

    ASSERT_THROW(network.validateNetwork(), InferenceEngineException);
}