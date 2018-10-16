// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/graph_tools.hpp>
#include "graph_test_base.hpp"
#include <unordered_set>
#include "mock_icnn_network.hpp"
#include <gmock/gmock-generated-function-mockers.h>
#include <gmock/gmock-generated-matchers.h>
#include <gmock/gmock-more-actions.h>
#include "xml_father.hpp"
#include "ie_common.h"
#include <memory>

using namespace testing;
using namespace InferenceEngine;
using namespace std;
using namespace GraphTest;

class GraphToolsTest : public GraphTestsBase {

};

TEST_F(GraphToolsTest, canRunSimpleDFS) {

    CONNECT(0, 1);
    CONNECT(0, 2);
    CONNECT(1, 3);
    CONNECT(2, 3);

    EXPECT_CALL(*this, visited(0 ,0)).Times(1);
    EXPECT_CALL(*this, visited(1, IsBetween(1,3))).Times(1);
    EXPECT_CALL(*this, visited(2, IsBetween(1,3))).Times(1);
    EXPECT_CALL(*this, visited(3, 2)).Times(1);

    int idx = 0;
    CNNNetDFS(layers[0], [&] (const CNNLayerPtr & layer) {
        visited(ID(layer), idx++);
    });
}


TEST_F(GraphToolsTest, canRunCycleDFS) {

    CONNECT(0, 1);
    CONNECT(1, 2);
    CONNECT(2, 0);

    EXPECT_CALL(*this, visited(0 ,0)).Times(1);
    EXPECT_CALL(*this, visited(1, 1)).Times(1);
    EXPECT_CALL(*this, visited(2, 2)).Times(1);

    int idx = 0;
    CNNNetDFS(layers[0], [&] (const CNNLayerPtr & layer) {
        visited(ID(layer), idx++);
    });
}


TEST_F(GraphToolsTest, canRunBFS) {

    CONNECT(0, 1);
    CONNECT(0, 2);
    CONNECT(0, 3);
    CONNECT(1, 4);

    EXPECT_CALL(*this, visited(0 ,0)).Times(1);
    EXPECT_CALL(*this, visited(1, IsBetween(1,3))).Times(1);
    EXPECT_CALL(*this, visited(2, IsBetween(1,3))).Times(1);
    EXPECT_CALL(*this, visited(3, IsBetween(1,3))).Times(1);
    EXPECT_CALL(*this, visited(4, 4)).Times(1);

    int idx = 0;
    CNNNetBFS(layers[0], [&] (const InferenceEngine::CNNLayerPtr & layer) {
        visited(ID(layer), idx++);
    });
}


TEST_F(GraphToolsTest, canRunNBFS) {

    CONNECT(0, 1);
    CONNECT(0, 2);
    CONNECT(0, 3);
    CONNECT(1, 4);

    EXPECT_CALL(*this, visited(0 ,0)).Times(1);
    EXPECT_CALL(*this, visited(1, IsBetween(1,3))).Times(1);
    EXPECT_CALL(*this, visited(2, IsBetween(1,3))).Times(1);
    EXPECT_CALL(*this, visited(3, IsBetween(1,3))).Times(1);

    int idx = 0;
    CNNNetNBFS(layers[0], 1, [&] (const InferenceEngine::CNNLayerPtr & layer) {
        visited(ID(layer), idx++);
    });
}

TEST_F(GraphToolsTest, canSortTopologically) {

    CONNECT(0, 1);
    CONNECT(2, 1);
    CONNECT(1, 4);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillOnce(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));
    auto sorted = CNNNetSortTopologically(mockNet);

    EXPECT_EQ(sorted.size(), 4);

    //  first element can be 0 or 2 depending on implementation
    EXPECT_TRUE(
        sorted[0]->name=="0" && sorted[1]->name=="2" ||
        sorted[0]->name=="2" && sorted[1]->name=="0");

    EXPECT_STREQ(sorted[2]->name.c_str(), "1");
    EXPECT_STREQ(sorted[3]->name.c_str(), "4");
}

TEST_F(GraphToolsTest, canDetectLoopsWhileSortTing) {

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

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillOnce(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));
    ASSERT_ANY_THROW(CNNNetSortTopologically(mockNet));
}


TEST_F(GraphToolsTest, canSortIfInputsPointsToLayerWithMultiInputs) {

    CONNECT(1, 2);
    CONNECT(3, 4);
    CONNECT(4, 2);
    CONNECT(3, 5);
    CONNECT(5, 2);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillOnce(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    auto sorted = CNNNetSortTopologically(mockNet);

    vector<vector<string>> expected = {
        {"1", "3", "4", "5", "2"},
        {"3", "4", "5", "1", "2"},
        {"3", "5", "4", "1", "2"},
        {"1", "3", "5", "4", "2"},
    };

    bool bFailed = true;
    for (auto ex: expected) {
        bFailed = false;
        for (auto i = 0; i < ex.size(); i++) {
            if (sorted[i]->name != ex[i]) {
                bFailed = true;
                break;
            }
        }
        if (!bFailed) break;
    }
    std::stringstream actual;
    for (auto x : sorted) {
        actual << x->name << " ";
    }

    EXPECT_FALSE(bFailed) << actual.str() << "doesn't match: one of expected" ;
}

TEST_F(GraphToolsTest, canGetAllMemoryInputsLayersFromStandardInputs) {


    // 1->4--┐
    //       |
    // 2->5->6
    //    |
    // 3->7

    CONNECT(1, 4);
    CONNECT(2, 5);
    CONNECT(3, 7);
    CONNECT(4, 5);
    CONNECT(5, 6);
    CONNECT(5, 7);


    EXPECT_CALL(mockNet, getInputsInfo(_)).WillOnce(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareSomeInputs(maps, {1});
    })));
    auto allInputLayers = CNNNetGetAllInputLayers(mockNet);
    ASSERT_EQ(3, allInputLayers.size());
    auto element = allInputLayers.begin();
    ASSERT_STREQ("1", element->get()->name.c_str());
    element++;
    ASSERT_STREQ("2", element->get()->name.c_str());
    element++;
    ASSERT_STREQ("3", element->get()->name.c_str());
}

TEST_F(GraphToolsTest, canGetSingleInputLayer) {
    // 1->2
    CONNECT(1, 2);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillOnce(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareSomeInputs(maps, {1});
    })));
    auto allInputLayers = CNNNetGetAllInputLayers(mockNet);
    ASSERT_EQ(1, allInputLayers.size());
}

TEST_F(GraphToolsTest, canIterateOverCNNNetwork) {
    CONNECT(1, 2);
    CONNECT(1, 3);
    CONNECT(2, 6);
    CONNECT(3, 6);
    CONNECT(3, 4);
    CONNECT(3, 5);
    CONNECT(4, 5);
    CONNECT(4, 7);
    CONNECT(6, 7);
    CONNECT(7, 8);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    std::vector<CNNLayerPtr>resultedOrder;
    for (auto l : wrap) {
        resultedOrder.push_back(l);
    }

    ASSERT_EQ(wrap.size(), 8);
    ASSERT_STREQ(resultedOrder[0]->name.c_str(), "2");
    ASSERT_STREQ(resultedOrder[1]->name.c_str(), "6");
    ASSERT_STREQ(resultedOrder[2]->name.c_str(), "1");
    ASSERT_STREQ(resultedOrder[3]->name.c_str(), "7");
    ASSERT_STREQ(resultedOrder[4]->name.c_str(), "3");
    ASSERT_STREQ(resultedOrder[5]->name.c_str(), "8");
    ASSERT_STREQ(resultedOrder[6]->name.c_str(), "4");
    ASSERT_STREQ(resultedOrder[7]->name.c_str(), "5");
}

TEST_F(GraphToolsTest, canIterateOverCNNNetworkWithCycle) {
    CONNECT(1, 2);
    CONNECT(2, 3);
    CONNECT(3, 4);
    CONNECT(4, 2);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    std::vector<CNNLayerPtr>resultedOrder;
    for (auto l : wrap) {
        resultedOrder.push_back(l);
    }

    ASSERT_EQ(wrap.size(), 4);
    ASSERT_STREQ(resultedOrder[0]->name.c_str(), "2");
    ASSERT_STREQ(resultedOrder[1]->name.c_str(), "3");
    ASSERT_STREQ(resultedOrder[2]->name.c_str(), "1");
    ASSERT_STREQ(resultedOrder[3]->name.c_str(), "4");
}

TEST_F(GraphToolsTest, canCompareCNNNetworkIterators) {
    CONNECT(1, 2);
    CONNECT(1, 3);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillOnce(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    auto i = std::begin(wrap);
    auto i2 = i;
    i2++;

    ASSERT_NE(i, i2);
    i++;
    ASSERT_EQ(i, i2);
}

TEST_F(GraphToolsTest, canIterateOverEmptyNetwork) {
    CONNECT(1, 2);
    CONNECT(2, 1);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillOnce(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    ASSERT_EQ(std::begin(wrap), std::end(wrap));
}


TEST_F(GraphToolsTest, CNNNetworkInsertLayerThrowsForNullPointers) {
    CNNLayerPtr nullLayer;
    ASSERT_ANY_THROW(CNNNetworkInsertLayer(nullLayer, nullLayer, nullLayer));
}

TEST_F(GraphToolsTest, CanNotInsertLayerIntoNonAdjiacendLayers) {
    CONNECT(1, 2);
    CONNECT(2, 3);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0,1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
        l = layerByName(name);
        return l== nullptr ? GENERAL_ERROR : OK;
    })));

    auto l = wrap.getLayerByName("1");
    auto r = wrap.getLayerByName("3");

    ASSERT_ANY_THROW(CNNNetworkInsertLayer(l, r, createGenericLayer("3")));
}

TEST_F(GraphToolsTest, CNNNetworkInsertLayerSimpleCase) {
    CONNECT(1, 2);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0, 1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
        l = layerByName(name);
        return l== nullptr ? GENERAL_ERROR : OK;
    })));

    auto l = wrap.getLayerByName("1");
    auto r = wrap.getLayerByName("2");

    CNNNetworkInsertLayer(l, r, createGenericLayer("3"));

    ASSERT_CONNECTION(3, 2);
    ASSERT_CONNECTION(1, 3);
}

TEST_F(GraphToolsTest, CNNNetworkInsertLayerSimpleCaseWithMultipleOutputs) {
    CONNECT(1, 2);
    CONNECT(1, 3);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0,1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
        l = layerByName(name);
        return l== nullptr ? GENERAL_ERROR : OK;
    })));

    auto l = wrap.getLayerByName("1");
    auto r = wrap.getLayerByName("3");

    CNNNetworkInsertLayer(l, r, createGenericLayer("4"));

    ASSERT_CONNECTION(4, 3);
    ASSERT_CONNECTION(1, 4);
    ASSERT_CONNECTION(1, 2);
}


TEST_F(GraphToolsTest, CNNNetworkInsertLayerSimpleCaseWithMultipleInputs) {
    CONNECT(1, 2);
    CONNECT(3, 2);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0,1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
        l = layerByName(name);
        return l== nullptr ? GENERAL_ERROR : OK;
    })));

    auto l = wrap.getLayerByName("3");
    auto r = wrap.getLayerByName("2");

    CNNNetworkInsertLayer(l, r, createGenericLayer("4"));

    ASSERT_CONNECTION(4, 2);
    ASSERT_CONNECTION(3, 4);
    ASSERT_CONNECTION(1, 2);
}

TEST_F(GraphToolsTest, CNNNetworkInsertAfterLastLayer) {
    CONNECT(1, 2);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0, 1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
        l = layerByName(name);
        return l== nullptr ? GENERAL_ERROR : OK;
    })));

    auto l = wrap.getLayerByName("2");

    CNNNetworkInsertLayer(l, nullptr, createGenericLayer("3"));

    ASSERT_CONNECTION(1, 2);
    ASSERT_CONNECTION(2, 3);
}

TEST_F(GraphToolsTest, CNNNetworkInsertAfterAll) {
    CONNECT(1, 2);
    CONNECT(1, 3);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0, 1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
        l = layerByName(name);
        return l== nullptr ? GENERAL_ERROR : OK;
    })));

    CNNNetworkInsertLayer(wrap.getLayerByName("1"), nullptr, createGenericLayer("5"));

    ASSERT_CONNECTION(1, 5);
    ASSERT_CONNECTION(5, 2);
    ASSERT_CONNECTION(5, 3);
}

TEST_F(GraphToolsTest, CNNNetworkInsertAllAfterSplit) {

    CONNECT_FROM_PORT(1, 0, 2);
    CONNECT_FROM_PORT(1, 1, 3);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0, 1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
        l = layerByName(name);
        return l== nullptr ? GENERAL_ERROR : OK;
    })));

    CNNNetworkInsertLayer(wrap.getLayerByName("1"), nullptr, createGenericLayer("5"));

    ASSERT_CONNECTION(1, 5);
    ASSERT_CONNECTION(5, 2);
    ASSERT_CONNECTION(5, 3);
}

TEST_F(GraphToolsTest, CNNNetworkInsert1AfterSplit) {

    CONNECT_FROM_PORT(1, 0, 2);
    CONNECT_FROM_PORT(1, 1, 3);
    CONNECT_FROM_PORT(1, 2, 4);

    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
        prepareInputs(maps);
    })));

    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0, 1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
        l = layerByName(name);
        return l== nullptr ? GENERAL_ERROR : OK;
    })));

    CNNNetworkInsertLayer(wrap.getLayerByName("1"), wrap.getLayerByName("4"), createGenericLayer("5"));

    ASSERT_CONNECTION(1, 2);
    ASSERT_CONNECTION(1, 3);
    ASSERT_CONNECTION(1, 5);
    ASSERT_CONNECTION(5, 4);
}


//TEST_F(GraphToolsTest, CNNNetworkInsertLayerBeforeAll) {
//    CONNECT(1, 2);
//
//    EXPECT_CALL(mockNet, GetInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap & maps){
//        prepareInputs(maps);
//    })));
//
//    EXPECT_CALL(mockNet, getLayerByName(_,_,_)).WillRepeatedly(WithArgs<0, 1>(Invoke([&](const char* name, InferenceEngine::CNNLayerPtr& l){
//        l = layerByName(name);
//        return l== nullptr ? GENERAL_ERROR : OK;
//    })));
//
//    CNNNetworkInsertLayer(wrap.getLayerByName("1"), nullptr, createGenericLayer("3"));
//
//    ASSERT_STREQ("2", wrap.getLayerByName("3")->outData[0]->inputTo.begin()->second->name.c_str());
//    ASSERT_STREQ("1", CNNNetPrevLayerName(wrap.getLayerByName("3")).c_str());
//    ASSERT_STREQ("3", wrap.getLayerByName("1")->outData[0]->inputTo.begin()->second->name.c_str());
//    ASSERT_STREQ("3", CNNNetPrevLayerName(wrap.getLayerByName("2")).c_str());
//}
