// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <gtest/gtest.h>
#include "layer_transform.hpp"

#include "unit_test_utils/mocks/mock_icnn_network.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace ::testing;

class LayerTransformTest : public ::testing::Test {
 protected:
    struct TransformMocker {
        MOCK_CONST_METHOD0(mockTransform, bool());
        template <class T>
        bool operator () (T ) const {
            return mockTransform();
        }
    };
    TransformMocker tmock;
    void SetUp() override  {
    }
};

TEST_F(LayerTransformTest, canInjectIntoConvolutionLayer) {

    ConvolutionLayer lc(LayerParams{"name", "type", Precision::FP32});

    lc._kernel.clear();
    lc._kernel.insert(X_AXIS, 23);
    lc._kernel.insert(Y_AXIS, 24);

    auto layerWithData = injectData<int>(lc);
    dynamic_cast<details::LayerInjector<ConvolutionLayer, int>*>(layerWithData.get())->injected = 5;

    ASSERT_EQ(dynamic_cast<ConvolutionLayer*>(layerWithData.get())->_kernel[X_AXIS],  23);
    ASSERT_EQ(dynamic_cast<ConvolutionLayer*>(layerWithData.get())->_kernel[Y_AXIS],  24);
}

TEST_F(LayerTransformTest, canInjectValue) {

    ConvolutionLayer lc(LayerParams{"name", "type", Precision::FP32});

    lc._kernel.clear();
    lc._kernel.insert(X_AXIS, 23);
    lc._kernel.insert(Y_AXIS, 24);

    auto layerWithData = injectData<int>(lc, 6);
    ASSERT_EQ((dynamic_cast<details::LayerInjector<ConvolutionLayer, int>*>(layerWithData.get())->injected), 6);

    ASSERT_EQ(dynamic_cast<ConvolutionLayer*>(layerWithData.get())->_kernel[X_AXIS],  23);
    ASSERT_EQ(dynamic_cast<ConvolutionLayer*>(layerWithData.get())->_kernel[Y_AXIS],  24);
}

TEST_F(LayerTransformTest, canAccessInjectedValue) {

    ConvolutionLayer lc(LayerParams{"name", "type", Precision::FP32});

    lc._kernel.clear();
    lc._kernel.insert(X_AXIS, 23);
    lc._kernel.insert(Y_AXIS, 24);

    auto layerWithData = injectData<int>(lc, 7);
    auto injectedData = getInjectedData<int>(layerWithData);

    ASSERT_NE(injectedData, nullptr);
    ASSERT_EQ(*injectedData, 7);

    ASSERT_EQ(dynamic_cast<ConvolutionLayer*>(layerWithData.get())->_kernel[X_AXIS],  23);
    ASSERT_EQ(dynamic_cast<ConvolutionLayer*>(layerWithData.get())->_kernel[Y_AXIS],  24);
}

TEST_F(LayerTransformTest, returnNullIfNotInjected) {

    ConvolutionLayer lc(LayerParams{"name", "type", Precision::FP32});

    lc._kernel.clear();
    lc._kernel.insert(X_AXIS, 23);
    lc._kernel.insert(Y_AXIS, 24);

    auto layerWithData = injectData<int>(lc, 7);

    ASSERT_EQ(getInjectedData<float>(layerWithData), nullptr);

    ASSERT_EQ(dynamic_cast<ConvolutionLayer*>(layerWithData.get())->_kernel[X_AXIS],  23);
    ASSERT_EQ(dynamic_cast<ConvolutionLayer*>(layerWithData.get())->_kernel[Y_AXIS],  24);
}

struct SomeData {
    int ivalue;
    std::string name;
    float value;
};

TEST_F(LayerTransformTest, canInjectStruct) {

    FullyConnectedLayer fc(LayerParams{"name", "type", Precision::FP32});
    fc._out_num = 9;

    auto layerWithData = injectData<SomeData>(fc, SomeData({11, "myname", 12.f}));

    auto some = getInjectedData<SomeData>(layerWithData);

    ASSERT_NE(some, nullptr);
    ASSERT_STREQ(some->name.c_str(), "myname");
    ASSERT_EQ(some->ivalue, 11);
    ASSERT_FLOAT_EQ(some->value, 12.f);
    ASSERT_EQ(dynamic_cast<FullyConnectedLayer*>(layerWithData.get())->_out_num,  9);

}
//  out data array items is fully copied, not just references to them
TEST_F(LayerTransformTest, injectioWillCopyOutData) {

    auto fc = std::make_shared<FullyConnectedLayer>(LayerParams{"name", "type", Precision::FP32});
    ASSERT_NE(fc, nullptr);
    fc->_out_num = 9;

    auto data  = std::make_shared<Data>("N1", Precision::FP32);
    getCreatorLayer(data) = fc;
    fc->outData.push_back(data);

    auto layerWithData = injectData<SomeData>(fc, SomeData({11, "myname", 12.f}));

    ASSERT_EQ(getCreatorLayer(data).lock(), getCreatorLayer(layerWithData->outData[0]).lock());
    ASSERT_NE(data.get(), layerWithData->outData[0].get());
}

TEST_F(LayerTransformTest, injectioWillCopyInputData) {

    auto fc = std::make_shared<FullyConnectedLayer>(LayerParams{"name", "type", Precision::FP32});
    ASSERT_NE(fc, nullptr);
    fc->_out_num = 9;

    auto data  = std::make_shared<Data>("N1", Precision::FP32);
    getCreatorLayer(data) = fc;
    fc->insData.push_back(data);

    auto layerWithData = injectData<SomeData>(fc, SomeData({11, "myname", 12.f}));

    ASSERT_EQ(data.get(), layerWithData->insData[0].lock().get());
}

TEST_F(LayerTransformTest, transformWillOnlyTransformOnce) {

    auto fc = std::make_shared<FullyConnectedLayer>(LayerParams{"name", "type", Precision::FP32});
    ASSERT_NE(fc, nullptr);
    fc->_out_num = 9;

    EXPECT_CALL(tmock, mockTransform()).WillOnce(Return(true));

    // CNNLayer might be selected in case of overloads
    transformLayer(fc, tmock);
}

TEST_F(LayerTransformTest, transformCanGoToParentIfChildTransformNotImplemented) {

    auto fc = std::make_shared<FullyConnectedLayer>(LayerParams{"name", "type", Precision::FP32});
    ASSERT_NE(fc, nullptr);
    fc->_out_num = 9;

    Sequence s1;
    EXPECT_CALL(tmock, mockTransform()).InSequence(s1).WillOnce(Return(false));
    EXPECT_CALL(tmock, mockTransform()).InSequence(s1).WillOnce(Return(false));
    EXPECT_CALL(tmock, mockTransform()).InSequence(s1).WillOnce(Return(true));

    // CNNLayer might be selected in case of overloads
    transformLayer(fc, tmock);
}