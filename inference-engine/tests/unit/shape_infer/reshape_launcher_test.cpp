// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include <shape_infer/ie_reshape_launcher.hpp>
#include <blob_factory.hpp>
#include <shape_infer/mock_ishape_infer_impl.hpp>
#include <shape_infer/mock_reshaper_launcher.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;
using namespace ::testing;

class ReshapeLauncherTest : public ::testing::Test {
protected:
    void SetUp() override {
        notEmptyData = getNotEmptyData();
        impl = std::make_shared<MockIShapeInferImpl>();
    };
    std::vector<Blob::CPtr> getBlobs(const std::vector<SizeVector>& shapes) {
        std::vector<Blob::CPtr> inBlobs;
        for (auto const& dims : shapes) {
            TensorDesc desc(Precision::FP32, dims, TensorDesc::getLayoutByDims(dims));
            auto blob = make_blob_with_precision(desc);
            inBlobs.push_back(blob);
        }
        return inBlobs;
    }
public:
    StatusCode sts = GENERAL_ERROR;
    ResponseDesc resp;
    static const std::string TEST_NAME;
    DataPtr notEmptyData;
    MockIShapeInferImpl::Ptr impl;
    SizeVector inDims{1};
    SizeVector outDims{2};
    std::map<std::string, std::string> changedParams{{TEST_NAME, TEST_NAME}};
public:
    DataPtr getNotEmptyData() {
        return std::make_shared<Data>(TEST_NAME, Precision::FP32, Layout::C);
    }
};

const std::string ReshapeLauncherTest::TEST_NAME = "TEST_NAME";

TEST_F(ReshapeLauncherTest, failedToCreateWithNullLayer) {
    const CNNLayer* layer = nullptr;
    ASSERT_THROW(ReshapeLauncher launcher(layer, impl), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, failedToCreateWithNullInsData) {
    CNNLayer layer({});
    layer.outData = {notEmptyData};
    ASSERT_THROW(ReshapeLauncher launcher(&layer, impl), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, failedToCreateWithExpiredInsData) {
    CNNLayer layer({});
    layer.outData = {notEmptyData};
    DataWeakPtr expired = std::make_shared<Data>(TEST_NAME, Precision::UNSPECIFIED);
    layer.insData = {expired};
    ASSERT_THROW(ReshapeLauncher launcher(&layer, impl), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, failedToCreateWithEmptyOutData) {
    CNNLayer layer({});
    layer.insData = {notEmptyData};
    ASSERT_THROW(ReshapeLauncher launcher(&layer, impl), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, failedToCreateWithNullOutData) {
    CNNLayer layer({});
    layer.insData = {notEmptyData};
    layer.outData = {nullptr};
    ASSERT_THROW(ReshapeLauncher launcher(&layer, impl), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, failedToCreateWithEmptyImpl) {
    CNNLayer layer({});
    layer.outData = {notEmptyData};
    layer.insData = {notEmptyData};
    impl = nullptr;
    ASSERT_THROW(ReshapeLauncher launcher(&layer, impl), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, canCreateReshapeLauncher) {
    CNNLayer layer({});
    layer.outData = {notEmptyData};
    layer.insData = {notEmptyData};
    ReshapeLauncher launcher(&layer, impl);
}

TEST_F(ReshapeLauncherTest, throwOnReshapeWihtNotEnoughShapes) {
    CNNLayer layer({});
    layer.outData = {notEmptyData};
    layer.insData = {notEmptyData, notEmptyData};
    ReshapeLauncher launcher(&layer, impl);

    launcher.setShapeByName(inDims, TEST_NAME);
    try {
        launcher.reshape({});
        FAIL() << "Reshape should be failed!";
    } catch (...) {}
}

TEST_F(ReshapeLauncherTest, implIsCalledOnReshape) {
    CNNLayer layer({});
    layer.insData = {notEmptyData};
    auto initializer = std::make_shared<MockReshapeLauncher::TestLauncherInitializer>();
    ReshapeLauncher launcher(&layer, impl, initializer);
    auto inputController = initializer->getInputController();
    auto outputController = initializer->getOutputController();
    std::vector<SizeVector> shapes{inDims};
    auto blobs = getBlobs(shapes);
    EXPECT_CALL(*inputController, setShapeByName(inDims, TEST_NAME));
    EXPECT_CALL(*inputController, getBlobs(true)).WillOnce(Return(blobs));
    EXPECT_CALL(*outputController, setShapes(_));
    EXPECT_CALL(*outputController, propagateShapes(_));
    EXPECT_CALL(*impl.get(), inferShapes(blobs, _, _, _, _)).WillOnce(Return(OK));
    launcher.setShapeByName(inDims, TEST_NAME);
    launcher.reshape({});
}

TEST_F(ReshapeLauncherTest, canApplyChanges) {
    CNNLayer layer({});
    layer.outData = {getNotEmptyData()};
    layer.insData = {notEmptyData};
    ReshapeLauncher launcher(&layer, impl);
    launcher.setShapeByName(inDims, TEST_NAME);

    EXPECT_CALL(*impl.get(), inferShapes(_, _, _, _, _)).
            WillOnce(DoAll(
            WithArg<3>(Invoke([&](std::vector<SizeVector>& outShape) { outShape.push_back(outDims); })), Return(OK)));
    launcher.reshape({});
    launcher.applyChanges(&layer);

    auto insData = layer.insData;
    auto outData = layer.outData;
    ASSERT_EQ(1, insData.size());
    ASSERT_EQ(1, outData.size());
    auto ins0Data = insData[0].lock();
    auto out0Data = outData[0];
    ASSERT_NE(nullptr, ins0Data);
    ASSERT_NE(nullptr, out0Data);
    ASSERT_EQ(inDims, ins0Data->getDims());
    ASSERT_EQ(outDims, out0Data->getDims());
}

TEST_F(ReshapeLauncherTest, throwOnApplyingWithNotEnoughOutput) {
    CNNLayer layer({});
    layer.outData = {notEmptyData};
    layer.insData = {notEmptyData};
    ReshapeLauncher launcher(&layer, impl);
    launcher.setShapeByName(inDims, TEST_NAME);
    EXPECT_CALL(*impl.get(), inferShapes(_, _, _, _, _)).
            WillOnce(DoAll(
            WithArg<3>(Invoke([&](std::vector<SizeVector>& outShape) {
                outShape.push_back(outDims);
                outShape.push_back(outDims);
            })),
            Return(OK)));
    ASSERT_THROW(launcher.reshape({}), InferenceEngineException);
    ASSERT_THROW(launcher.applyChanges(&layer), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, throwOnApplyingWithNotEnoughShapes) {
    CNNLayer layer({});
    layer.outData = {notEmptyData, notEmptyData};
    layer.insData = {notEmptyData};
    ReshapeLauncher launcher(&layer, impl);
    launcher.setShapeByName(inDims, TEST_NAME);
    EXPECT_CALL(*impl.get(), inferShapes(_, _, _, _, _)).
            WillOnce(DoAll(
            WithArg<3>(Invoke([&](std::vector<SizeVector>& outShape) { outShape.push_back(outDims); })),
            Return(OK)));
    ASSERT_THROW(launcher.reshape({}), InferenceEngineException);
    ASSERT_THROW(launcher.applyChanges(&layer), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, canNotApplyForLayerWithAnotherName) {
    CNNLayer layer1({});
    layer1.outData = {notEmptyData};
    layer1.insData = {notEmptyData};
    CNNLayer layer2({});
    layer2.name = TEST_NAME;
    ReshapeLauncher launcher(&layer1, impl);
    {  // to not fail because of empty input and output shapes
        launcher.setShapeByName(inDims, TEST_NAME);
        EXPECT_CALL(*impl.get(), inferShapes(_, _, _, _, _)).
                WillOnce(DoAll(
                WithArg<3>(Invoke([&](std::vector<SizeVector>& outShape) { outShape.push_back(outDims); })),
                Return(OK)));
        launcher.reshape({});
    }
    ASSERT_THROW(launcher.applyChanges(&layer2), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, DISABLED_canNotApplyForLayerWithAnotherParams) {
    CNNLayer layer1({});
    layer1.outData = {notEmptyData};
    layer1.insData = {notEmptyData};
    CNNLayer layer2({});
    layer2.params = changedParams;
    ReshapeLauncher launcher(&layer1, impl);
    {  // to not fail because of empty input and output shapes
        launcher.setShapeByName(inDims, TEST_NAME);
        EXPECT_CALL(*impl.get(), inferShapes(_, _, _, _, _)).
                WillOnce(DoAll(
                WithArg<3>(Invoke([&](std::vector<SizeVector>& outShape) { outShape.push_back(outDims); })),
                Return(OK)));
        launcher.reshape({});
    }
    ASSERT_THROW(launcher.applyChanges(&layer2), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, canNotApplyForLayerWithEmptyInShapes) {
    CNNLayer layer1({});
    layer1.outData = {notEmptyData};
    layer1.insData = {notEmptyData};
    CNNLayer layer2({});
    layer2.params = changedParams;
    ReshapeLauncher launcher(&layer1, impl);
    {  // to not fail because of inconsistent number of input/outputs
        layer1.insData.clear();
        layer1.outData.clear();
    }
    ASSERT_THROW(launcher.applyChanges(&layer2), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, canNotApplyForLayerWithEmptyOutShapes) {
    CNNLayer layer1({});
    layer1.outData = {notEmptyData};
    layer1.insData = {notEmptyData};
    CNNLayer layer2({});
    layer2.params = changedParams;
    ReshapeLauncher launcher(&layer1, impl);
    {  // to not fail because of inconsistent number of input/outputs
        launcher.setShapeByName(inDims, TEST_NAME);
        layer1.outData.clear();
    }
    ASSERT_THROW(launcher.applyChanges(&layer2), InferenceEngineException);
}

TEST_F(ReshapeLauncherTest, canReset) {
    auto initializer = std::make_shared<MockReshapeLauncher::TestLauncherInitializer>();
    MockReshapeLauncher launcher(initializer);
    auto inputController = initializer->getInputController();
    auto outputController = initializer->getOutputController();
    EXPECT_CALL(*inputController, reset()).Times(1);
    EXPECT_CALL(*outputController, reset()).Times(1);
    launcher.realReset();
}
