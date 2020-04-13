// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include "unit_test_utils/mocks/shape_infer/mock_ishape_infer_impl.hpp"
#include "unit_test_utils/mocks/shape_infer/mock_reshaper_launcher.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;
using namespace ::testing;

class InputReshapeLauncherTest : public ::testing::Test {
protected:
    void SetUp() override {
        notEmptyData = getNotEmptyData();
        impl = std::make_shared<MockIShapeInferImpl>();
    };

public:
    static const std::string TEST_NAME;
    DataPtr notEmptyData;
    MockIShapeInferImpl::Ptr impl;
    SizeVector outDims{2};
public:
    DataPtr getNotEmptyData() {
        return std::make_shared<Data>(TEST_NAME, Precision::UNSPECIFIED, Layout::C);
    }

    CNNLayerPtr createLayer(const std::string& name = TEST_NAME, const std::string& type = "Input") {
        LayerParams params{name, type, Precision::UNSPECIFIED};
        auto layer = std::make_shared<CNNLayer>(params);
        if (layer == nullptr) {
            THROW_IE_EXCEPTION << "InputReshapeLauncherTest::createLayer(). Could not create CNNLayer";
        }
        layer->outData = {notEmptyData};
        notEmptyData->setDims(outDims);
        return layer;
    }
};

const std::string InputReshapeLauncherTest::TEST_NAME = "TEST_NAME";

TEST_F(InputReshapeLauncherTest, failedToCreateWithNullLayer) {
    const CNNLayer* layer = nullptr;
    ASSERT_THROW(InputReshapeLauncher launcher(layer, impl), InferenceEngineException);
}

TEST_F(InputReshapeLauncherTest, failedToCreateWithEmptyOutData) {
    CNNLayer layer({});
    ASSERT_THROW(InputReshapeLauncher launcher(&layer, impl), InferenceEngineException);
}

TEST_F(InputReshapeLauncherTest, failedToCreateWithNullOutData) {
    CNNLayer layer({});
    layer.outData = {nullptr};
    ASSERT_THROW(InputReshapeLauncher launcher(&layer, impl), InferenceEngineException);
}

TEST_F(InputReshapeLauncherTest, failedToCreateWithNotInputType) {
    CNNLayer layer({});
    layer.outData = {notEmptyData};
    ASSERT_THROW(InputReshapeLauncher launcher(&layer, impl), InferenceEngineException);
}

TEST_F(InputReshapeLauncherTest, canCreateReshapeLauncher) {
    ASSERT_NO_THROW(InputReshapeLauncher launcher(createLayer().get(), impl));
}

TEST_F(InputReshapeLauncherTest, canPushShapes) {
    InputReshapeLauncher launcher(createLayer().get(), impl);
    ASSERT_NO_THROW(launcher.setShapeByName(outDims, TEST_NAME));
}

TEST_F(InputReshapeLauncherTest, canPropagateWithNotEnoughShapes) {
    InputReshapeLauncher launcher(createLayer().get(), impl);
    launcher.reshape({});
}

TEST_F(InputReshapeLauncherTest, throwOnPropagateWithEmptyLaunchers) {
    auto layer = createLayer();
    layer->outData[0]->inputTo = {{{}, createLayer(TEST_NAME, TEST_NAME)}};
    InputReshapeLauncher launcher(layer.get(), impl);
    launcher.setShapeByName(outDims, TEST_NAME);
    ASSERT_NO_THROW();
    ASSERT_THROW(launcher.reshape({}), InferenceEngineException);
}

TEST_F(InputReshapeLauncherTest, throwOnPropagateWithoutProperLauncher) {
    auto layer = createLayer();
    layer->outData[0]->inputTo = {{{}, createLayer(TEST_NAME + "another", TEST_NAME)}};
    InputReshapeLauncher inLauncher(layer.get(), impl);
    inLauncher.setShapeByName(outDims, TEST_NAME);
    auto launcher = std::make_shared<MockReshapeLauncher>();
    EXPECT_CALL(*launcher.get(), getLayerName()).WillOnce(Return(TEST_NAME));
    ASSERT_THROW(inLauncher.reshape({{launcher}}), InferenceEngineException);
}

TEST_F(InputReshapeLauncherTest, canPropagate) {
    auto layer = createLayer();
    layer->outData[0]->inputTo = {{{}, createLayer(TEST_NAME, TEST_NAME)}};
    InputReshapeLauncher inLauncher(layer.get(), impl);
    auto launcher = std::make_shared<MockReshapeLauncher>();
    EXPECT_CALL(*launcher.get(), setShapeByName(outDims, TEST_NAME));
    EXPECT_CALL(*launcher.get(), getLayerName()).WillOnce(Return(TEST_NAME));
    inLauncher.setShapeByName(outDims, TEST_NAME);
    inLauncher.reshape({{launcher}});
}

TEST_F(InputReshapeLauncherTest, canReset) {
    auto layer = createLayer();
    InputReshapeLauncher launcher(layer.get(), impl);
    ASSERT_NO_THROW(launcher.reset());
}

TEST_F(InputReshapeLauncherTest, canApplyWithoutSettingShapes) {
    auto layer = createLayer();
    layer->outData.push_back(notEmptyData);
    InputReshapeLauncher launcher(layer.get(), impl);
    ASSERT_NO_THROW(launcher.applyChanges(layer.get()));
}

TEST_F(InputReshapeLauncherTest, canNotApplyForLayerWithAnotherName) {
    auto layer1 = createLayer("");
    auto layer2 = createLayer();
    InputReshapeLauncher launcher(layer1.get(), impl);
    launcher.setShapeByName(outDims, TEST_NAME);
    ASSERT_THROW(launcher.applyChanges(layer2.get()), InferenceEngineException);
}

TEST_F(InputReshapeLauncherTest, canApplyChanges) {
    auto layer = createLayer();
    InputReshapeLauncher launcher(layer.get(), impl);
    launcher.setShapeByName(outDims, TEST_NAME);
    launcher.applyChanges(layer.get());

    auto outData = layer->outData;
    ASSERT_EQ(1, outData.size());
    auto out0Data = outData[0];
    ASSERT_NE(nullptr, out0Data);
    ASSERT_EQ(outDims, out0Data->getDims());
}

TEST_F(InputReshapeLauncherTest, canGetShapesFromLayer) {
    CNNLayer layer({});
    layer.outData = {notEmptyData};
    notEmptyData->setDims(outDims);
    auto initializer = std::make_shared<MockReshapeLauncher::TestLauncherInitializer>();
    InputReshapeLauncher launcher(&layer, impl, initializer);
    auto outputController = initializer->getOutputController();
    EXPECT_CALL(*outputController, getIRShapes()).WillOnce(Return(std::vector<SizeVector>{outDims}));
    EXPECT_CALL(*outputController, getShapes(false)).WillOnce(Return(std::vector<SizeVector>{SizeVector()}));
    EXPECT_CALL(*outputController, setShapeByIndex(outDims, 0));
    EXPECT_CALL(*outputController, propagateShapes(_));
    launcher.reshape({});
}
