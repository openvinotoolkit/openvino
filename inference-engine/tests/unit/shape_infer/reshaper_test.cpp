// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-matchers.h>

#include <inference_engine/shape_infer/ie_reshaper.hpp>
#include <shape_infer/mock_ishape_infer_impl.hpp>
#include <shape_infer/mock_shape_infer_extension.hpp>
#include <mock_icnn_network.hpp>
#include <../graph_tools/graph_test_base.hpp>
#include <shape_infer/mock_reshaper_launcher.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;
using namespace ::testing;
using namespace ::GraphTest;

class ReshaperTest : public GraphTestsBase {
protected:
    class TestLauncherCreator : public LauncherCreator {
    public:
        struct Mocks {
            MockReshapeLauncher::Ptr launcher;
            MockInputController* iController;
            MockOutputController* oController;
            MockIShapeInferImpl::Ptr shapeInferImpl;

            Mocks(const MockReshapeLauncher::Ptr& _launcher, MockInputController* _iController,
                  MockOutputController* _oController, const MockIShapeInferImpl::Ptr& _shapeInferImpl) :
                    launcher(_launcher), iController(_iController), oController(_oController),
                    shapeInferImpl(_shapeInferImpl) {}
        };

        ReshapeLauncher::Ptr
        createNotInputLauncher(const CNNLayer* layer, const std::vector<IShapeInferExtensionPtr>& extensions) override {
            return createLauncher(layer);
        }

        ReshapeLauncher::Ptr
        createInputLauncher(const CNNLayer* layer, const std::vector<IShapeInferExtensionPtr>& extensions) override {
            return createLauncher(layer);
        }

        std::vector<Mocks> getMocks() {
            return _mocks;
        }

    private:
        ReshapeLauncher::Ptr createLauncher(const CNNLayer* layer) {
            auto initializer = std::make_shared<MockReshapeLauncher::TestLauncherInitializer>();
            auto shapeInferImpl = std::make_shared<MockIShapeInferImpl>();
            auto mockLauncher = std::make_shared<MockReshapeLauncher>(initializer, layer, shapeInferImpl);
            _mocks.emplace_back(mockLauncher, initializer->getInputController(), initializer->getOutputController(),
                                shapeInferImpl);
            return mockLauncher;
        }

    private:
        std::vector<Mocks> _mocks;
    };

    class TestEmptyLauncherCreator : public LauncherCreator {
    public:
        ReshapeLauncher::Ptr
        createNotInputLauncher(const CNNLayer* layer, const std::vector<IShapeInferExtensionPtr>& extensions) override {
            return std::make_shared<FakeReshapeLauncher>(layer, std::make_shared<MockIShapeInferImpl>());;
        }

        ReshapeLauncher::Ptr
        createInputLauncher(const CNNLayer* layer, const std::vector<IShapeInferExtensionPtr>& extensions) override {
            return std::make_shared<InputReshapeLauncher>(layer, std::make_shared<MockIShapeInferImpl>());
        }
    };

    void prepareInputs(InputsDataMap& inputsMap, int batchSize = 1) override {
        GraphTestsBase::prepareInputs(inputsMap);
        for (auto layer = lhsLayers.begin(); layer != lhsLayers.end(); layer++) {
            if ((*layer)->insData.empty()) {
                (*layer)->type = "Input";
            }
        }
    }

    void SetUp() override {
        GraphTestsBase::SetUp();
        impl = std::make_shared<MockIShapeInferImpl>();
        CONNECT(0, 1);
    };

public:
    StatusCode sts = GENERAL_ERROR;
    ResponseDesc resp;
    static const std::string TEST_NAME;
    MockIShapeInferImpl::Ptr impl;
    Reshaper::Ptr reshaper;
};

const std::string ReshaperTest::TEST_NAME = "TEST_NAME";

TEST_F(ReshaperTest, canCreateReshaper) {
    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));
    Reshaper reshaper(mockNet);
}

TEST_F(ReshaperTest, throwOnAddNullExtension) {
    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));
    Reshaper reshaper(mockNet);
    MockShapeInferExtension::Ptr extension;
    ASSERT_THROW(reshaper.AddExtension(extension), InferenceEngineException);
}

TEST_F(ReshaperTest, canAddExtensionWithNotRegistered) {
    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));
    Reshaper reshaper(mockNet);
    const char* notRegistered[] = {TEST_NAME.c_str()};
    auto extension = std::make_shared<MockShapeInferExtension>();
    EXPECT_CALL(*extension.get(), getPrimitiveTypes(_, _, _)).WillOnce(DoAll(
            WithArg<0>(Invoke([&](char**& type) { type = const_cast<char**>(notRegistered); })),
            WithArg<1>(Invoke([&](unsigned int& size) { size = 1; })),
            Return(OK)));
    reshaper.AddExtension(extension);
}

TEST_F(ReshaperTest, throwOnExtensionWithAlreadyRegisteredImpl) {
    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));
    Reshaper reshaper(mockNet);
    auto extension = std::make_shared<MockShapeInferExtension>();
    const char* registered[] = {TEST_NAME.c_str(), "Convolution"};
    EXPECT_CALL(*extension.get(), getPrimitiveTypes(_, _, _)).WillOnce(DoAll(
            WithArg<0>(Invoke([&](char**& type) { type = const_cast<char**>(registered); })),
            WithArg<1>(Invoke([&](unsigned int& size) { size = 2; })),
            Return(OK)));
    ASSERT_THROW(reshaper.AddExtension(extension), InferenceEngineException);
}

TEST_F(ReshaperTest, canResetOnReshape) {
    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));
    auto testCreator = std::make_shared<TestLauncherCreator>();
    Reshaper reshaper(mockNet, testCreator);
    auto mocks = testCreator->getMocks();
    auto inputMock = mocks[0];
    EXPECT_CALL(*(inputMock.launcher).get(), setShapeByName(_, _));
    for (auto it:mocks) {
        EXPECT_CALL(*(it.launcher).get(), getLayerName()).WillRepeatedly(Return(it.launcher->realGetLayerName()));
        EXPECT_CALL(*(it.launcher).get(), reset());
        EXPECT_CALL(*(it.launcher).get(), reshape(_));
        EXPECT_CALL(*(it.launcher).get(), applyChanges(_));
    }

    const char* registered[] = {TEST_NAME.c_str()};
    auto extension = std::make_shared<MockShapeInferExtension>();
    EXPECT_CALL(*extension.get(), getPrimitiveTypes(_, _, _)).WillOnce(DoAll(
            WithArg<0>(Invoke([&](char**& type) { type = const_cast<char**>(registered); })),
            WithArg<1>(Invoke([&](unsigned int& size) { size = 1; })),
            Return(OK)));
    reshaper.AddExtension(extension);

    reshaper.run({{"0", {2}}});
}

TEST_F(ReshaperTest, canUpdateFakeImpl) {
    EXPECT_CALL(mockNet, getInputsInfo(_)).WillRepeatedly(WithArg<0>(Invoke([&](InputsDataMap& maps) {
        prepareInputs(maps);
    })));
    auto testCreator = std::make_shared<TestEmptyLauncherCreator>();
    Reshaper reshaper(mockNet, testCreator);
    auto newImpl = std::make_shared<MockIShapeInferImpl>();

    const char* registered[] = {""};
    auto extension = std::make_shared<MockShapeInferExtension>();
    EXPECT_CALL(*extension.get(), getPrimitiveTypes(_, _, _)).WillOnce(DoAll(
            WithArg<0>(Invoke([&](char**& type) { type = const_cast<char**>(registered); })),
            WithArg<1>(Invoke([&](unsigned int& size) { size = 1; })),
            Return(OK)));
    EXPECT_CALL(*extension.get(), getShapeInferImpl(_, _, _)).WillOnce(DoAll(
            WithArg<0>(Invoke([&](IShapeInferImpl::Ptr& impl) { impl = newImpl; })),
            Return(OK)));
    reshaper.AddExtension(extension);

    EXPECT_CALL(*newImpl.get(), inferShapes(_, _, _, _, _)).
            WillOnce(DoAll(
            WithArg<3>(Invoke([&](std::vector<SizeVector>& outShape) { outShape.push_back({2}); })), Return(OK)));
    reshaper.run({{"0", {2}}});
}
