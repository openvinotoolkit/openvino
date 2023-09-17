// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>
#include <future>

#include "base/behavior_test_utils.hpp"
#include "shared_test_classes/subgraph/basic_lstm.hpp"

namespace BehaviorTestsDefinitions {
using InferRequestIOBBlobTest = BehaviorTestsUtils::InferRequestTests;

TEST_P(InferRequestIOBBlobTest, CanCreateInferRequest) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
}

TEST_P(InferRequestIOBBlobTest, failToSetNullptrForInput) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetNullptrForOutput) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr outputBlob = nullptr;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, outputBlob),
                 InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetUninitializedInputBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetUninitializedOutputBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob;
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob),
            InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, setNotAllocatedInput) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
}

TEST_P(InferRequestIOBBlobTest, setNotAllocatedOutput) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
}

TEST_P(InferRequestIOBBlobTest, getAfterSetInputDoNotChangeInput) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> inputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob = nullptr;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));

    ASSERT_TRUE(actualBlob);
    ASSERT_FALSE(actualBlob->buffer() == nullptr);
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(cnnNet.getInputsInfo().begin()->second->getTensorDesc() == actualBlob->getTensorDesc());
}

TEST_P(InferRequestIOBBlobTest, getAfterSetInputDoNotChangeOutput) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> inputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, inputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(actualBlob);
    ASSERT_FALSE(actualBlob->buffer() == nullptr);
    ASSERT_EQ(inputBlob.get(), actualBlob.get());

    ASSERT_TRUE(cnnNet.getOutputsInfo().begin()->second->getTensorDesc() == actualBlob->getTensorDesc());
}

TEST_P(InferRequestIOBBlobTest, failToSetBlobWithIncorrectName) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    const char incorrect_input_name[] = "incorrect_input_name";
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    blob->allocate();
    ASSERT_THROW(req.SetBlob(incorrect_input_name, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetInputWithIncorrectSizes) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    auto td = cnnNet.getInputsInfo().begin()->second->getTensorDesc();
    auto dims = td.getDims();
    dims[0] *= 2;
    td.reshape(dims);

    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(td);
    blob->allocate();
    ASSERT_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, failToSetOutputWithIncorrectSizes) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    auto td = cnnNet.getOutputsInfo().begin()->second->getTensorDesc();
    auto dims = td.getDims();
    dims[0] *= 2;
    td.reshape(dims);

    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(td);
    blob->allocate();
    ASSERT_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, canInferWithoutSetAndGetInOutSync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestIOBBlobTest, canInferWithoutSetAndGetInOutAsync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedInputBlobAfterGetBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedInputBlobAfterGetAndSetBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob;
    req = execNet.CreateInferRequest();
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW(req.StartAsync());
}

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedInputBlobAfterSetBlobSync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedInputBlobAfterSetBlobAsync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW({ req.StartAsync(); req.Wait(); }, InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedOutputBlobAfterSetBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
    ASSERT_THROW({ req.StartAsync(); req.Wait(); }, InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, canProcessDeallocatedOutputBlobAfterGetAndSetBlob) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob =
            FuncTestUtils::createAndFillBlob(cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    blob->allocate();
    ASSERT_NO_THROW(blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, blob));
    blob->deallocate();
    ASSERT_THROW(req.Infer(), InferenceEngine::Exception);
    ASSERT_THROW({ req.StartAsync(); req.Wait(); }, InferenceEngine::Exception);
}

TEST_P(InferRequestIOBBlobTest, secondCallGetInputDoNotReAllocateData) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTest, secondCallGetOutputDoNotReAllocateData) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTest, secondCallGetInputAfterInferSync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW({ req.StartAsync(); req.Wait(); });
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTest, secondCallGetOutputAfterInferSync) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    InferenceEngine::Blob::Ptr blob1, blob2;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    ASSERT_NO_THROW(req.Infer());
    ASSERT_NO_THROW({ req.StartAsync(); req.Wait(); });
    ASSERT_NO_THROW(blob1 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_NO_THROW(blob2 = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(blob1.get(), blob2.get());
}

TEST_P(InferRequestIOBBlobTest, canSetInputBlobForInferRequest) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob));
    InferenceEngine::Blob::Ptr actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first));
    ASSERT_EQ(inputBlob, actualBlob);
}

TEST_P(InferRequestIOBBlobTest, canSetOutputBlobForInferRequest) {
    // Create InferRequest
    InferenceEngine::InferRequest req = execNet.CreateInferRequest();
    std::shared_ptr<InferenceEngine::Blob> outputBlob = FuncTestUtils::createAndFillBlob(
            cnnNet.getOutputsInfo().begin()->second->getTensorDesc());
    ASSERT_NO_THROW(req.SetBlob(cnnNet.getOutputsInfo().begin()->first, outputBlob));
    std::shared_ptr<InferenceEngine::Blob> actualBlob;
    ASSERT_NO_THROW(actualBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
    ASSERT_EQ(outputBlob.get(), actualBlob.get());
}

TEST_P(InferRequestIOBBlobTest, canInferWithSetInOutBlobs) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, inputBlob);
    InferenceEngine::Blob::Ptr outputBlob =
            FuncTestUtils::createAndFillBlob(cnnNet.getInputsInfo().begin()->second->getTensorDesc());
    req.SetBlob(cnnNet.getInputsInfo().begin()->first, outputBlob);
    ASSERT_NO_THROW(req.Infer());
}

TEST_P(InferRequestIOBBlobTest, canInferWithGetIn) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getInputsInfo().begin()->first);
    ASSERT_NO_THROW(req.Infer());
    InferenceEngine::StatusCode sts;
    ASSERT_NO_THROW({ req.StartAsync(); sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
}

TEST_P(InferRequestIOBBlobTest, canInferWithGetOut) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::Blob::Ptr inputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    ASSERT_NO_THROW(req.Infer());
    InferenceEngine::StatusCode sts;
    ASSERT_NO_THROW({ req.StartAsync(); sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY); });
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    ASSERT_NO_THROW(InferenceEngine::Blob::Ptr outputBlob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first));
}

TEST_P(InferRequestIOBBlobTest, canReallocateExternalBlobViaGet) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 10, 10});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("param");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        relu->set_friendly_name("relu");
        auto result = std::make_shared<ngraph::op::Result>(relu);
        result->set_friendly_name("result");

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    // Create CNNNetwork from ngraph::Function
    InferenceEngine::CNNNetwork cnnNet(ngraph);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    // Create InferRequest
    auto req = execNet.CreateInferRequest();
    auto inBlob = req.GetBlob("param");
    auto outBlob = req.GetBlob("relu");
    inBlob->allocate();
    outBlob->allocate();

    ASSERT_NO_THROW(req.Infer());
}

class InferRequestIOBBlobSetPrecisionTest : public BehaviorTestsUtils::BehaviorTestsBasicBase,
                                            public BehaviorTestsUtils::IEInferRequestTestBase {
protected:
    void SetUp() override {
        std::tie(netPrecision, target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
        cnnNet = InferenceEngine::CNNNetwork(function);
        execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            ::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }

    InferenceEngine::ExecutableNetwork execNet;
    InferenceEngine::CNNNetwork cnnNet;
};


TEST_P(InferRequestIOBBlobSetPrecisionTest, CanSetOutBlobWithDifferentPrecision) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    for (auto& outputInfo : cnnNet.getOutputsInfo()) {
        InferenceEngine::TensorDesc td(netPrecision, outputInfo.second->getTensorDesc().getDims(), outputInfo.second->getTensorDesc().getLayout());
        InferenceEngine::Blob::Ptr blob = FuncTestUtils::createAndFillBlob(td);
        if (outputInfo.second->getTensorDesc().getPrecision() == netPrecision) {
            ASSERT_NO_THROW(req.SetBlob(outputInfo.first, blob));
        } else {
            ASSERT_THROW(req.SetBlob(outputInfo.first, blob), InferenceEngine::Exception);
        }
    }
}

TEST_P(InferRequestIOBBlobSetPrecisionTest, CanSetInBlobWithDifferentPrecision) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    for (auto& inputInfo : cnnNet.getInputsInfo()) {
        InferenceEngine::TensorDesc td(netPrecision, inputInfo.second->getTensorDesc().getDims(), inputInfo.second->getTensorDesc().getLayout());
        InferenceEngine::Blob::Ptr blob = FuncTestUtils::createAndFillBlob(td);
        if (inputInfo.second->getTensorDesc().getPrecision() == netPrecision) {
            ASSERT_NO_THROW(req.SetBlob(inputInfo.first, blob));
        } else {
            ASSERT_THROW(req.SetBlob(inputInfo.first, blob), InferenceEngine::Exception);
        }
    }
}

typedef std::tuple<
        InferenceEngine::Layout,            // Network precision
        std::string,                        // Device name
        std::map<std::string, std::string>  // Config
> InferRequestIOBBlobSetLayoutParams;

class InferRequestIOBBlobSetLayoutTest : public testing::WithParamInterface<InferRequestIOBBlobSetLayoutParams>,
                                         public ov::test::behavior::APIBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestIOBBlobSetLayoutParams> obj) {
        using namespace ov::test::utils;
        InferenceEngine::Layout  layout;
        std::string target_device;
        std::map<std::string, std::string> configuration;
        std::tie(layout, target_device, configuration) = obj.param;
        std::ostringstream result;
        result << "layout=" << layout << "_";
        result << "target_device=" << target_device << "_";
        if (!configuration.empty()) {
            result << "config=" << configuration;
        }
        return result.str();
    }

    void SetUp()  override {
        std::tie(layout, target_device, configuration) = this->GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        function = ngraph::builder::subgraph::makeConvPoolRelu();
        cnnNet = InferenceEngine::CNNNetwork(function);
        execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    }

    void TearDown() override {
        if (!configuration.empty()) {
            PluginCache::get().reset();
        }
        APIBaseTest::SetUp();
    }

    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    std::shared_ptr<ngraph::Function> function;
    InferenceEngine::Layout layout;
    InferenceEngine::CNNNetwork cnnNet;
    InferenceEngine::ExecutableNetwork execNet;
    std::map<std::string, std::string> configuration;
};

TEST_P(InferRequestIOBBlobSetLayoutTest, CanSetInBlobWithDifferentLayouts) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());

    for (auto& inputInfo : cnnNet.getInputsInfo()) {
        InferenceEngine::TensorDesc td;
        InferenceEngine::Blob::Ptr blob;
        if (FuncTestUtils::checkLayout(layout, inputInfo.second->getTensorDesc().getDims())) {
            ASSERT_NO_THROW(td = InferenceEngine::TensorDesc(inputInfo.second->getTensorDesc().getPrecision(),
                                                                       inputInfo.second->getTensorDesc().getDims(), layout));
            ASSERT_NO_THROW(blob = FuncTestUtils::createAndFillBlob(td));
            if (inputInfo.second->getLayout() == layout || layout == InferenceEngine::Layout::ANY ||
                layout == InferenceEngine::Layout::BLOCKED || layout == InferenceEngine::Layout::SCALAR) {
                ASSERT_NO_THROW(req.SetBlob(inputInfo.first, blob));
            } else {
                ASSERT_ANY_THROW(req.SetBlob(inputInfo.first, blob));
            }
        } else {
            ASSERT_THROW(td = InferenceEngine::TensorDesc(inputInfo.second->getTensorDesc().getPrecision(),
                                                          inputInfo.second->getTensorDesc().getDims(), layout), InferenceEngine::Exception);
            ASSERT_THROW(blob = FuncTestUtils::createAndFillBlob(td), InferenceEngine::Exception);
        }
    }
}

TEST_P(InferRequestIOBBlobSetLayoutTest, CanSetOutBlobWithDifferentLayouts) {
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    for (auto& outputInfo : cnnNet.getOutputsInfo()) {
        InferenceEngine::TensorDesc td;
        InferenceEngine::Blob::Ptr blob;
        if (FuncTestUtils::checkLayout(layout, outputInfo.second->getTensorDesc().getDims())) {
            ASSERT_NO_THROW(td = InferenceEngine::TensorDesc(outputInfo.second->getTensorDesc().getPrecision(),
                                                             outputInfo.second->getTensorDesc().getDims(), layout));
            ASSERT_NO_THROW(blob = FuncTestUtils::createAndFillBlob(td));
            if (outputInfo.second->getLayout() == layout || layout == InferenceEngine::Layout::ANY ||
                layout == InferenceEngine::Layout::BLOCKED || layout == InferenceEngine::Layout::SCALAR) {
                ASSERT_NO_THROW(req.SetBlob(outputInfo.first, blob));
            } else {
                ASSERT_ANY_THROW(req.SetBlob(outputInfo.first, blob));
            }
        } else {
            ASSERT_THROW(td = InferenceEngine::TensorDesc(outputInfo.second->getTensorDesc().getPrecision(),
                                                          outputInfo.second->getTensorDesc().getDims(), layout), InferenceEngine::Exception);
            ASSERT_THROW(blob = FuncTestUtils::createAndFillBlob(td), InferenceEngine::Exception);
        }
    }
}

}  // namespace BehaviorTestsDefinitions
