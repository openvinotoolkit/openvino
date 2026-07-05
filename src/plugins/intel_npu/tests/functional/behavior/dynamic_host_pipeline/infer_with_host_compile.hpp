// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstring>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ov::Model> createMaxPoolModel() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                         ov::PartialShape{1, 16, ov::Dimension(10, 720), ov::Dimension(10, 1280)});
    input->set_friendly_name("input1");

    auto maxpool = std::make_shared<ov::op::v1::MaxPool>(input,
                                                         Strides{1, 1},
                                                         Shape{0, 0},
                                                         Shape{0, 0},
                                                         Shape{1, 1},
                                                         op::RoundingType::FLOOR,
                                                         op::PadType::EXPLICIT);
    maxpool->set_friendly_name("MaxPool_2");

    auto result = std::make_shared<ov::op::v0::Result>(maxpool);
    result->set_friendly_name("output");
    auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "MaxPool");
    // making input and output to be NHWC
    auto preProc = ov::preprocess::PrePostProcessor(model);
    preProc.input(0).tensor().set_layout("NHWC");
    preProc.input(0).model().set_layout("NCHW");
    preProc.output(0).tensor().set_layout("NHWC");
    preProc.output(0).model().set_layout("NCHW");

    model = preProc.build();

    return model;
}

inline std::shared_ptr<ov::Model> createCustomNetModel() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16,
                                                         ov::PartialShape{1, 16, ov::Dimension(1, 1080), ov::Dimension(10, 1920)});
    input->set_friendly_name("Parameter_59");

    auto make_conv_add = [](const ov::Output<ov::Node>& data,
                            const std::string& convName,
                            const std::string& addName,
                            float weightValue,
                            float biasValue) -> ov::Output<ov::Node> {
        const std::vector<float> weightValues(16 * 16, weightValue);
        const std::vector<float> biasValues(16, biasValue);

        auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{16, 16, 1, 1}, weightValues);
        auto conv = std::make_shared<ov::op::v1::Convolution>(data,
                                                              weights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1},
                                                              ov::op::PadType::EXPLICIT);
        conv->set_friendly_name(convName);

        auto bias = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 1, 1}, biasValues);
        auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
        add->set_friendly_name(addName);
        return add;
    };

    auto x = make_conv_add(input, "Convolution_61", "Add_63", 0.01f, 0.001f);
    x = make_conv_add(x, "Convolution_65", "Add_67", 0.011f, 0.001f);

    auto relu68 = std::make_shared<ov::op::v0::Relu>(x);
    relu68->set_friendly_name("Relu_68");
    x = relu68;

    x = make_conv_add(x, "Convolution_70", "Add_72", 0.012f, 0.001f);
    auto relu73 = std::make_shared<ov::op::v0::Relu>(x);
    relu73->set_friendly_name("Relu_73");
    x = relu73;

    x = make_conv_add(x, "Convolution_75", "Add_77", 0.013f, 0.001f);
    auto relu78 = std::make_shared<ov::op::v0::Relu>(x);
    relu78->set_friendly_name("Relu_78");
    x = relu78;

    x = make_conv_add(x, "Convolution_82", "Add_84", 0.014f, 0.001f);
    auto relu85 = std::make_shared<ov::op::v0::Relu>(x);
    relu85->set_friendly_name("Relu_85");
    x = relu85;

    x = make_conv_add(x, "Convolution_87", "Add_89", 0.015f, 0.001f);
    auto relu90 = std::make_shared<ov::op::v0::Relu>(x);
    relu90->set_friendly_name("Relu_90");
    x = relu90;

    x = make_conv_add(x, "Convolution_92", "Add_94", 0.016f, 0.001f);
    auto relu95 = std::make_shared<ov::op::v0::Relu>(x);
    relu95->set_friendly_name("Relu_95");
    x = relu95;

    auto multiplyScale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 16, 1, 1}, {0.5f});
    auto multiply97 = std::make_shared<ov::op::v1::Multiply>(x, multiplyScale);
    multiply97->set_friendly_name("Multiply_97");

    auto add98 = std::make_shared<ov::op::v1::Add>(multiply97, multiply97);
    add98->set_friendly_name("Add_98");

    x = make_conv_add(add98, "Convolution_100", "Add_102", 0.017f, 0.001f);

    auto result = std::make_shared<ov::op::v0::Result>(x);
    result->set_friendly_name("Result_104");

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "CustomNet");

    // making input and output to be NHWC
    auto preProc = ov::preprocess::PrePostProcessor(model);
    preProc.input(0).tensor().set_layout("NHWC");
    preProc.input(0).model().set_layout("NCHW");
    preProc.output(0).tensor().set_layout("NHWC");
    preProc.output(0).model().set_layout("NCHW");

    model = preProc.build();

    return model;
}

using InferWithHostCompileParams = std::tuple<std::string,  // Device name
                                              ov::AnyMap,   // Config
                                              std::string   // Model name
                                              >;

// These tests are required by the NPU plugin to verify the support of dynamic shape during
// compilation and inference on different NPU drivers
class InferWithHostCompileTests : public testing::WithParamInterface<InferWithHostCompileParams>,
                                  public OVInferRequestTestBase {
public:
    enum class RuntimeCompareStatus {
        ready,
        skip,
        fail,
    };

    struct ScopedLogCapture {
        ScopedLogCapture();
        ~ScopedLogCapture();

        void clear();
        std::string str() const;

    private:
        std::stringstream stream;
        std::function<void(std::string_view)> callback;

        friend class InferWithHostCompileTests;
    };

    struct RuntimeCompareContext {
        std::shared_ptr<ov::Model> model;
        ov::CompiledModel compiledModel;
        ov::CompiledModel referenceCompiledModel;
        ov::InferRequest reqDynamic;
        ov::InferRequest reqReference;
    };

    struct RuntimeCompareSetupResult {
        RuntimeCompareStatus status = RuntimeCompareStatus::ready;
        std::string message;
        RuntimeCompareContext context;
    };

    static std::string getTestCaseName(testing::TestParamInfo<InferWithHostCompileParams> obj) {
        std::string target_device;
        ov::AnyMap configuration;
        std::string modelName;
        std::tie(target_device, configuration, modelName) = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << target_device << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
                result << "_";
            }
        }
        result << "model=" << modelName;
        return result.str();
    }

    void SetUp() {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        std::tie(target_device, configuration, selectedModelName) = this->GetParam();

        std::vector<std::string> deviceNames =
            core->get_property("NPU", ov::available_devices.name()).as<std::vector<std::string>>();
        for (auto name : deviceNames) {
            if (target_device.find(name) != std::string::npos) {
                isTargetDevice = true;
                break;
            }
        }
        originalLogLevel = core->get_property("NPU", ov::log::level.name()).as<ov::log::Level>();

        APIBaseTest::SetUp();
    }

    void TearDown() {
        core->set_property("NPU", ov::log::level(originalLogLevel));
    }

    static void compareInferenceResult(const std::shared_ptr<ov::Model>& model,
                                       ov::InferRequest& reqDynamic,
                                       ov::InferRequest& reqReference);

    static void inferAndCompare(const std::shared_ptr<ov::Model>& model,
                                ov::InferRequest& reqDynamic,
                                ov::InferRequest& reqReference,
                                const std::string& dumpPrefix);

    static void setInputInferAndCompare(const std::shared_ptr<ov::Model>& model,
                                        ov::InferRequest& reqDynamic,
                                        ov::InferRequest& reqReference,
                                        const ov::Tensor& inputTensor,
                                        const std::string& dumpPrefix);

    static bool logContains(const ScopedLogCapture& logCapture, const std::string& expectedEntry);

    static std::shared_ptr<ov::Model> createModelByName(const std::string& modelName);

    RuntimeCompareSetupResult prepareRuntimeCompareContext(const std::shared_ptr<ov::Model>& model);

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::string selectedModelName;
    bool isTargetDevice = false;
    ov::log::Level originalLogLevel = ov::log::Level::ERR;
};

InferWithHostCompileTests::ScopedLogCapture::ScopedLogCapture()
    : callback([this](std::string_view s) {
          stream << s << std::endl;
      }) {
    ov::util::set_log_callback(callback);
}

InferWithHostCompileTests::ScopedLogCapture::~ScopedLogCapture() {
    ov::util::reset_log_callback();
}

void InferWithHostCompileTests::ScopedLogCapture::clear() {
    stream.str("");
    stream.clear();
}

std::string InferWithHostCompileTests::ScopedLogCapture::str() const {
    return stream.str();
}

void InferWithHostCompileTests::compareInferenceResult(const std::shared_ptr<ov::Model>& model,
                                                       ov::InferRequest& reqDynamic,
                                                       ov::InferRequest& reqReference) {
    const auto inputTensor = reqDynamic.get_input_tensor(0);
    const auto npuOutputTensor = reqDynamic.get_tensor(model->output());
    const auto referenceOutputTensor = reqReference.get_tensor(model->output());

    ov::test::utils::compare(referenceOutputTensor, npuOutputTensor, npuOutputTensor.get_element_type());
}

void InferWithHostCompileTests::inferAndCompare(const std::shared_ptr<ov::Model>& model,
                                                ov::InferRequest& reqDynamic,
                                                ov::InferRequest& reqReference,
                                                const std::string& stage) {
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    OV_ASSERT_NO_THROW(reqReference.infer());
    try {
        compareInferenceResult(model, reqDynamic, reqReference);
    } catch (const ov::Exception& e) {
        FAIL() << "Inference result comparison failed at stage " << stage << ": " << e.what();
    }
}

void InferWithHostCompileTests::setInputInferAndCompare(const std::shared_ptr<ov::Model>& model,
                                                        ov::InferRequest& reqDynamic,
                                                        ov::InferRequest& reqReference,
                                                        const ov::Tensor& inputTensor,
                                                        const std::string& stage) {
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inputTensor));
    OV_ASSERT_NO_THROW(reqReference.set_input_tensor(0, inputTensor));
    inferAndCompare(model, reqDynamic, reqReference, stage);
}

bool InferWithHostCompileTests::logContains(const ScopedLogCapture& logCapture, const std::string& expectedEntry) {
    return logCapture.str().find(expectedEntry) != std::string::npos;
}

std::shared_ptr<ov::Model> InferWithHostCompileTests::createModelByName(const std::string& modelName) {
    if (modelName == "CustomNet") {
        return createCustomNetModel();
    }
    if (modelName == "MaxPool") {
        return createMaxPoolModel();
    }

    OPENVINO_THROW("Unknown model name for InferWithHostCompileTests: ", modelName);
}

InferWithHostCompileTests::RuntimeCompareSetupResult InferWithHostCompileTests::prepareRuntimeCompareContext(
    const std::shared_ptr<ov::Model>& model) {
    RuntimeCompareSetupResult result;
    result.context.model = model;

    try {
        result.context.compiledModel = core->compile_model(model, target_device, configuration);
    } catch (const ov::Exception& e) {
        result.status = RuntimeCompareStatus::fail;
        result.message = std::string("Failed to compile model for target device: ") + e.what();
        return result;
    }

    try {
        result.context.referenceCompiledModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception& e) {
        result.status = RuntimeCompareStatus::skip;
        result.message = std::string("CPU plugin is not available for reference comparison: ") + e.what();
        return result;
    }

    try {
        result.context.reqDynamic = result.context.compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        result.status = RuntimeCompareStatus::fail;
        result.message = std::string("Failed to create dynamic infer request: ") + e.what();
        return result;
    }

    try {
        result.context.reqReference = result.context.referenceCompiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        result.status = RuntimeCompareStatus::fail;
        result.message = std::string("Failed to create reference infer request: ") + e.what();
        return result;
    }
    return result;
}

TEST_P(InferWithHostCompileTests, CompileAndImportAndInfer) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }
    auto model = createModelByName(selectedModelName);

    ov::CompiledModel compiledModel;

    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    ov::InferRequest reqDynamic;
    ov::CompiledModel importedModel;
    OV_ASSERT_NO_THROW(importedModel = core->import_model(modelStream, target_device));
    OV_ASSERT_NO_THROW(reqDynamic = importedModel.create_infer_request());
    OV_ASSERT_NO_THROW(reqDynamic.infer());
}

// Compile, infer with a large shape, then shrink the input shape and verify both output correctness and command-list
// reuse behavior.
TEST_P(InferWithHostCompileTests, CompileAndInferWithDecreasedSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createModelByName(selectedModelName);
    ScopedLogCapture logCapture;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));
    auto setupResult = prepareRuntimeCompareContext(model);
    if (setupResult.status == RuntimeCompareStatus::fail) {
        FAIL() << setupResult.message;
    }
    if (setupResult.status == RuntimeCompareStatus::skip) {
        GTEST_SKIP() << setupResult.message;
    }
    auto& testContext = setupResult.context;

    // Start with the largest shape in the dynamic range.
    ov::Shape shape = {1, 720, 1280, 16};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor,
                            "CompileAndInferWithDecreasedSize_first");
    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    inferAndCompare(model, testContext.reqDynamic, testContext.reqReference, "CompileAndInferWithDecreasedSize_second");
    // Reusing the same input should keep the existing command list intact.
    ASSERT_TRUE(logContains(logCapture, "Reuse command list without update since no tensor change detected"))
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << logCapture.str();

    logCapture.clear();
    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor1,
                            "CompileAndInferWithDecreasedSize_third");
    // A new host tensor with the same shape should still reuse the command list.
    ASSERT_TRUE(logContains(logCapture, "Reuse command list without update since no tensor change detected"))
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for third "
           "inference, but got: "
        << logCapture.str();

    logCapture.clear();
    ov::Shape shape2 = {1, 720, 720, 16};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor3,
                            "CompileAndInferWithDecreasedSize_fourth");
    // Shrinking the shape should force runtime reconfiguration for the new tensor layout.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << logCapture.str();
}

// Compile, infer with a small shape, then grow the input shape and verify both output correctness and command-list
// reuse behavior.
TEST_P(InferWithHostCompileTests, CompileAndInferWithIncreasedSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createModelByName(selectedModelName);
    ScopedLogCapture logCapture;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));
    auto setupResult = prepareRuntimeCompareContext(model);
    if (setupResult.status == RuntimeCompareStatus::fail) {
        FAIL() << setupResult.message;
    }
    if (setupResult.status == RuntimeCompareStatus::skip) {
        GTEST_SKIP() << setupResult.message;
    }

    auto& testContext = setupResult.context;

    // Start with a smaller valid dynamic shape.
    ov::Shape shape = {1, 720, 720, 16};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor,
                            "CompileAndInferWithIncreasedSize_first");
    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    inferAndCompare(model, testContext.reqDynamic, testContext.reqReference, "CompileAndInferWithIncreasedSize_second");
    // Reusing the same input should keep the existing command list intact.
    ASSERT_TRUE(logContains(logCapture, "Reuse command list without update since no tensor change detected"))
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << logCapture.str();

    logCapture.clear();
    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor1,
                            "CompileAndInferWithIncreasedSize_third");
    // A new host tensor with the same shape should still reuse the command list.
    ASSERT_TRUE(logContains(logCapture, "Reuse command list without update since no tensor change detected"))
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for third "
           "inference, but got: "
        << logCapture.str();

    logCapture.clear();
    ov::Shape shape2 = {1, 720, 1280, 16};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor3,
                            "CompileAndInferWithIncreasedSize_fourth");
    // Growing the shape should force runtime reconfiguration for the new tensor layout.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << logCapture.str();
}

// Exercise imported Level Zero tensors and verify both output correctness and command-list pointer updates.
TEST_P(InferWithHostCompileTests, CompileAndInferWithZeroTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createModelByName(selectedModelName);
    ScopedLogCapture logCapture;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));
    auto setupResult = prepareRuntimeCompareContext(model);
    if (setupResult.status == RuntimeCompareStatus::fail) {
        FAIL() << setupResult.message;
    }
    if (setupResult.status == RuntimeCompareStatus::skip) {
        GTEST_SKIP() << setupResult.message;
    }
    auto& testContext = setupResult.context;

    // Start from a regular host tensor.
    ov::Shape shape = {1, 720, 1280, 16};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor,
                            "CompileAndInferWithZeroTensor_first");

    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    ov::InferRequest reqDynamic1 = testContext.compiledModel.create_infer_request();
    ov::InferRequest reqReference1 = testContext.referenceCompiledModel.create_infer_request();
    setInputInferAndCompare(model, reqDynamic1, reqReference1, inTensor, "CompileAndInferWithZeroTensor_second");
    // A fresh infer request rebuilds runtime state on its first execution.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    auto outputTensorFromReq = testContext.reqDynamic.get_tensor(model->output());
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            outputTensorFromReq,
                            "CompileAndInferWithZeroTensor_third");
    // Feeding an imported output tensor, ptr change detected and rebuild runtime
    // TODO: Update commandlist once dynamic stride supported
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for third inference, but got: "
        << logCapture.str();

    logCapture.clear();
    auto zeroContext = core->get_default_context(target_device);
    auto inputHostTensorForForthInfer = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
    auto hostTensorSourceForForthInfer =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ASSERT_EQ(hostTensorSourceForForthInfer.get_byte_size(), inputHostTensorForForthInfer.get_byte_size())
        << "Source and destination tensors must have identical byte sizes for copy";
    std::memcpy(inputHostTensorForForthInfer.data(),
                hostTensorSourceForForthInfer.data(),
                hostTensorSourceForForthInfer.get_byte_size());
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            inputHostTensorForForthInfer,
                            "CompileAndInferWithZeroTensor_fourth");
    // Feeding a context-allocated host tensor, ptr change detected and rebuild runtime
    // TODO: Update commandlist once dynamic stride supported
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference, but got: "
        << logCapture.str();

    logCapture.clear();
    auto outputShape = reqDynamic1.get_tensor(model->output()).get_shape();
    auto zeroOutputTensorForFifthInfer = zeroContext.create_host_tensor(model->input().get_element_type(), outputShape);
    auto hostTensorSourceForOutputForFifthInfer =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), outputShape, 100, 0);
    ASSERT_EQ(hostTensorSourceForOutputForFifthInfer.get_byte_size(), zeroOutputTensorForFifthInfer.get_byte_size())
        << "Source and destination tensors must have identical byte sizes for copy";
    std::memcpy(zeroOutputTensorForFifthInfer.data(),
                hostTensorSourceForOutputForFifthInfer.data(),
                hostTensorSourceForOutputForFifthInfer.get_byte_size());
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensor(model->output(), zeroOutputTensorForFifthInfer));
    inferAndCompare(model, reqDynamic1, reqReference1, "CompileAndInferWithZeroTensor_fifth");
    // Feeding a context-allocated host tensor as output, ptr change detected and rebuild runtime
    // TODO: Update commandlist once dynamic stride supported
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for fifth inference, but got: "
        << logCapture.str();

    logCapture.clear();
    auto inputTensorForSixthInfer =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(),
                                                reqDynamic1.get_tensor(model->input()).get_shape(),
                                                100,
                                                0);

    auto outputShapeForSixthInfer = reqDynamic1.get_tensor(model->output()).get_shape();
    auto zeroOutputTensorForSixthInfer =
        zeroContext.create_host_tensor(model->input().get_element_type(), outputShapeForSixthInfer);
    auto hostTensorSourceForOutputForSixthInfer =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), outputShapeForSixthInfer, 100, 0);
    ASSERT_EQ(hostTensorSourceForOutputForSixthInfer.get_byte_size(), zeroOutputTensorForSixthInfer.get_byte_size())
        << "Source and destination tensors must have identical byte sizes for copy";
    std::memcpy(zeroOutputTensorForSixthInfer.data(),
                hostTensorSourceForOutputForSixthInfer.data(),
                hostTensorSourceForOutputForSixthInfer.get_byte_size());
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensor(model->output(), zeroOutputTensorForSixthInfer));
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            inputTensorForSixthInfer,
                            "CompileAndInferWithZeroTensor_sixth");
    // Feeding a context-allocated host tensor, ptr change detected and rebuild runtime
    // TODO: Update commandlist once dynamic stride supported
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for sixth inference, but got: "
        << logCapture.str();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
