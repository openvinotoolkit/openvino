// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

// Builds a model with the ESPCN_x2_gh architecture (single-channel input, DepthToSpace x2 upscaling).
// The batch/height/width bounds and NHWC-vs-NCHW layout are parameterized so every test model in this file
// (previously MaxPool/CustomNet variants) shares the same, already-validated graph shape.
inline std::shared_ptr<ov::Model> createESPCNX2Model(ov::Dimension batchDimension = ov::Dimension(1, 2),
                                                      ov::Dimension heightDimension = ov::Dimension(32, 64),
                                                      ov::Dimension widthDimension = ov::Dimension(32, 64),
                                                      bool nhwcLayout = true) {
    const ov::PartialShape inputShape = nhwcLayout
                                            ? ov::PartialShape{batchDimension, heightDimension, widthDimension, 1}
                                            : ov::PartialShape{batchDimension, 1, heightDimension, widthDimension};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
    input->set_friendly_name("IteratorGetNext:0");
    input->get_output_tensor(0).set_names({"IteratorGetNext:0"});

    ov::Output<ov::Node> nchwInput = input;
    if (nhwcLayout) {
        // set_tensors()/batched inference requires the batch (N) dimension to be identifiable via layout.
        input->set_layout("NHWC");
        auto transposeOrder = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        nchwInput = std::make_shared<ov::op::v1::Transpose>(input, transposeOrder);
    } else {
        input->set_layout("NCHW");
    }

    const auto makeConvAdd = [](const ov::Output<ov::Node>& data,
                                size_t inputChannels,
                                size_t outputChannels,
                                size_t kernelSize,
                                float weightValue,
                                float biasValue) -> ov::Output<ov::Node> {
        auto weights = ov::op::v0::Constant::create(ov::element::f32,
                                                    ov::Shape{outputChannels,
                                                              inputChannels,
                                                              kernelSize,
                                                              kernelSize},
                                                    {weightValue});
        auto convolution = std::make_shared<ov::op::v1::Convolution>(data,
                                                                     weights,
                                                                     ov::Strides{1, 1},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::CoordinateDiff{0, 0},
                                                                     ov::Strides{1, 1},
                                                                     ov::op::PadType::SAME_UPPER);
        auto bias = ov::op::v0::Constant::create(ov::element::f32,
                                                 ov::Shape{1, outputChannels, 1, 1},
                                                 {biasValue});
        return std::make_shared<ov::op::v1::Add>(convolution, bias);
    };

    auto firstConv = makeConvAdd(nchwInput, 1, 64, 5, 0.01f, 0.001f);
    auto firstRelu = std::make_shared<ov::op::v0::Relu>(firstConv);
    auto secondConv = makeConvAdd(firstRelu, 64, 32, 3, 0.011f, 0.001f);
    auto secondRelu = std::make_shared<ov::op::v0::Relu>(secondConv);
    auto thirdConv = makeConvAdd(secondRelu, 32, 4, 3, 0.012f, 0.001f);
    auto depthToSpace = std::make_shared<ov::op::v0::DepthToSpace>(
        thirdConv,
        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        2);
    auto output = std::make_shared<ov::op::v0::Tanh>(depthToSpace);
    output->set_friendly_name("NCHW_output");
    output->get_output_tensor(0).set_names({"NCHW_output:0"});

    auto result = std::make_shared<ov::op::v0::Result>(output);
    if (!nhwcLayout) {
        result->set_layout("NCHW");
    }

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "ESPCN_x2_gh");
}

inline bool isESPCNX2Model(const std::string& modelName) {
    return modelName == "ESPCN_x2_gh";
}

inline ov::Shape dynamicNHWInputShape(const std::string& modelName, size_t batch, bool useLargeShape = false) {
    if (isESPCNX2Model(modelName)) {
        const size_t spatialDimension = useLargeShape ? 64 : 32;
        return {batch, spatialDimension, spatialDimension, 1};
    }
    return useLargeShape ? ov::Shape{batch, 1080, 1920, 1} : ov::Shape{batch, 720, 1280, 1};
}

// All test models now share the ESPCN_x2_gh architecture: single output channel, spatial dims doubled by
// DepthToSpace, regardless of the bounds used to build the input.
inline ov::Shape dynamicNHWOutputShape(const std::string&, const ov::Shape& inputShape) {
    return {inputShape[0], 1, inputShape[1] * 2, inputShape[2] * 2};
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
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(target_device, configuration, selectedModelName) = this->GetParam();

        configuration[ov::intel_npu::compile_log_level.name()] = ov::log::Level::ERR;
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
        return createESPCNX2Model(ov::Dimension(1), ov::Dimension(1, 1080), ov::Dimension(10, 1920));
    }
    if (modelName == "CustomNet_DynBatch") {
        return createESPCNX2Model(ov::Dimension(1, 10), ov::Dimension(1, 1080), ov::Dimension(10, 1920));
    }
    if (modelName == "MaxPool") {
        return createESPCNX2Model(ov::Dimension(1), ov::Dimension(10, 720), ov::Dimension(10, 1280));
    }
    if (modelName == "MaxPool_NCHW") {
        return createESPCNX2Model(ov::Dimension(1), ov::Dimension(10, 720), ov::Dimension(10, 1280), false);
    }
    if (modelName == "MaxPool_NCHW_DynBatch") {
        return createESPCNX2Model(ov::Dimension(1, 10), ov::Dimension(10, 720), ov::Dimension(10, 1280), false);
    }
    if (isESPCNX2Model(modelName)) {
        return createESPCNX2Model();
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
    if (isESPCNX2Model(selectedModelName)) {
        GTEST_SKIP() << "ESPCN_x2_gh is covered by the DynamicNHW tests";
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
    if (isESPCNX2Model(selectedModelName)) {
        GTEST_SKIP() << "ESPCN_x2_gh is covered by the DynamicNHW tests";
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
    ov::Shape shape = {1, 720, 1280, 1};
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

    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor1,
                            "CompileAndInferWithDecreasedSize_third");

    logCapture.clear();
    ov::Shape shape2 = {1, 720, 720, 1};
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
    if (isESPCNX2Model(selectedModelName)) {
        GTEST_SKIP() << "ESPCN_x2_gh is covered by the DynamicNHW tests";
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
    ov::Shape shape = {1, 720, 720, 1};
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

    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor1,
                            "CompileAndInferWithIncreasedSize_third");

    logCapture.clear();
    ov::Shape shape2 = {1, 720, 1280, 1};
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
    if (isESPCNX2Model(selectedModelName)) {
        GTEST_SKIP() << "ESPCN_x2_gh is covered by the DynamicNHW tests";
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
    ov::Shape shape = {1, 720, 1280, 1};
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
    // ESPCN_x2_gh upsamples 2x and its output is NCHW, so it cannot be reused as an NHWC input tensor;
    // reuse another request's input tensor instead to still exercise pointer-change detection.
    auto inputTensorFromReq = testContext.reqDynamic.get_tensor(model->input());
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            inputTensorFromReq,
                            "CompileAndInferWithZeroTensor_third");
    // Feeding an imported tensor from another infer request, ptr change detected and rebuild runtime
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

TEST_P(InferWithHostCompileTests, DynamicNHWUsesOneVMExecution) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }
    // MaxPool dynamic models contain operators that are not yet supported by the dynamic pipeline.
    // CustomNet_DynBatch is used to verify aggregation of N=1 tensors into one N=2 VM execution.
    if (selectedModelName != "CustomNet_DynBatch" && !isESPCNX2Model(selectedModelName)) {
        GTEST_SKIP() << "Only applies to the dynamic-batch model";
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

    ov::InferRequest reqDynamic1 = testContext.compiledModel.create_infer_request();
    ov::InferRequest reqReference1 = testContext.referenceCompiledModel.create_infer_request();

    // A single N=2 tensor must execute as one dynamic VM inference.
    const ov::Shape batchShape = dynamicNHWInputShape(selectedModelName, 2);
    auto fullBatchTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), batchShape, 100, 0);
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            fullBatchTensor,
                            "DynamicBatchUsesOneVMExecution_full_batch");
    ASSERT_EQ(reqDynamic1.get_tensor(model->output()).get_shape(),
              dynamicNHWOutputShape(selectedModelName, batchShape));

    const auto countVMExecutions = [](const std::string& log) {
        constexpr std::string_view marker = "Start to execute graph with runtime engine";
        size_t count = 0;
        size_t position = 0;
        while ((position = log.find(marker, position)) != std::string::npos) {
            ++count;
            position += marker.size();
        }
        return count;
    };
    ASSERT_EQ(countVMExecutions(logCapture.str()), 1u) << logCapture.str();

    logCapture.clear();
    // Two N=1 tensors must be aggregated into one N=2 inference rather than executed separately.
    const ov::Shape singleBatchShape = dynamicNHWInputShape(selectedModelName, 1);
    std::vector<ov::Tensor> tensorBatch;
    tensorBatch.push_back(
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), singleBatchShape, 100, 0));
    tensorBatch.push_back(
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), singleBatchShape, 100, 100));
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensors(testContext.compiledModel.input(), tensorBatch));
    OV_ASSERT_NO_THROW(reqReference1.set_tensors(testContext.referenceCompiledModel.input(), tensorBatch));
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    OV_ASSERT_NO_THROW(reqReference1.infer());
    ASSERT_EQ(reqDynamic1.get_tensor(model->output()).get_shape(),
              dynamicNHWOutputShape(selectedModelName, batchShape));
    ov::test::utils::compare(reqReference1.get_tensor(model->output()),
                             reqDynamic1.get_tensor(model->output()),
                             model->output().get_element_type());

    ASSERT_EQ(countVMExecutions(logCapture.str()), 1u) << logCapture.str();
}

// Grow N, H and W simultaneously (still within the model's declared bounds: N in [1,10], H in [1,1080], W in
// [10,1920]) and verify both output correctness and command-list reconfiguration behavior.
TEST_P(InferWithHostCompileTests, DynamicNHWIncreasedSize) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }
    // MaxPool dynamic models contain operators that are not yet supported by the dynamic pipeline.
    if (selectedModelName != "CustomNet_DynBatch" && !isESPCNX2Model(selectedModelName)) {
        GTEST_SKIP() << "Only applies to the dynamic-batch model";
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

    // Start with a small valid N/H/W combination.
    ov::Shape smallShape = dynamicNHWInputShape(selectedModelName, 1);
    ov::Tensor smallTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), smallShape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            smallTensor,
                            "DynamicNHWIncreasedSize_small");
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    // Grow N, H and W at once.
    ov::Shape largeShape = dynamicNHWInputShape(selectedModelName, 2, true);
    ov::Tensor largeTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), largeShape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            largeTensor,
                            "DynamicNHWIncreasedSize_large");
    ASSERT_EQ(testContext.reqDynamic.get_tensor(model->output()).get_shape(),
              dynamicNHWOutputShape(selectedModelName, largeShape));
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' after growing N/H/W simultaneously, but "
           "got: "
        << logCapture.str();
}

// Shrink N, H and W simultaneously and verify both output correctness and command-list reconfiguration behavior.
TEST_P(InferWithHostCompileTests, DynamicNHWDecreasedSize) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }
    // MaxPool dynamic models contain operators that are not yet supported by the dynamic pipeline.
    if (selectedModelName != "CustomNet_DynBatch" && !isESPCNX2Model(selectedModelName)) {
        GTEST_SKIP() << "Only applies to the dynamic-batch model";
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

    // Start with the larger N/H/W combination.
    ov::Shape largeShape = dynamicNHWInputShape(selectedModelName, 2, true);
    ov::Tensor largeTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), largeShape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            largeTensor,
                            "DynamicNHWDecreasedSize_large");
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    // Shrink N, H and W at once.
    ov::Shape smallShape = dynamicNHWInputShape(selectedModelName, 1);
    ov::Tensor smallTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), smallShape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            smallTensor,
                            "DynamicNHWDecreasedSize_small");
    ASSERT_EQ(testContext.reqDynamic.get_tensor(model->output()).get_shape(),
              dynamicNHWOutputShape(selectedModelName, smallShape));
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' after shrinking N/H/W simultaneously, "
           "but got: "
        << logCapture.str();
}

using InferWithDefaultHostCompileTests = InferWithHostCompileTests;

inline bool isByteCodeBlob(const std::string& blob) {
    const size_t headerSize = std::min(blob.size(), size_t{20});
    const std::string_view header(blob.data(), headerSize);
    return header.find("NPUByte\x00") != std::string_view::npos;
};

inline bool isElfBlob(const std::string& blob) {
    const size_t headerSize = std::min(blob.size(), size_t{20});
    const std::string_view header(blob.data(), headerSize);
    return header.find("ELF\x00") != std::string_view::npos;
};

TEST_P(InferWithDefaultHostCompileTests, CompileDynamicModelWithNoHostCompileMode) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createModelByName(selectedModelName);

    ov::CompiledModel compiledModel;
    // Compilation shall pass since load of openvino_intel_npu_mlir_runtime is deffered with NPU_CREATE_EXECUTOR=0
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    if (modelStream.str().empty()) {
        FAIL() << "Exported model stream is empty";
    }

    if (selectedModelName == "MaxPool_NCHW_DynBatch") {
        ASSERT_TRUE(isElfBlob(modelStream.str())) << "Expected exported model to be an ELF blob";
    } else if (selectedModelName == "MaxPool_NCHW") {
        ASSERT_TRUE(isByteCodeBlob(modelStream.str())) << "Expected exported model to be a bytecode";
    }

    ov::InferRequest reqDynamic;
    try {
        ov::CompiledModel importedModel = core->import_model(modelStream, target_device);
        reqDynamic = importedModel.create_infer_request();
    } catch (const ov::Exception& e) {
        if (std::string(e.what()).find("Cannot load library") == std::string::npos) {
            FAIL() << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        } else {
            GTEST_SKIP() << "Cannot load library, skip test.";
        }
    }

    OV_ASSERT_NO_THROW(reqDynamic.infer());
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

const std::vector<std::string> devices = {"NPU.4000", "NPU.5010"};

const std::vector<ov::AnyMap> configs = {
    {
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_COMPILATION_MODE", "HostCompile_Interpreter"},
        {"NPU_CREATE_EXECUTOR", "0"},
    },
    {
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_COMPILATION_MODE", "HostCompile_Interpreter"},
    },
    {
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_COMPILATION_MODE", "HostCompile_Interpreter"},
        {"NPU_CREATE_EXECUTOR", "0"},
        {"NPU_BATCH_MODE", "PLUGIN"},
    },
    {
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_COMPILATION_MODE", "HostCompile_Interpreter"},
        {"NPU_BATCH_MODE", "PLUGIN"},
    },
};

// All model names below build the same ESPCN_x2_gh graph (see createModelByName) with different dynamic
// N/H/W bounds and layouts, so each existing test's concrete/reused shapes remain valid.
const std::vector<std::string> modelNames = {"CustomNet", "CustomNet_DynBatch", "MaxPool", "ESPCN_x2_gh"};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferWithHostCompileTests,
                         ::testing::Combine(::testing::ValuesIn(devices),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(modelNames)),
                         ov::test::utils::appendPlatformTypeTestName<InferWithHostCompileTests>);

const std::vector<ov::AnyMap> defaultHostCompileconfigs = {
    {
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_CREATE_EXECUTOR", "0"},
    },
    {
        {"NPU_COMPILER_TYPE", "PLUGIN"},
        {"NPU_CREATE_EXECUTOR", "0"},
        {"NPU_BATCH_MODE", "PLUGIN"},
    },
};

const std::vector<std::string> defaultHCModelNames = {"MaxPool_NCHW", "MaxPool_NCHW_DynBatch"};
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferWithDefaultHostCompileTests,
                         ::testing::Combine(::testing::ValuesIn(devices),
                                            ::testing::ValuesIn(defaultHostCompileconfigs),
                                            ::testing::ValuesIn(defaultHCModelNames)),
                         ov::test::utils::appendPlatformTypeTestName<InferWithDefaultHostCompileTests>);
