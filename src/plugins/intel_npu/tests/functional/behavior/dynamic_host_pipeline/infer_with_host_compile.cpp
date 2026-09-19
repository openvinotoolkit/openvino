// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
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

// Builds a model with the ESPCN_x2 architecture (single-channel input, DepthToSpace x2 upscaling).
// The batch/height/width bounds and NHWC-vs-NCHW layout are parameterized so every test model in this file
// (see espcnModelConfigs) shares the same, already-validated graph shape.
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

    // The output is always NCHW-shaped (DepthToSpace produces NCHW). Tagging the result with an N-carrying
    // layout lets the compiler's batch auto-detection collect a batch coefficient for the output, which is
    // required (together with the input) to employ the debatch/host_pipeline method for the dynamic batch.
    // Without it the dynamic N reaches the NCE pipeline and crashes DepthToSpace/SplitNCEOpsOntoWorkloads.
    auto result = std::make_shared<ov::op::v0::Result>(output);
    result->set_layout("NCHW");

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "ESPCN_x2_gh");
}

// Every test model is the same ESPCN_x2 graph built with different dynamic N/H/W bounds and layout. Keeping the
// declared bounds and the model characteristics in one registry lets the test shapes be derived from the bounds
// (see makeInputShape) instead of hard-coding numbers that must be kept in sync by hand.
struct DynamicModelConfig {
    ov::Dimension batch;
    ov::Dimension height;
    ov::Dimension width;
    bool nhwcLayout;
    bool tinyVariant;  // small single-channel variant, only exercised by the DynamicNHW tests
};

inline const std::map<std::string, DynamicModelConfig>& espcnModelConfigs() {
    static const std::map<std::string, DynamicModelConfig> configs = {
        {"ESPCN_x2_DynHW_FHD", {ov::Dimension(1), ov::Dimension(1, 1080), ov::Dimension(10, 1920), true, false}},
        {"ESPCN_x2_DynNHW_FHD", {ov::Dimension(1, 10), ov::Dimension(1, 1080), ov::Dimension(10, 1920), true, false}},
        {"ESPCN_x2_DynHW_HD", {ov::Dimension(1), ov::Dimension(10, 720), ov::Dimension(10, 1280), true, false}},
        // Spatial upper bounds are kept large enough for the compiler's multi-cluster tiling of the dynamic H/W;
        // smaller bounds hit a compiler crash in MultiClusterStrategyAssignment on multi-tile devices.
        {"ESPCN_x2_DynNHW_Tiny", {ov::Dimension(1, 2), ov::Dimension(32, 270), ov::Dimension(32, 555), true, true}},
        {"ESPCN_x2_DynHW_HD_NCHW", {ov::Dimension(1), ov::Dimension(10, 720), ov::Dimension(10, 1280), false, false}},
        {"ESPCN_x2_DynNHW_HD_NCHW",
         {ov::Dimension(1, 10), ov::Dimension(10, 720), ov::Dimension(10, 1280), false, false}},
    };
    return configs;
}

inline const DynamicModelConfig& getModelConfig(const std::string& modelName) {
    const auto& configs = espcnModelConfigs();
    const auto it = configs.find(modelName);
    OPENVINO_ASSERT(it != configs.end(), "Unknown model name for InferWithHostCompileTests: ", modelName);
    return it->second;
}

inline bool isTinyDynamicModel(const std::string& modelName) {
    return getModelConfig(modelName).tinyVariant;
}

inline bool hasDynamicBatch(const std::string& modelName) {
    return getModelConfig(modelName).batch.is_dynamic();
}

// Resolve the model input's declared bounds into a concrete shape: batch is set explicitly (it is always the
// leading dimension for both NHWC and NCHW), static dims are kept as-is, and each dynamic spatial dim uses its
// upper bound for the large shape or half of it (never below the lower bound) for the small shape.
inline ov::Shape makeInputShape(const std::shared_ptr<ov::Model>& model, size_t batch, bool useLargeShape) {
    const ov::PartialShape& partialShape = model->input().get_partial_shape();
    ov::Shape shape;
    shape.reserve(partialShape.size());
    for (size_t i = 0; i < partialShape.size(); ++i) {
        const ov::Dimension& dim = partialShape[i];
        if (i == 0) {
            shape.push_back(batch);
        } else if (dim.is_static()) {
            shape.push_back(static_cast<size_t>(dim.get_length()));
        } else {
            const auto interval = dim.get_interval();
            const int64_t value =
                useLargeShape ? interval.get_max_val() : std::max(interval.get_min_val(), interval.get_max_val() / 2);
            shape.push_back(static_cast<size_t>(value));
        }
    }
    return shape;
}

// All test models share the ESPCN_x2 architecture: single output channel with the spatial dims doubled by
// DepthToSpace. The output is NCHW, so this expects an NHWC input shape {N, H, W, 1}.
inline ov::Shape dynamicNHWOutputShape(const ov::Shape& inputShape) {
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

    void SetUp() override {
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

    void TearDown() override {
        core->set_property("NPU", ov::log::level(originalLogLevel));
        APIBaseTest::TearDown();
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
    const DynamicModelConfig& config = getModelConfig(modelName);
    return createESPCNX2Model(config.batch, config.height, config.width, config.nhwcLayout);
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
        result.message = std::string("TEMPLATE plugin is not available for reference comparison: ") + e.what();
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

// GTEST_SKIP() must return from the test body, so the shared device guard lives in a macro rather than a helper.
#define SKIP_IF_NOT_TARGET_DEVICE()                     \
    SKIP_IF_CURRENT_TEST_IS_DISABLED()                  \
    if (!isTargetDevice) {                              \
        GTEST_SKIP() << "Skip test for current device"; \
    }

TEST_P(InferWithHostCompileTests, CompileAndImportAndInfer) {
    SKIP_IF_NOT_TARGET_DEVICE()
    if (isTinyDynamicModel(selectedModelName)) {
        GTEST_SKIP() << "The tiny ESPCN_x2 model is covered by the DynamicNHW tests";
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
    SKIP_IF_NOT_TARGET_DEVICE()
    if (isTinyDynamicModel(selectedModelName)) {
        GTEST_SKIP() << "The tiny ESPCN_x2 model is covered by the DynamicNHW tests";
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

    // Start with the largest shape allowed by the model's declared bounds.
    ov::Shape shape = makeInputShape(model, 1, /*useLargeShape=*/true);
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
    ov::Shape shape2 = makeInputShape(model, 1, /*useLargeShape=*/false);
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
    SKIP_IF_NOT_TARGET_DEVICE()
    if (isTinyDynamicModel(selectedModelName)) {
        GTEST_SKIP() << "The tiny ESPCN_x2 model is covered by the DynamicNHW tests";
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

    // Start with the smallest shape allowed by the model's declared bounds.
    ov::Shape shape = makeInputShape(model, 1, /*useLargeShape=*/false);
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
    ov::Shape shape2 = makeInputShape(model, 1, /*useLargeShape=*/true);
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
    SKIP_IF_NOT_TARGET_DEVICE()
    if (isTinyDynamicModel(selectedModelName)) {
        GTEST_SKIP() << "The tiny ESPCN_x2 model is covered by the DynamicNHW tests";
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

    // Start from a regular host tensor sized to the model's upper bounds.
    ov::Shape shape = makeInputShape(model, 1, /*useLargeShape=*/true);
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
    // A plain host tensor is copied into the request's reused internal L0 buffer, so its data pointer never changes
    // and the runtime keeps reusing the command list. Feed a context-allocated (Level Zero) tensor instead: it is
    // imported directly with a distinct data pointer, so the runtime detects the change and rebuilds the command list.
    auto zeroContext = core->get_default_context(target_device);
    auto inputTensorForThirdInfer = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
    auto hostTensorSourceForThirdInfer =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 50);
    ASSERT_EQ(hostTensorSourceForThirdInfer.get_byte_size(), inputTensorForThirdInfer.get_byte_size())
        << "Source and destination tensors must have identical byte sizes for copy";
    std::memcpy(inputTensorForThirdInfer.data(),
                hostTensorSourceForThirdInfer.data(),
                hostTensorSourceForThirdInfer.get_byte_size());
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            inputTensorForThirdInfer,
                            "CompileAndInferWithZeroTensor_third");
    // Feeding a context-allocated tensor with a new data pointer, ptr change detected and rebuild runtime
    // TODO: Update commandlist once dynamic stride supported
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for third inference, but got: "
        << logCapture.str();

    logCapture.clear();
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
    auto zeroOutputTensorForFifthInfer =
        zeroContext.create_host_tensor(model->output().get_element_type(), outputShape);
    auto hostTensorSourceForOutputForFifthInfer =
        ov::test::utils::create_and_fill_tensor(model->output().get_element_type(), outputShape, 100, 0);
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
        zeroContext.create_host_tensor(model->output().get_element_type(), outputShapeForSixthInfer);
    auto hostTensorSourceForOutputForSixthInfer =
        ov::test::utils::create_and_fill_tensor(model->output().get_element_type(), outputShapeForSixthInfer, 100, 0);
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
    SKIP_IF_NOT_TARGET_DEVICE()
    // Only dynamic-batch ESPCN_x2 variants can aggregate two N=1 tensors into one N=2 VM execution.
    if (!hasDynamicBatch(selectedModelName)) {
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
    const ov::Shape batchShape = makeInputShape(model, 2, /*useLargeShape=*/false);
    auto fullBatchTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), batchShape, 100, 0);
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            fullBatchTensor,
                            "DynamicBatchUsesOneVMExecution_full_batch");
    ASSERT_EQ(reqDynamic1.get_tensor(model->output()).get_shape(), dynamicNHWOutputShape(batchShape));

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
    const ov::Shape singleBatchShape = makeInputShape(model, 1, /*useLargeShape=*/false);
    std::vector<ov::Tensor> tensorBatch;
    tensorBatch.push_back(
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), singleBatchShape, 100, 0));
    tensorBatch.push_back(
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), singleBatchShape, 100, 100));
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensors(testContext.compiledModel.input(), tensorBatch));
    OV_ASSERT_NO_THROW(reqReference1.set_tensors(testContext.referenceCompiledModel.input(), tensorBatch));
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    OV_ASSERT_NO_THROW(reqReference1.infer());
    ASSERT_EQ(reqDynamic1.get_tensor(model->output()).get_shape(), dynamicNHWOutputShape(batchShape));
    ov::test::utils::compare(reqReference1.get_tensor(model->output()),
                             reqDynamic1.get_tensor(model->output()),
                             model->output().get_element_type());

    ASSERT_EQ(countVMExecutions(logCapture.str()), 1u) << logCapture.str();
}

// Grow N, H and W simultaneously within the model's declared bounds and verify both output correctness and
// command-list reconfiguration behavior.
TEST_P(InferWithHostCompileTests, DynamicNHWIncreasedSize) {
    SKIP_IF_NOT_TARGET_DEVICE()
    // Only dynamic-batch ESPCN_x2 variants are exercised by the dynamic-batch VM pipeline here.
    if (!hasDynamicBatch(selectedModelName)) {
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
    ov::Shape smallShape = makeInputShape(model, 1, /*useLargeShape=*/false);
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
    ov::Shape largeShape = makeInputShape(model, 2, /*useLargeShape=*/true);
    ov::Tensor largeTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), largeShape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            largeTensor,
                            "DynamicNHWIncreasedSize_large");
    ASSERT_EQ(testContext.reqDynamic.get_tensor(model->output()).get_shape(), dynamicNHWOutputShape(largeShape));
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' after growing N/H/W simultaneously, but "
           "got: "
        << logCapture.str();
}

// Shrink N, H and W simultaneously and verify both output correctness and command-list reconfiguration behavior.
TEST_P(InferWithHostCompileTests, DynamicNHWDecreasedSize) {
    SKIP_IF_NOT_TARGET_DEVICE()
    // Only dynamic-batch ESPCN_x2 variants are exercised by the dynamic-batch VM pipeline here.
    if (!hasDynamicBatch(selectedModelName)) {
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
    ov::Shape largeShape = makeInputShape(model, 2, /*useLargeShape=*/true);
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
    ov::Shape smallShape = makeInputShape(model, 1, /*useLargeShape=*/false);
    ov::Tensor smallTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), smallShape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            smallTensor,
                            "DynamicNHWDecreasedSize_small");
    ASSERT_EQ(testContext.reqDynamic.get_tensor(model->output()).get_shape(), dynamicNHWOutputShape(smallShape));
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
    SKIP_IF_NOT_TARGET_DEVICE()

    auto model = createModelByName(selectedModelName);

    ov::CompiledModel compiledModel;
    // Compilation shall pass since load of openvino_intel_npu_mlir_runtime is deffered with NPU_CREATE_EXECUTOR=0
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    if (modelStream.str().empty()) {
        FAIL() << "Exported model stream is empty";
    }

    if (hasDynamicBatch(selectedModelName)) {
        ASSERT_TRUE(isElfBlob(modelStream.str())) << "Expected exported model to be an ELF blob";
    } else {
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

#undef SKIP_IF_NOT_TARGET_DEVICE

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

// Every name maps to the same ESPCN_x2 graph (see espcnModelConfigs) built with different dynamic N/H/W bounds
// and layout; the concrete test shapes are derived from those bounds via makeInputShape.
const std::vector<std::string> modelNames = {"ESPCN_x2_DynHW_FHD",
                                             "ESPCN_x2_DynNHW_FHD",
                                             "ESPCN_x2_DynHW_HD",
                                             "ESPCN_x2_DynNHW_Tiny"};

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

const std::vector<std::string> defaultHCModelNames = {"ESPCN_x2_DynHW_HD_NCHW", "ESPCN_x2_DynNHW_HD_NCHW"};
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferWithDefaultHostCompileTests,
                         ::testing::Combine(::testing::ValuesIn(devices),
                                            ::testing::ValuesIn(defaultHostCompileconfigs),
                                            ::testing::ValuesIn(defaultHCModelNames)),
                         ov::test::utils::appendPlatformTypeTestName<InferWithDefaultHostCompileTests>);
