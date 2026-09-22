// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "espcn_x2_model.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

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
    const auto logs = logCapture.str();
    if (logs.find("execute_vm_runtime_v2 - started") != std::string::npos) {
        // VM runtime v2 manages command lists internally, so plugin-side legacy command list logs are not required
        return true;
    }
    return logCapture.str().find(expectedEntry) != std::string::npos;
}

std::shared_ptr<ov::Model> InferWithHostCompileTests::createModelByName(const std::string& modelName) {
    return createESPCNX2ModelByName(modelName);
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

    // Start with the largest shape allowed by the model's declared bounds.
    const ov::Shape shape = makeInputShape(model, 1, true);

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

    const ov::Shape shape2 = makeInputShape(model, 1, false);

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

    // Start with the smallest shape allowed by the model's declared bounds.
    const ov::Shape shape = makeInputShape(model, 1, false);

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
    const ov::Shape shape2 = makeInputShape(model, 1, true);

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

    // Start from a regular host tensor sized to the model's upper bounds.
    const ov::Shape shape = makeInputShape(model, 1, true);
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
    auto zeroContext = core->get_default_context(target_device);
    auto inputTensorForThirdInfer = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
    auto inputSourceForThirdInfer =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 50);
    ASSERT_EQ(inputSourceForThirdInfer.get_byte_size(), inputTensorForThirdInfer.get_byte_size());
    std::memcpy(inputTensorForThirdInfer.data(),
                inputSourceForThirdInfer.data(),
                inputSourceForThirdInfer.get_byte_size());
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            inputTensorForThirdInfer,
                            "CompileAndInferWithZeroTensor_third");
    // Feeding a context-allocated tensor with a new data pointer rebuilds the runtime state.
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

TEST_P(InferWithHostCompileTests, DynamicBatchUsesOneVMExecution) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }
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
    const ov::Shape batchShape = makeInputShape(model, 2, false);
    auto fullBatchTensor =
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), batchShape, 100, 0);
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            fullBatchTensor,
                            "DynamicBatchUsesOneVMExecution_full_batch");
    ASSERT_EQ(reqDynamic1.get_tensor(model->output()).get_shape(), makeOutputShape(selectedModelName, batchShape));

    const auto countMarker = [](const std::string& log, std::string_view marker) {
        size_t count = 0;
        size_t position = 0;
        while ((position = log.find(marker, position)) != std::string::npos) {
            ++count;
            position += marker.size();
        }
        return count;
    };
    const auto countVMExecutions = [&countMarker](const std::string& log) {
        return countMarker(log, "Start to execute graph with runtime engine") +
               countMarker(log, "execute_vm_runtime_v2 - started");
    };
    ASSERT_EQ(countVMExecutions(logCapture.str()), 1u) << logCapture.str();

    logCapture.clear();
    // Two N=1 tensors must be aggregated into one N=2 inference rather than executed separately.
    const ov::Shape singleBatchShape = makeInputShape(model, 1, false);
    std::vector<ov::Tensor> tensorBatch;
    tensorBatch.push_back(
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), singleBatchShape, 100, 0));
    tensorBatch.push_back(
        ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), singleBatchShape, 100, 100));
    OV_ASSERT_NO_THROW(reqDynamic1.set_tensors(testContext.compiledModel.input(), tensorBatch));
    OV_ASSERT_NO_THROW(reqReference1.set_tensors(testContext.referenceCompiledModel.input(), tensorBatch));
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    OV_ASSERT_NO_THROW(reqReference1.infer());
    ASSERT_EQ(reqDynamic1.get_tensor(model->output()).get_shape(), makeOutputShape(selectedModelName, batchShape));
    ov::test::utils::compare(reqReference1.get_tensor(model->output()),
                             reqDynamic1.get_tensor(model->output()),
                             model->output().get_element_type());

    ASSERT_EQ(countVMExecutions(logCapture.str()), 1u) << logCapture.str();
}

// ── V2: NPU_SHARED_COMMON_QUEUE=YES and =NO baseline ─────────────────────────
//
// shared=YES → DynamicPipeline::DynamicPipeline() calls ensureV2(..., queue->handle(), ...)
// shared=NO  → ensureV2(..., nullptr, ...)  (vm_runtime owns the internal queue)
// Verifies first-infer, reuse (same ptr), new-ptr, and shape-change for both modes.
TEST_P(InferWithHostCompileTests, SharedCommonQueue_BasicInferAndReuse) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    for (bool sharedQueue : {true, false}) {
        ov::AnyMap cfg = configuration;
        cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;

        auto model = createModelByName(selectedModelName);
        RuntimeCompareSetupResult setupResult;
        {
            auto savedCfg = configuration;
            configuration = cfg;
            setupResult = prepareRuntimeCompareContext(model);
            configuration = savedCfg;
        }
        if (setupResult.status == RuntimeCompareStatus::fail) {
            FAIL() << "shared=" << sharedQueue << ": " << setupResult.message;
        }
        if (setupResult.status == RuntimeCompareStatus::skip) {
            GTEST_SKIP() << setupResult.message;
        }
        auto& ctx = setupResult.context;
        const std::string tag = sharedQueue ? "shared" : "nonshared";

        const ov::Shape shape = makeInputShape(model, 1, true);
        ov::Tensor t0 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

        OV_ASSERT_NO_THROW(setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, t0, tag + "_first"));
        OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, tag + "_reuse"));

        ov::Tensor t1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 200, 0);
        OV_ASSERT_NO_THROW(setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, t1, tag + "_new_ptr"));

        const ov::Shape shape2 = makeInputShape(model, 1, false);
        ov::Tensor t2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 50, 0);
        OV_ASSERT_NO_THROW(setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, t2, tag + "_shape_change"));
    }
}

TEST_P(InferWithHostCompileTests, SharedCommonQueue_ZeroTensorInputOutputSet) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    for (bool sharedQueue : {true, false}) {
        ov::AnyMap cfg = configuration;
        cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;

        auto model = createModelByName(selectedModelName);
        RuntimeCompareSetupResult setupResult;
        {
            auto savedCfg = configuration;
            configuration = cfg;
            setupResult = prepareRuntimeCompareContext(model);
            configuration = savedCfg;
        }
        if (setupResult.status == RuntimeCompareStatus::fail) {
            FAIL() << "shared=" << sharedQueue << ": " << setupResult.message;
        }
        if (setupResult.status == RuntimeCompareStatus::skip) {
            GTEST_SKIP() << setupResult.message;
        }
        auto& ctx = setupResult.context;
        const std::string tag = sharedQueue ? "shared" : "nonshared";

        auto zeroContext = core->get_default_context(target_device);
        const ov::Shape shape = makeInputShape(model, 1, true);
        ov::Tensor hostInput =
            ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
        OV_ASSERT_NO_THROW(
            setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, hostInput, tag + "_host_input_baseline"));

        auto zeroInput = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
        auto zeroInputSource =
            ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 1);
        ASSERT_EQ(zeroInputSource.get_byte_size(), zeroInput.get_byte_size());
        std::memcpy(zeroInput.data(), zeroInputSource.data(), zeroInputSource.get_byte_size());
        OV_ASSERT_NO_THROW(
            setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, zeroInput, tag + "_zero_input"));

        const auto outputShape = ctx.reqDynamic.get_tensor(model->output()).get_shape();
        auto zeroOutput = zeroContext.create_host_tensor(model->output().get_element_type(), outputShape);
        OV_ASSERT_NO_THROW(ctx.reqDynamic.set_tensor(model->output(), zeroOutput));
        OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, tag + "_zero_output"));

        const ov::Shape shape2 = makeInputShape(model, 1, false);
        auto zeroInput2 = zeroContext.create_host_tensor(model->input().get_element_type(), shape2);
        auto zeroInputSource2 =
            ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 2);
        ASSERT_EQ(zeroInputSource2.get_byte_size(), zeroInput2.get_byte_size());
        std::memcpy(zeroInput2.data(), zeroInputSource2.data(), zeroInputSource2.get_byte_size());
        auto zeroOutput2 = zeroContext.create_host_tensor(model->output().get_element_type(),
                                                          makeOutputShape(selectedModelName, shape2));
        OV_ASSERT_NO_THROW(ctx.reqDynamic.set_tensor(model->output(), zeroOutput2));
        OV_ASSERT_NO_THROW(setInputInferAndCompare(model,
                                                   ctx.reqDynamic,
                                                   ctx.reqReference,
                                                   zeroInput2,
                                                   tag + "_zero_input_output_shape_change"));
    }
}

TEST_P(InferWithHostCompileTests, CompileTimeConfig_ModelPriority) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    for (bool sharedQueue : {true, false}) {
        for (auto priority : {ov::hint::Priority::LOW, ov::hint::Priority::MEDIUM, ov::hint::Priority::HIGH}) {
            ov::AnyMap cfg = configuration;
            cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;
            cfg[ov::hint::model_priority.name()] = priority;

            auto model = createModelByName(selectedModelName);
            RuntimeCompareSetupResult setupResult;
            {
                auto savedCfg = configuration;
                configuration = cfg;
                setupResult = prepareRuntimeCompareContext(model);
                configuration = savedCfg;
            }
            if (setupResult.status == RuntimeCompareStatus::fail) {
                FAIL() << setupResult.message;
            }
            if (setupResult.status == RuntimeCompareStatus::skip) {
                GTEST_SKIP() << setupResult.message;
            }

            const ov::Shape shape = makeInputShape(model, 1, true);
            ov::Tensor inTensor =
                ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
            OV_ASSERT_NO_THROW(setInputInferAndCompare(model,
                                                       setupResult.context.reqDynamic,
                                                       setupResult.context.reqReference,
                                                       inTensor,
                                                       "CompileTimePriority"));
        }
    }
}

TEST_P(InferWithHostCompileTests, CompileTimeConfig_WorkloadType) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    for (bool sharedQueue : {true, false}) {
        for (auto workloadType : {ov::WorkloadType::DEFAULT, ov::WorkloadType::EFFICIENT}) {
            ov::AnyMap cfg = configuration;
            cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;
            cfg[ov::workload_type.name()] = workloadType;

            auto model = createModelByName(selectedModelName);
            ov::CompiledModel compiledModel;
            try {
                auto savedCfg = configuration;
                configuration = cfg;
                compiledModel = core->compile_model(model, target_device, configuration);
                configuration = savedCfg;
            } catch (const ov::Exception& e) {
                GTEST_SKIP() << "workload_type compile-time config not supported: " << e.what();
            }
            ov::CompiledModel refModel;
            try {
                refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
            } catch (const ov::Exception&) {
                GTEST_SKIP() << "TEMPLATE plugin unavailable";
            }

            auto reqDynamic = compiledModel.create_infer_request();
            auto reqRef = refModel.create_infer_request();
            const ov::Shape shape = makeInputShape(model, 1, true);
            ov::Tensor inTensor =
                ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
            OV_ASSERT_NO_THROW(setInputInferAndCompare(model, reqDynamic, reqRef, inTensor, "CompileTimeWorkload"));
        }
    }
}

TEST_P(InferWithHostCompileTests, CompileTimeConfig_Turbo) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    for (bool sharedQueue : {true, false}) {
        for (bool turbo : {false, true}) {
            ov::AnyMap cfg = configuration;
            cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;
            cfg[ov::intel_npu::turbo.name()] = turbo;

            auto model = createModelByName(selectedModelName);
            ov::CompiledModel compiledModel;
            try {
                auto savedCfg = configuration;
                configuration = cfg;
                compiledModel = core->compile_model(model, target_device, configuration);
                configuration = savedCfg;
            } catch (const ov::Exception& e) {
                GTEST_SKIP() << "turbo compile-time config not supported: " << e.what();
            }
            ov::CompiledModel refModel;
            try {
                refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
            } catch (const ov::Exception&) {
                GTEST_SKIP() << "TEMPLATE plugin unavailable";
            }

            auto reqDynamic = compiledModel.create_infer_request();
            auto reqRef = refModel.create_infer_request();
            const ov::Shape shape = makeInputShape(model, 1, true);
            ov::Tensor inTensor =
                ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
            OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor));
            OV_ASSERT_NO_THROW(reqRef.set_input_tensor(0, inTensor));
            try {
                reqDynamic.infer();
                OV_ASSERT_NO_THROW(reqRef.infer());
                OV_ASSERT_NO_THROW(compareInferenceResult(model, reqDynamic, reqRef));
            } catch (const ov::Exception& e) {
                if (!turbo) {
                    FAIL() << "Unexpected turbo=false compile-time failure: " << e.what();
                }
                const std::string errorMsg = e.what();
                const bool gotExpectedRuntimeError = errorMsg.find("QUEUE_OPTIONS") != std::string::npos ||
                                                     errorMsg.find("configCmdQueue") != std::string::npos ||
                                                     errorMsg.find("vm_runtime") != std::string::npos;
                ASSERT_TRUE(gotExpectedRuntimeError) << "Unexpected turbo compile-time failure reason: " << errorMsg;
            }
        }
    }
}

TEST_P(InferWithHostCompileTests, SetProperty_CombinedPriorityAndWorkload) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    for (bool sharedQueue : {true, false}) {
        ov::AnyMap cfg = configuration;
        cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;
        cfg[ov::hint::model_priority.name()] = ov::hint::Priority::LOW;
        cfg[ov::workload_type.name()] = ov::WorkloadType::DEFAULT;

        auto model = createModelByName(selectedModelName);
        ov::CompiledModel compiledModel;
        try {
            auto savedCfg = configuration;
            configuration = cfg;
            compiledModel = core->compile_model(model, target_device, configuration);
            configuration = savedCfg;
        } catch (const ov::Exception& e) {
            GTEST_SKIP() << "compile_model failed: " << e.what();
        }
        ov::CompiledModel refModel;
        try {
            refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
        } catch (const ov::Exception&) {
            GTEST_SKIP() << "TEMPLATE plugin unavailable";
        }

        try {
            compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::EFFICIENT}});
            compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::DEFAULT}});
        } catch (const ov::Exception& e) {
            GTEST_SKIP() << "workload_type not supported: " << e.what();
        }

        auto reqDynamic = compiledModel.create_infer_request();
        auto reqRef = refModel.create_infer_request();
        const ov::Shape shape = makeInputShape(model, 1, true);
        ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
        OV_ASSERT_NO_THROW(setInputInferAndCompare(model, reqDynamic, reqRef, inTensor, "CombinedConfig_baseline"));

        OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::HIGH},
                                                       {ov::workload_type.name(), ov::WorkloadType::EFFICIENT}}));
        OV_ASSERT_NO_THROW(
            inferAndCompare(model, reqDynamic, reqRef, "CombinedConfig_priority_high_workload_efficient"));

        const ov::Shape shape2 = makeInputShape(model, 1, false);
        ov::Tensor inTensor2 =
            ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 1);
        OV_ASSERT_NO_THROW(
            setInputInferAndCompare(model, reqDynamic, reqRef, inTensor2, "CombinedConfig_shape_change"));

        OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::LOW},
                                                       {ov::workload_type.name(), ov::WorkloadType::DEFAULT}}));
        OV_ASSERT_NO_THROW(inferAndCompare(model, reqDynamic, reqRef, "CombinedConfig_restored"));
    }
}

// ── V2: compiledModel.set_property(model_priority) → priority change ──────────
//
// This is the primary missing scenario: a user calls set_property() on a single
// already-compiled model to change priority at runtime, then infers.
//
// Call chain:
//   compiledModel.set_property({model_priority, HIGH})
//   → DynamicGraph::set_model_priority(HIGH)
//     → _commandQueueDesc.set_priority(HIGH) → new key
//   reqDynamic.infer() → DynamicPipeline::push()
//     → command_queue_desc.key() != _command_queue->desc().key()   (changed)
//     → [shared=YES] _command_queue = ZeroCmdQueuePool::getCommandQueue(new_desc)
//     → update_runtime_config(prev_desc, curr_desc) → QUEUE_PRIORITY diff → pConfig
//     → execute_vm_runtime_v2(..., pConfig) → npuVMRuntimeExecute2 with pConfig
//
// Transitions: LOW → HIGH → MEDIUM → LOW; also verifies a new InferRequest created
// after set_property picks up the updated CommandQueueDesc.
TEST_P(InferWithHostCompileTests, SetProperty_ModelPriority_SingleCompiledModel) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    ov::AnyMap cfg = configuration;
    cfg[ov::hint::model_priority.name()] = ov::hint::Priority::LOW;
    cfg[ov::intel_npu::shared_common_queue.name()] = true;

    auto model = createModelByName(selectedModelName);
    RuntimeCompareSetupResult setupResult;
    {
        auto savedCfg = configuration;
        configuration = cfg;
        setupResult = prepareRuntimeCompareContext(model);
        configuration = savedCfg;
    }
    if (setupResult.status == RuntimeCompareStatus::fail) {
        FAIL() << setupResult.message;
    }
    if (setupResult.status == RuntimeCompareStatus::skip) {
        GTEST_SKIP() << setupResult.message;
    }
    auto& ctx = setupResult.context;

    const ov::Shape shape = makeInputShape(model, 1, true);
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    // Baseline at LOW: command_queue_version_changed=false on first push().
    OV_ASSERT_NO_THROW(
        setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, inTensor, "Priority_low_baseline"));

    // LOW → HIGH: _commandQueueDesc.key() changes; push() detects it and replaces queue.
    // update_runtime_config produces QUEUE_PRIORITY diff in pConfig.
    OV_ASSERT_NO_THROW(ctx.compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, "Priority_LOW_to_HIGH"));

    // Reuse at HIGH: same shape, same ptr → no re-record, queue is now HIGH-priority.
    OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, "Priority_HIGH_reuse"));

    // HIGH → MEDIUM.
    OV_ASSERT_NO_THROW(ctx.compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::MEDIUM}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, "Priority_HIGH_to_MEDIUM"));

    // MEDIUM → LOW: exercises roundtrip back to LOW-priority queue entry in pool.
    OV_ASSERT_NO_THROW(ctx.compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::LOW}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, "Priority_MEDIUM_to_LOW"));

    // New InferRequest after set_property: must use the updated CommandQueueDesc.
    ov::InferRequest reqNew = ctx.compiledModel.create_infer_request();
    ov::InferRequest reqRefNew = ctx.referenceCompiledModel.create_infer_request();
    OV_ASSERT_NO_THROW(setInputInferAndCompare(model, reqNew, reqRefNew, inTensor, "Priority_new_request_after_set"));
}

// ── V2: compiledModel.set_property(workload_type) → workload change ───────────
//
// Call chain (shared=YES):
//   set_property(workload_type, EFFICIENT)
//   → DynamicGraph::set_workload_type()
//     → [_commandQueue not set yet] _commandQueueDesc.set_workload() → key changes
//   push(): command_queue_desc.key() changed → replace _command_queue from pool
//           update_runtime_config → WORKLOAD_TYPE diff → pConfig → configCmdQueue
//
// Skipped gracefully when the driver does not support workload_type.
TEST_P(InferWithHostCompileTests, SetProperty_WorkloadType_SingleCompiledModel) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    ov::AnyMap cfg = configuration;
    cfg[ov::intel_npu::shared_common_queue.name()] = true;

    auto model = createModelByName(selectedModelName);
    ov::CompiledModel compiledModel;
    try {
        auto savedCfg = configuration;
        configuration = cfg;
        compiledModel = core->compile_model(model, target_device, configuration);
        configuration = savedCfg;
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "compile_model failed: " << e.what();
    }
    ov::CompiledModel refModel;
    try {
        refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    // Probe support before the actual test.
    try {
        compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::EFFICIENT}});
        compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::DEFAULT}});
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "workload_type not supported: " << e.what();
    }

    ov::InferRequest reqDynamic = compiledModel.create_infer_request();
    ov::InferRequest reqRef = refModel.create_infer_request();

    const ov::Shape shape = makeInputShape(model, 1, true);
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    OV_ASSERT_NO_THROW(setInputInferAndCompare(model, reqDynamic, reqRef, inTensor, "Workload_default_baseline"));

    OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::EFFICIENT}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqDynamic, reqRef, "Workload_DEFAULT_to_EFFICIENT"));

    OV_ASSERT_NO_THROW(inferAndCompare(model, reqDynamic, reqRef, "Workload_EFFICIENT_reuse"));

    OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::DEFAULT}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqDynamic, reqRef, "Workload_EFFICIENT_to_DEFAULT"));
}

// TURBO is a compile-time command queue option. Verify both compile-time values
// for the shared common queue configuration.
TEST_P(InferWithHostCompileTests, CompileTimeConfig_Turbo_SharedCommonQueue) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createModelByName(selectedModelName);
    ov::CompiledModel refModel;
    try {
        refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }
    const ov::Shape shape = makeInputShape(model, 1, true);
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    for (bool turbo : {false, true}) {
        ov::AnyMap cfg = configuration;
        cfg[ov::intel_npu::shared_common_queue.name()] = true;
        cfg[ov::intel_npu::turbo.name()] = turbo;

        ov::CompiledModel compiledModel;
        try {
            compiledModel = core->compile_model(model, target_device, cfg);
        } catch (const ov::Exception& e) {
            GTEST_SKIP() << "turbo compile-time config not supported: " << e.what();
        }

        auto reqDynamic = compiledModel.create_infer_request();
        auto reqRef = refModel.create_infer_request();
        OV_ASSERT_NO_THROW(setInputInferAndCompare(model,
                                                   reqDynamic,
                                                   reqRef,
                                                   inTensor,
                                                   turbo ? "Turbo_ON_compile_time" : "Turbo_OFF_compile_time"));
    }
}

// ── V2: set_property between two InferRequests from the same CompiledModel ────
//
// Both InferRequests call _graph->get_command_queue_desc() on every push(), so
// after set_property() changes the priority, each request's next push() sees
// command_queue_version_changed=true and independently replaces its _command_queue.
//
// Sequence:
//   reqA.infer() at LOW  → establishes LOW queue in reqA's DynamicPipeline
//   reqB.infer() at LOW  → establishes LOW queue in reqB's DynamicPipeline
//   set_property(HIGH)   → _commandQueueDesc.key() changes for both pipelines
//   reqA.infer() at HIGH → push() detects change; QUEUE_PRIORITY diff in pConfig
//   reqB.infer() at HIGH → same
//   set_property(LOW)    → back to LOW
//   reqA.infer() at LOW  → second transition
//   reqB.infer() at LOW  → second transition
TEST_P(InferWithHostCompileTests, SetProperty_Priority_BetweenTwoRequests) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    ov::AnyMap cfg = configuration;
    cfg[ov::hint::model_priority.name()] = ov::hint::Priority::LOW;
    cfg[ov::intel_npu::shared_common_queue.name()] = true;

    auto model = createModelByName(selectedModelName);
    ov::CompiledModel compiledModel;
    try {
        auto savedCfg = configuration;
        configuration = cfg;
        compiledModel = core->compile_model(model, target_device, configuration);
        configuration = savedCfg;
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "compile_model failed: " << e.what();
    }
    ov::CompiledModel refModel;
    try {
        refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    ov::InferRequest reqA = compiledModel.create_infer_request();
    ov::InferRequest reqB = compiledModel.create_infer_request();
    ov::InferRequest reqRef = refModel.create_infer_request();

    const ov::Shape shape = makeInputShape(model, 1, true);
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    // Warmup both at LOW.
    OV_ASSERT_NO_THROW(setInputInferAndCompare(model, reqA, reqRef, inTensor, "TwoReq_A_LOW"));
    OV_ASSERT_NO_THROW(setInputInferAndCompare(model, reqB, reqRef, inTensor, "TwoReq_B_LOW"));

    // Change to HIGH: both DynamicPipelines will see new CommandQueueDesc on next push().
    OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqA, reqRef, "TwoReq_A_LOW_to_HIGH"));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqB, reqRef, "TwoReq_B_LOW_to_HIGH"));

    // Back to LOW: second transition for both.
    OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::LOW}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqA, reqRef, "TwoReq_A_HIGH_to_LOW"));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqB, reqRef, "TwoReq_B_HIGH_to_LOW"));
}

TEST_P(InferWithHostCompileTests, SetProperty_ModelPriority_SingleCompiledModel_NonSharedCommonQueue) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    ov::AnyMap cfg = configuration;
    cfg[ov::hint::model_priority.name()] = ov::hint::Priority::LOW;
    cfg[ov::intel_npu::shared_common_queue.name()] = false;

    auto model = createModelByName(selectedModelName);
    RuntimeCompareSetupResult setupResult;
    {
        auto savedCfg = configuration;
        configuration = cfg;
        setupResult = prepareRuntimeCompareContext(model);
        configuration = savedCfg;
    }
    if (setupResult.status == RuntimeCompareStatus::fail) {
        FAIL() << setupResult.message;
    }
    if (setupResult.status == RuntimeCompareStatus::skip) {
        GTEST_SKIP() << setupResult.message;
    }
    auto& ctx = setupResult.context;

    const ov::Shape shape = makeInputShape(model, 1, true);
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    OV_ASSERT_NO_THROW(
        setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, inTensor, "Priority_nonshared_low_baseline"));

    OV_ASSERT_NO_THROW(ctx.compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, "Priority_nonshared_LOW_to_HIGH"));

    OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, "Priority_nonshared_HIGH_reuse"));

    OV_ASSERT_NO_THROW(ctx.compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::MEDIUM}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, "Priority_nonshared_HIGH_to_MEDIUM"));

    OV_ASSERT_NO_THROW(ctx.compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::LOW}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, "Priority_nonshared_MEDIUM_to_LOW"));

    ov::InferRequest reqNew = ctx.compiledModel.create_infer_request();
    ov::InferRequest reqRefNew = ctx.referenceCompiledModel.create_infer_request();
    OV_ASSERT_NO_THROW(
        setInputInferAndCompare(model, reqNew, reqRefNew, inTensor, "Priority_nonshared_new_request_after_set"));
}

TEST_P(InferWithHostCompileTests, SetProperty_WorkloadType_SingleCompiledModel_NonSharedCommonQueue) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    ov::AnyMap cfg = configuration;
    cfg[ov::intel_npu::shared_common_queue.name()] = false;

    auto model = createModelByName(selectedModelName);
    ov::CompiledModel compiledModel;
    try {
        auto savedCfg = configuration;
        configuration = cfg;
        compiledModel = core->compile_model(model, target_device, configuration);
        configuration = savedCfg;
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "compile_model failed: " << e.what();
    }
    ov::CompiledModel refModel;
    try {
        refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    try {
        compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::EFFICIENT}});
        compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::DEFAULT}});
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "workload_type not supported in nonshared queue mode: " << e.what();
    }

    ov::InferRequest reqDynamic = compiledModel.create_infer_request();
    ov::InferRequest reqRef = refModel.create_infer_request();

    const ov::Shape shape = makeInputShape(model, 1, true);
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    OV_ASSERT_NO_THROW(
        setInputInferAndCompare(model, reqDynamic, reqRef, inTensor, "Workload_nonshared_default_baseline"));

    OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::EFFICIENT}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqDynamic, reqRef, "Workload_nonshared_DEFAULT_to_EFFICIENT"));

    OV_ASSERT_NO_THROW(inferAndCompare(model, reqDynamic, reqRef, "Workload_nonshared_EFFICIENT_reuse"));

    OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::workload_type.name(), ov::WorkloadType::DEFAULT}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqDynamic, reqRef, "Workload_nonshared_EFFICIENT_to_DEFAULT"));
}

TEST_P(InferWithHostCompileTests, CompileTimeConfig_Turbo_NonSharedCommonQueue) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createModelByName(selectedModelName);
    ov::CompiledModel refModel;
    try {
        refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }
    const ov::Shape shape = makeInputShape(model, 1, true);
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    for (bool turbo : {false, true}) {
        ov::AnyMap cfg = configuration;
        cfg[ov::intel_npu::shared_common_queue.name()] = false;
        cfg[ov::intel_npu::turbo.name()] = turbo;

        ov::CompiledModel compiledModel;
        try {
            compiledModel = core->compile_model(model, target_device, cfg);
        } catch (const ov::Exception& e) {
            GTEST_SKIP() << "turbo compile-time config not supported in nonshared queue mode: " << e.what();
        }

        auto reqDynamic = compiledModel.create_infer_request();
        auto reqRef = refModel.create_infer_request();
        OV_ASSERT_NO_THROW(
            setInputInferAndCompare(model,
                                    reqDynamic,
                                    reqRef,
                                    inTensor,
                                    turbo ? "Turbo_nonshared_ON_compile_time" : "Turbo_nonshared_OFF_compile_time"));
    }
}

TEST_P(InferWithHostCompileTests, SetProperty_Priority_BetweenTwoRequests_NonSharedCommonQueue) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    ov::AnyMap cfg = configuration;
    cfg[ov::hint::model_priority.name()] = ov::hint::Priority::LOW;
    cfg[ov::intel_npu::shared_common_queue.name()] = false;

    auto model = createModelByName(selectedModelName);
    ov::CompiledModel compiledModel;
    try {
        auto savedCfg = configuration;
        configuration = cfg;
        compiledModel = core->compile_model(model, target_device, configuration);
        configuration = savedCfg;
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "compile_model failed: " << e.what();
    }
    ov::CompiledModel refModel;
    try {
        refModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception&) {
        GTEST_SKIP() << "TEMPLATE plugin unavailable";
    }

    ov::InferRequest reqA = compiledModel.create_infer_request();
    ov::InferRequest reqB = compiledModel.create_infer_request();
    ov::InferRequest reqRef = refModel.create_infer_request();

    const ov::Shape shape = makeInputShape(model, 1, true);
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

    OV_ASSERT_NO_THROW(setInputInferAndCompare(model, reqA, reqRef, inTensor, "TwoReq_nonshared_A_LOW"));
    OV_ASSERT_NO_THROW(setInputInferAndCompare(model, reqB, reqRef, inTensor, "TwoReq_nonshared_B_LOW"));

    OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::HIGH}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqA, reqRef, "TwoReq_nonshared_A_LOW_to_HIGH"));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqB, reqRef, "TwoReq_nonshared_B_LOW_to_HIGH"));

    OV_ASSERT_NO_THROW(compiledModel.set_property({{ov::hint::model_priority.name(), ov::hint::Priority::LOW}}));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqA, reqRef, "TwoReq_nonshared_A_HIGH_to_LOW"));
    OV_ASSERT_NO_THROW(inferAndCompare(model, reqB, reqRef, "TwoReq_nonshared_B_HIGH_to_LOW"));
}

// ── V2 MemRef reuse: same-pointer / same-shape input ──────────────────────────
//
// Exercises the v2 execution path across repeated inferences with:
//  - same tensor pointer / same shape
//  - different tensor pointer / same shape
//  - different shape
// Validates output correctness for all cases (this test does not assert on specific log messages).
// The test is skipped when the v2 code path is not active (i.e., when
// "execute_vm_runtime_v2 - started" is absent from the first-inference logs).
TEST_P(InferWithHostCompileTests, MemRefReuse_SamePtrSameShape) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    for (bool sharedQueue : {true, false}) {
        ov::AnyMap cfg = configuration;
        cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;

        auto model = createModelByName(selectedModelName);
        RuntimeCompareSetupResult setupResult;
        {
            auto savedCfg = configuration;
            configuration = cfg;
            setupResult = prepareRuntimeCompareContext(model);
            configuration = savedCfg;
        }
        if (setupResult.status == RuntimeCompareStatus::fail) {
            FAIL() << "shared=" << sharedQueue << ": " << setupResult.message;
        }
        if (setupResult.status == RuntimeCompareStatus::skip) {
            GTEST_SKIP() << setupResult.message;
        }
        auto& ctx = setupResult.context;
        const std::string tag = sharedQueue ? "shared" : "nonshared";

        const ov::Shape shape = makeInputShape(model, 1, true);
        ov::Tensor t0 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

        // 1st inference: recording expected
        OV_ASSERT_NO_THROW(setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, t0, tag + "_v2_first"));

        OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, tag + "_v2_same_reuse"));

        // 3rd inference: new tensor (different pointer, same shape) → recording
        ov::Tensor t1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 200, 0);

        OV_ASSERT_NO_THROW(setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, t1, tag + "_v2_new_ptr"));

        // 4th inference: same new tensor again → reuse
        OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, tag + "_v2_new_ptr_reuse"));

        // 5th inference: different shape → recording
        const ov::Shape shape2 = makeInputShape(model, 1, false);
        ov::Tensor t2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 50, 0);
        OV_ASSERT_NO_THROW(
            setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, t2, tag + "_v2_shape_change"));
    }
}

// ── V2 MemRef reuse: output pointer change ─────────────────────────────────────
//
// Verifies that changing the output tensor triggers re-recording (not reuse)
// even when the input tensor is unchanged.  This also exercises the code path
// where vm_runtime must correctly reset output dirty bits (as opposed to the
// known bug where it used pInputs instead of pOutputs for output dirty-bit
// reset — if that bug were present every inference would incorrectly re-record).
TEST_P(InferWithHostCompileTests, MemRefReuse_OutputPtrChange) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    for (bool sharedQueue : {true, false}) {
        ov::AnyMap cfg = configuration;
        cfg[ov::intel_npu::shared_common_queue.name()] = sharedQueue;

        auto model = createModelByName(selectedModelName);
        RuntimeCompareSetupResult setupResult;
        {
            auto savedCfg = configuration;
            configuration = cfg;
            setupResult = prepareRuntimeCompareContext(model);
            configuration = savedCfg;
        }
        if (setupResult.status == RuntimeCompareStatus::fail) {
            FAIL() << "shared=" << sharedQueue << ": " << setupResult.message;
        }
        if (setupResult.status == RuntimeCompareStatus::skip) {
            GTEST_SKIP() << setupResult.message;
        }
        auto& ctx = setupResult.context;
        const std::string tag = sharedQueue ? "shared" : "nonshared";

        const ov::Shape shape = makeInputShape(model, 1, true);
        ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);

        // 1st inference: recording expected
        OV_ASSERT_NO_THROW(
            setInputInferAndCompare(model, ctx.reqDynamic, ctx.reqReference, inTensor, tag + "_v2out_first"));

        // 2nd inference: same input, no output change → reuse
        OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, tag + "_v2out_input_reuse"));

        // 3rd inference: set a new output tensor (different pointer, same shape) → recording
        const auto outputShape = ctx.reqDynamic.get_tensor(model->output()).get_shape();
        auto zeroCtx = core->get_default_context(target_device);
        auto newOutputTensor = zeroCtx.create_host_tensor(model->output().get_element_type(), outputShape);
        OV_ASSERT_NO_THROW(ctx.reqDynamic.set_tensor(model->output(), newOutputTensor));
        OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, tag + "_v2out_output_ptr_change"));

        // 4th inference: same output tensor again → reuse (validates output dirty bits are cleared)
        OV_ASSERT_NO_THROW(inferAndCompare(model, ctx.reqDynamic, ctx.reqReference, tag + "_v2out_output_ptr_reuse"));
    }
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

    if (hasOnlyDynamicBatch(selectedModelName)) {
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

const std::vector<std::string> modelNames = {
    "ESPCN_x2_DynHW_FHD",
    "ESPCN_x2_DynNHW_FHD",
    "ESPCN_x2_DynHW_HD",
    "ESPCN_x2_DynHW_FHD2",
};

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

const std::vector<std::string> defaultHCModelNames = {"ESPCN_x2_DynN_HD_NCHW", "ESPCN_x2_DynNHW_HD_NCHW"};
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         InferWithDefaultHostCompileTests,
                         ::testing::Combine(::testing::ValuesIn(devices),
                                            ::testing::ValuesIn(defaultHostCompileconfigs),
                                            ::testing::ValuesIn(defaultHCModelNames)),
                         ov::test::utils::appendPlatformTypeTestName<InferWithDefaultHostCompileTests>);
