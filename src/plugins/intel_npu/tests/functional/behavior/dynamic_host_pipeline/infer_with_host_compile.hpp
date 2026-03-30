// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <common_test_utils/ov_tensor_utils.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "intel_npu/utils/zero/zero_init.hpp"
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
    // Use a reduced model when CHECK_SIMPLE_MODEL=1 to simplify local debugging.
    const char* check_simple_model = std::getenv("CHECK_SIMPLE_MODEL");
    if (check_simple_model && std::string(check_simple_model) == "1") {
        auto input =
            std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape{1, 16, 720, ov::Dimension(10, 1280)});
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

        return std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "MaxPool");
    }

    auto input =
        std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 16, 720, ov::Dimension(10, 1280)});
    input->set_friendly_name("input1");

    auto maxpool = std::make_shared<ov::op::v1::MaxPool>(input,
                                                         Strides{1, 1},
                                                         Shape{1, 1},
                                                         Shape{1, 1},
                                                         Shape{3, 3},
                                                         op::RoundingType::FLOOR,
                                                         op::PadType::EXPLICIT);
    maxpool->set_friendly_name("MaxPool_2");

    auto scale = ov::opset6::Constant::create(element::f32, Shape{}, {2.0f});
    scale->set_friendly_name("scale_const");

    auto mul = std::make_shared<ov::op::v1::Multiply>(maxpool, scale);
    mul->set_friendly_name("Mul_1");

    auto result = std::make_shared<ov::op::v0::Result>(mul);
    result->set_friendly_name("output");

    return std::make_shared<Model>(ResultVector{result}, ParameterVector{input}, "MaxPool");
}

using InferWithHostCompileParams = std::tuple<std::string,  // Device name
                                              ov::AnyMap    // Config
                                              >;

// These tests are required by the NPU plugin to verify the support of dynamic shape during
// compilation and inference on different NPU drivers
class InferWithHostCompileTests : public testing::WithParamInterface<InferWithHostCompileParams>,
                                  public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferWithHostCompileParams> obj) {
        std::string target_device;
        ov::AnyMap configuration;
        std::tie(target_device, configuration) = obj.param;
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
        return result.str();
    }

    void SetUp() {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        std::tie(target_device, configuration) = this->GetParam();

        std::vector<std::string> deviceNames =
            core->get_property("NPU", ov::available_devices.name()).as<std::vector<std::string>>();
        for (auto name : deviceNames) {
            if (target_device.find(name) != std::string::npos) {
                isTargetDevice = true;
                break;
            }
        }

        APIBaseTest::SetUp();
    }

    void TearDown() {
        core->set_property("NPU", ov::log::level(ov::log::Level::ERR));
    }

    bool isLLVMFormat(std::stringstream& modelStream) {
        auto pos = modelStream.tellg();
        if (pos == std::streampos(-1)) {
            return false;
        }

        modelStream.seekg(0, std::ios::beg);

        // Host-compiled blobs are expected to expose the "llvm" marker near the beginning.
        std::string region(20, '\0');
        modelStream.read(&region[0], 20);
        region.resize(modelStream.gcount());

        modelStream.clear();
        modelStream.seekg(pos);

        return region.find("llvm") != std::string::npos;
    }

    static void dumpTensor(const ov::Tensor& tensor, std::string name);

    static void compareAndDumpInferenceResult(const std::shared_ptr<ov::Model>& model,
                                              ov::InferRequest& reqDynamic,
                                              ov::InferRequest& reqReference,
                                              const std::string& dumpPrefix);

    static void inferAndCompare(const std::shared_ptr<ov::Model>& model,
                                ov::InferRequest& reqDynamic,
                                ov::InferRequest& reqReference,
                                const std::string& dumpPrefix);

    static void setInputInferAndCompare(const std::shared_ptr<ov::Model>& model,
                                        ov::InferRequest& reqDynamic,
                                        ov::InferRequest& reqReference,
                                        const ov::Tensor& inputTensor,
                                        const std::string& dumpPrefix);

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    bool isTargetDevice = false;
};

void InferWithHostCompileTests::dumpTensor(const ov::Tensor& tensor, std::string name) {
    std::cout << "Tensor name: " << name << ", shape: " << tensor.get_shape()
              << ", element type: " << tensor.get_element_type() << std::endl;
    const float* data = tensor.data<float>();
    size_t count = ov::shape_size(tensor.get_shape());
    count = count > 50 ? 50 : count;
    for (size_t i = 0; i < count; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // Add a random suffix to avoid collisions when tests run in parallel.
    std::cout << "Dump tensor to file for debugging, tensor name: " << name << std::endl;
    std::string fileName = name + "_" + std::to_string(std::rand()) + ".txt";
    std::ofstream outFile(fileName);
    if (outFile.is_open()) {
        outFile << "Tensor name: " << name << ", shape: " << tensor.get_shape()
                << ", element type: " << tensor.get_element_type() << std::endl;
        size_t totalCount = ov::shape_size(tensor.get_shape());
        for (size_t i = 0; i < totalCount; i++) {
            outFile << data[i] << " ";
        }
        outFile << std::endl;
    }
    std::cout << "Tensor data dumped to file: " << fileName << std::endl;
}

void InferWithHostCompileTests::compareAndDumpInferenceResult(const std::shared_ptr<ov::Model>& model,
                                                              ov::InferRequest& reqDynamic,
                                                              ov::InferRequest& reqReference,
                                                              const std::string& dumpPrefix) {
    const auto inputTensor = reqDynamic.get_input_tensor(0);
    const auto npuOutputTensor = reqDynamic.get_tensor(model->output());
    const auto referenceOutputTensor = reqReference.get_tensor(model->output());
    // Dump tensors only when explicitly requested because file output can distort timing-sensitive behavior.
    const char* dumpTensorEnv = std::getenv("DUMP_TENSOR_FOR_DYNAMIC");
    if (dumpTensorEnv && std::string(dumpTensorEnv) == "1") {
        std::cout << "Dump tensor for dynamic shape test since DUMP_TENSOR_FOR_DYNAMIC is set to 1" << std::endl;
        std::cout << "Dump input tensor for " << dumpPrefix << ", shape: " << inputTensor.get_shape() << std::endl;
        dumpTensor(inputTensor, dumpPrefix + "_input");
        std::cout << "Dump output tensor from NPU for " << dumpPrefix << ", shape: " << npuOutputTensor.get_shape()
                  << std::endl;
        dumpTensor(npuOutputTensor, dumpPrefix + "_npu_output");
        std::cout << "Dump output tensor from Template plugin for " << dumpPrefix
                  << ", shape: " << referenceOutputTensor.get_shape() << std::endl;
        dumpTensor(referenceOutputTensor, dumpPrefix + "_template_output");
    }

    OV_ASSERT_NO_THROW(
        ov::test::utils::compare(referenceOutputTensor, npuOutputTensor, npuOutputTensor.get_element_type()));
}

void InferWithHostCompileTests::inferAndCompare(const std::shared_ptr<ov::Model>& model,
                                                ov::InferRequest& reqDynamic,
                                                ov::InferRequest& reqReference,
                                                const std::string& dumpPrefix) {
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    OV_ASSERT_NO_THROW(reqReference.infer());
    compareAndDumpInferenceResult(model, reqDynamic, reqReference, dumpPrefix);
}

void InferWithHostCompileTests::setInputInferAndCompare(const std::shared_ptr<ov::Model>& model,
                                                        ov::InferRequest& reqDynamic,
                                                        ov::InferRequest& reqReference,
                                                        const ov::Tensor& inputTensor,
                                                        const std::string& dumpPrefix) {
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inputTensor));
    OV_ASSERT_NO_THROW(reqReference.set_input_tensor(0, inputTensor));
    inferAndCompare(model, reqDynamic, reqReference, dumpPrefix);
}

// Test compilation without executor creation
TEST_P(InferWithHostCompileTests, Compile) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    auto model = createMaxPoolModel();

    ov::CompiledModel compiledModel;
    // Compilation shall pass since load of npu_mlir_runtime is deffered with NPU_CREATE_EXECUTOR=0
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    // With HostCompile, the modelStream shall contain "llvm.func"
    ASSERT_TRUE(isLLVMFormat(modelStream)) << "CompiledStream from HostCompile mode shall has 'llvm.func' inside it";
}

// Test compilation and import compiled blob without executor creation
TEST_P(InferWithHostCompileTests, CompileAndImport) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }
    auto model = createMaxPoolModel();

    ov::CompiledModel compiledModel;
    // Compilation shall pass since load of npu_mlir_runtime is deffered with NPU_CREATE_EXECUTOR=0
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    std::stringstream modelStream;
    OV_ASSERT_NO_THROW(compiledModel.export_model(modelStream));

    // With HostCompile, the modelStream shall contain "llvm.func"
    ASSERT_TRUE(isLLVMFormat(modelStream)) << "CompiledStream from HostCompile mode shall has 'llvm.func' inside it";

    ov::CompiledModel importedModel;
    OV_ASSERT_NO_THROW(core->import_model(modelStream, target_device, configuration));
}

// Compile, infer with a large shape, then shrink the input shape and verify both output correctness and command-list
// reuse behavior.
TEST_P(InferWithHostCompileTests, CompileAndInferWithDecreasedSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;
    ov::CompiledModel referenceCompiledModel;

    // Capture plugin logs so the test can verify command-list reuse decisions.
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback = [&](std::string_view s) {
        customLogger << s << std::endl;
    };
    ov::util::set_log_callback(customLogCallback);
    struct ResetLogCallbackGuard {
        ~ResetLogCallbackGuard() {
            ov::util::reset_log_callback();
        }
    } reset_log_callback_guard;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));

    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
    try {
        referenceCompiledModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "CPU plugin is not available for reference comparison: " << e.what();
    }

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // Host compile can be enabled even when the runtime library is unavailable.
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }
    ov::InferRequest reqReference = referenceCompiledModel.create_infer_request();

    // Start with the largest shape in the dynamic range.
    ov::Shape shape = {1, 16, 720, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor, "CompileAndInferWithDecreasedSize_first");
    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    inferAndCompare(model, reqDynamic, reqReference, "CompileAndInferWithDecreasedSize_second");
    // Reusing the same input should keep the existing command list intact.
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor1, "CompileAndInferWithDecreasedSize_third");
    // A new host tensor with the same shape should still reuse the command list.
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for third "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::Shape shape2 = {1, 16, 720, 720};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor3, "CompileAndInferWithDecreasedSize_fourth");
    // Shrinking the shape should force runtime reconfiguration for the new tensor layout.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << customLogger.str();
}

// Compile, infer with a small shape, then grow the input shape and verify both output correctness and command-list
// reuse behavior.
TEST_P(InferWithHostCompileTests, CompileAndInferWithIncreasedSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;
    ov::CompiledModel referenceCompiledModel;

    // Capture plugin logs so the test can verify command-list reuse decisions.
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback = [&](std::string_view s) {
        customLogger << s << std::endl;
    };
    ov::util::set_log_callback(customLogCallback);
    struct ResetLogCallbackGuard {
        ~ResetLogCallbackGuard() {
            ov::util::reset_log_callback();
        }
    } reset_log_callback_guard;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));

    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
    try {
        referenceCompiledModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "CPU plugin is not available for reference comparison: " << e.what();
    }

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // Host compile can be enabled even when the runtime library is unavailable.
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }
    ov::InferRequest reqReference = referenceCompiledModel.create_infer_request();

    // Start with a smaller valid dynamic shape.
    ov::Shape shape = {1, 16, 720, 720};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor, "CompileAndInferWithIncreasedSize_first");
    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    inferAndCompare(model, reqDynamic, reqReference, "CompileAndInferWithIncreasedSize_second");
    // Reusing the same input should keep the existing command list intact.
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor1, "CompileAndInferWithIncreasedSize_third");
    // A new host tensor with the same shape should still reuse the command list.
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for third "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::Shape shape2 = {1, 16, 720, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor3, "CompileAndInferWithIncreasedSize_fourth");
    // Growing the shape should force runtime reconfiguration for the new tensor layout.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << customLogger.str();
}

// Exercise imported Level Zero tensors and verify both output correctness and command-list pointer updates.
TEST_P(InferWithHostCompileTests, CompileAndInferWithZeroTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;
    ov::CompiledModel referenceCompiledModel;

    // Capture plugin logs so the test can verify command-list reuse decisions.
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback = [&](std::string_view s) {
        customLogger << s << std::endl;
    };
    ov::util::set_log_callback(customLogCallback);
    struct ResetLogCallbackGuard {
        ~ResetLogCallbackGuard() {
            ov::util::reset_log_callback();
        }
    } reset_log_callback_guard;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));

    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
    try {
        referenceCompiledModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "CPU plugin is not available for reference comparison: " << e.what();
    }

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // Host compile can be enabled even when the runtime library is unavailable.
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }
    ov::InferRequest reqReference = referenceCompiledModel.create_infer_request();

    // Start from a regular host tensor.
    ov::Shape shape = {1, 16, 720, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor, "CompileAndInferWithZeroTensor_first");

    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::InferRequest reqDynamic1 = compiledModel.create_infer_request();
    ov::InferRequest reqReference1 = referenceCompiledModel.create_infer_request();
    setInputInferAndCompare(model, reqDynamic1, reqReference1, inTensor, "CompileAndInferWithZeroTensor_second");
    // A fresh infer request rebuilds runtime state on its first execution.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    auto outputTensorFromReq = reqDynamic.get_tensor(model->output());
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            outputTensorFromReq,
                            "CompileAndInferWithZeroTensor_third");
    // Feeding an imported output tensor should update the command list to the new pointer.
    ASSERT_TRUE(customLogger.str().find("Update command list with new tensor pointer") != std::string::npos)
        << "Expected log to contain 'Update command list with new tensor pointer' for third "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    auto zeroContext = core->get_default_context(target_device);
    auto inputHostTensor = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
    auto hostTensorSource = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ASSERT_EQ(hostTensorSource.get_byte_size(), inputHostTensor.get_byte_size())
        << "Source and destination tensors must have identical byte sizes for copy";
    std::memcpy(inputHostTensor.data(), hostTensorSource.data(), hostTensorSource.get_byte_size());
    setInputInferAndCompare(model, reqDynamic1, reqReference1, inputHostTensor, "CompileAndInferWithZeroTensor_fourth");
    // Feeding a context-allocated host tensor should also update the command list to the new pointer.
    ASSERT_TRUE(customLogger.str().find("Update command list with new tensor pointer") != std::string::npos)
        << "Expected log to contain 'Update command list with new tensor pointer' for third "
           "inference, but got: "
        << customLogger.str();
}

// Compare HostCompile inference results against the Template plugin while also checking command-list reuse behavior.
TEST_P(InferWithHostCompileTests, CompileAndInferWithZeroTensorCompareWithReference) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;
    ov::CompiledModel referenceCompiledModel;

    // Capture plugin logs so the test can verify command-list reuse decisions.
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback = [&](std::string_view s) {
        customLogger << s << std::endl;
    };
    ov::util::set_log_callback(customLogCallback);
    struct ResetLogCallbackGuard {
        ~ResetLogCallbackGuard() {
            ov::util::reset_log_callback();
        }
    } reset_log_callback_guard;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));

    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
    try {
        referenceCompiledModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "CPU plugin is not available for reference comparison: " << e.what();
    }

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // Host compile can be enabled even when the runtime library is unavailable.
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }

    ov::InferRequest reqReference = referenceCompiledModel.create_infer_request();

    // Use a regular host tensor for the initial comparison against the Template plugin.
    ov::Shape shape = {1, 16, 720, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            reqDynamic,
                            reqReference,
                            inTensor,
                            "CompileAndInferWithZeroTensorCompareWithReference_first");

    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    inferAndCompare(model, reqDynamic, reqReference, "CompileAndInferWithZeroTensorCompareWithReference_second");
    // Reusing the same input should keep the existing command list intact.
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << customLogger.str();
    auto npuOutputTensorSecondRun = reqDynamic.get_tensor(model->output());

    customLogger.str("");
    customLogger.clear();
    ov::InferRequest reqDynamic1 = compiledModel.create_infer_request();
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    // A fresh infer request rebuilds runtime state on its first execution.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::InferRequest reqReference1 = referenceCompiledModel.create_infer_request();
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            npuOutputTensorSecondRun,
                            "CompileAndInferWithZeroTensorCompareWithReference_third");

    // Feeding an imported output tensor should update the command list to the new pointer.
    ASSERT_TRUE(customLogger.str().find("Update command list with new tensor pointer") != std::string::npos)
        << "Expected log to contain 'Update command list with new tensor pointer' for third "
           "inference, but got: "
        << customLogger.str();
}

// Exercise page-aligned external memory and verify both output correctness and command-list pointer updates.
TEST_P(InferWithHostCompileTests, CompileAndInferWithAlignedTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;
    ov::CompiledModel referenceCompiledModel;

    // Capture plugin logs so the test can verify command-list reuse decisions.
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback = [&](std::string_view s) {
        customLogger << s << std::endl;
    };
    ov::util::set_log_callback(customLogCallback);
    struct ResetLogCallbackGuard {
        ~ResetLogCallbackGuard() {
            ov::util::reset_log_callback();
        }
    } reset_log_callback_guard;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));

    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));
    try {
        referenceCompiledModel = core->compile_model(model, ov::test::utils::DEVICE_TEMPLATE);
    } catch (const ov::Exception& e) {
        GTEST_SKIP() << "CPU plugin is not available for reference comparison: " << e.what();
    }

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // Host compile can be enabled even when the runtime library is unavailable.
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }
    ov::InferRequest reqReference = referenceCompiledModel.create_infer_request();

    // Start from a regular host tensor.
    ov::Shape shape = {1, 16, 720, 768};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor, "CompileAndInferWithAlignedTensor_first");

    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    // Allocate page-aligned external memory so the import path can be exercised.
    auto alignedData = std::unique_ptr<float, void (*)(float*)>(
        static_cast<float*>(
            ::operator new(ov::shape_size(shape) * model->input().get_element_type().size(), std::align_val_t(4096))),
        [](float* ptr) {
            ::operator delete(ptr, std::align_val_t(4096));
        });
    ov::Tensor inTensor1(model->input().get_element_type(), shape, alignedData.get());
    ASSERT_EQ(inTensor.get_byte_size(), inTensor1.get_byte_size())
        << "Source and destination tensors must have identical byte sizes for copy";
    std::memcpy(inTensor1.data(), inTensor.data(), inTensor.get_byte_size());

    setInputInferAndCompare(model, reqDynamic, reqReference, inTensor1, "CompileAndInferWithAlignedTensor_second");

    if (::intel_npu::ZeroInitStructsHolder::getInstance()->isExternalMemoryStandardAllocationSupported()) {
        // Importable external memory should switch execution to the new tensor pointer.
        ASSERT_TRUE(customLogger.str().find("Update command list with new tensor pointer") != std::string::npos)
            << "Expected log to contain 'Update command list with new tensor pointer' for third "
               "inference, but got: "
            << customLogger.str();
    } else {
        // Without import support, execution falls back to copying into the existing internal allocation.
        ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                    std::string::npos)
            << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
               "inference, but got: "
            << customLogger.str();
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
