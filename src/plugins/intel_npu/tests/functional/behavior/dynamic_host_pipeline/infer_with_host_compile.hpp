// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <common_test_utils/ov_tensor_utils.hpp>
#include <iostream>
#include <sstream>
#include <string>
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

        // Assume the first 20 char of LLVM blob shall have key word 'llvm'
        std::string region(20, '\0');
        modelStream.read(&region[0], 20);
        region.resize(modelStream.gcount());

        modelStream.clear();
        modelStream.seekg(pos);

        return region.find("llvm") != std::string::npos;
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    bool isTargetDevice = false;
};

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

// The test to compile, create infer request and infer with dynamic shapes, the original shape is large, then set small
// shape
TEST_P(InferWithHostCompileTests, CompileAndInferWithDecreasedSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;

    // Create log callback function which will store log to string, the set to ov
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback =
        [&](std::string_view s) {  // switch to query allocation info for import flag when possible
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

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // check if the exception info is "Cannot load library"
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 700, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Rerun inferrequest with current tensor, the command list is reused without update since no tensor change detected
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor1));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused with
    // data copy
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for third "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::Shape shape2 = {1, 16, 720, 720};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Set new tensor with new shape, it can not be used by runtime directly, local LevelZero tensor are not reused
    // since the original one is too small, command list is reset to run with runtime
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << customLogger.str();
}

// The test to compile, create infer request and infer with dynamic shapes. the original shape is small, then set large
// shape
TEST_P(InferWithHostCompileTests, CompileAndInferWithIncreasedSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;

    // Create log callback function which will store log to string, the set to ov
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback =
        [&](std::string_view s) {  // switch to query allocation info for import flag when possible
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

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // check if the exception info is "Failed to create MLIR runtime engine"
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 720, 720};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // The first time to set tensor, the command list is reset to run with runtime
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Rerun inferrequest with current tensor, the command list is reused without update since no tensor change detected
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor1));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused with
    // data copy
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for third "
           "inference, but got: "
        << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::Shape shape2 = {1, 16, 720, 1280};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Set new tensor with new shape, it can not be used by runtime directly, local LevelZero tensor are not reused
    // since the original one is too small, command list is reset to run with runtime
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << customLogger.str();
}

// The test to compile, create infer request and infer with dynamic shapes. Set LevelZeroTensor to trigger command list
// update
TEST_P(InferWithHostCompileTests, CompileAndInferWithZeroTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;

    // Create log callback function which will store log to string, the set to ov
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback =
        [&](std::string_view s) {  // switch to query allocation info for import flag when possible
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

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // check if the exception info is "Failed to create MLIR runtime engine"
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 720, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());

    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::InferRequest reqDynamic1 = compiledModel.create_infer_request();
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();
    customLogger.str("");
    customLogger.clear();
    auto outputTensorFromReq = reqDynamic.get_tensor(model->output());
    OV_ASSERT_NO_THROW(reqDynamic1.set_input_tensor(0, outputTensorFromReq));
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    // Level zero tensor with same shape will be used instead of local tensor
    ASSERT_TRUE(customLogger.str().find("Update command list with new tensor pointer") != std::string::npos)
        << "Expected log to contain 'Update command list with new tensor pointer' for third "
           "inference, but got: "
        << customLogger.str();

    auto zeroContext = core->get_default_context(target_device);
    auto inputHostTensor = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
    OV_ASSERT_NO_THROW(reqDynamic1.set_input_tensor(0, inputHostTensor));
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    // Level zero tensor with same shape will be used instead of local tensor
    ASSERT_TRUE(customLogger.str().find("Update command list with new tensor pointer") != std::string::npos)
        << "Expected log to contain 'Update command list with new tensor pointer' for third "
           "inference, but got: "
        << customLogger.str();
}

void dumpTensor(const ov::Tensor& tensor);

void dumpTensor(const ov::Tensor& tensor) {
    std::cout << "Tensor shape: " << tensor.get_shape() << ", element type: " << tensor.get_element_type() << std::endl;
    const float* data = tensor.data<float>();
    size_t count = ov::shape_size(tensor.get_shape());
    count = count > 50 ? 50 : count;
    for (size_t i = 0; i < count; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

// The test to compile, create infer request and infer with a LevelZeroTensor input, then compare the output with a
// CPU reference result.
TEST_P(InferWithHostCompileTests, CompileAndInferWithZeroTensorCompareWithReference) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;
    ov::CompiledModel referenceCompiledModel;

    // Create log callback function which will store log to string, the set to ov
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback =
        [&](std::string_view s) {  // switch to query allocation info for import flag when possible
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
        // check if the exception info is "Failed to create MLIR runtime engine"
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }

    ov::InferRequest reqReference = referenceCompiledModel.create_infer_request();

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 720, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    OV_ASSERT_NO_THROW(reqReference.set_input_tensor(0, inTensor));
    OV_ASSERT_NO_THROW(reqReference.infer());

    auto npuOutputTensor = reqDynamic.get_tensor(model->output());
    auto referenceOutputTensor = reqReference.get_tensor(model->output());
    OV_ASSERT_NO_THROW(
        ov::test::utils::compare(referenceOutputTensor, npuOutputTensor, npuOutputTensor.get_element_type()));

    std::cout << "Output input tensor from NPU:" << std::endl;
    dumpTensor(inTensor);
    std::cout << "Output tensor from NPU:" << std::endl;
    dumpTensor(npuOutputTensor);
    std::cout << "Output tensor from reference:" << std::endl;
    dumpTensor(referenceOutputTensor);

    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    OV_ASSERT_NO_THROW(reqReference.infer());
    // Rerun inferrequest with current tensor, the command list is reused without update since no tensor change detected
    ASSERT_TRUE(customLogger.str().find("Reuse command list without update since no tensor change detected") !=
                std::string::npos)
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << customLogger.str();
    auto npuOutputTensorSecondRun = reqDynamic.get_tensor(model->output());
    auto referenceOutputTensorSecondRun = reqReference.get_tensor(model->output());
    OV_ASSERT_NO_THROW(ov::test::utils::compare(referenceOutputTensorSecondRun,
                                                npuOutputTensorSecondRun,
                                                npuOutputTensorSecondRun.get_element_type()));
    std::cout << "Output tensor from NPU after second inference:" << std::endl;
    dumpTensor(npuOutputTensorSecondRun);
    std::cout << "Output tensor from reference after second inference:" << std::endl;
    dumpTensor(referenceOutputTensorSecondRun);

    customLogger.str("");
    customLogger.clear();
    ov::InferRequest reqDynamic1 = compiledModel.create_infer_request();
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    ov::InferRequest reqReference1 = referenceCompiledModel.create_infer_request();
    OV_ASSERT_NO_THROW(reqDynamic1.set_input_tensor(0, npuOutputTensorSecondRun));
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    OV_ASSERT_NO_THROW(reqReference1.set_input_tensor(0, referenceOutputTensorSecondRun));
    OV_ASSERT_NO_THROW(reqReference1.infer());

    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(customLogger.str().find("Update command list with new tensor pointer") != std::string::npos)
        << "Expected log to contain 'Update command list with new tensor pointer' for third "
           "inference, but got: "
        << customLogger.str();

    auto npuOutputTensorThirdRun = reqDynamic1.get_tensor(model->output());
    auto referenceOutputTensorThirdRun = reqReference1.get_tensor(model->output());
    OV_ASSERT_NO_THROW(ov::test::utils::compare(referenceOutputTensorThirdRun,
                                                npuOutputTensorThirdRun,
                                                npuOutputTensorThirdRun.get_element_type()));
    std::cout << "Output tensor from NPU after third inference:" << std::endl;
    dumpTensor(npuOutputTensorThirdRun);
    std::cout << "Output tensor from reference after third inference:" << std::endl;
    dumpTensor(referenceOutputTensorThirdRun);
}

// The test to compile, create infer request and infer with dynamic shapes. Set tensor that can be imported by level
// zero to trigger command list update
TEST_P(InferWithHostCompileTests, CompileAndInferWithAlignedTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;

    // Create log callback function which will store log to string, the set to ov
    std::stringstream customLogger;
    std::function<void(std::string_view)> customLogCallback =
        [&](std::string_view s) {  // switch to query allocation info for import flag when possible
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

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        // check if the exception info is "Failed to create MLIR runtime engine"
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 720, 768};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());

    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(customLogger.str().find("Reset command list to run with runtime") != std::string::npos)
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << customLogger.str();

    customLogger.str("");
    customLogger.clear();
    // shape size is aligned to standard page size, align address as well
    auto data = static_cast<float*>(
        ::operator new(ov::shape_size(shape) * model->input().get_element_type().size(), std::align_val_t(4096)));
    // auto AlignedTensor = ov::make_tensor(model->input().get_element_type(), shape, data);
    ov::Tensor inTensor1(model->input().get_element_type(), shape, data);

    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor1));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(customLogger.str().find("Update command list with new tensor pointer") != std::string::npos)
        << "Expected log to contain 'Update command list with new tensor pointer' for third "
           "inference, but got: "
        << customLogger.str();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
