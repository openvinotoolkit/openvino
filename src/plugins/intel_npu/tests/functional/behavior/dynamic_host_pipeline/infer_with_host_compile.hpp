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
#include <type_traits>
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
            std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{1, 16, 720, ov::Dimension(10, 1280)});
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

    static bool logContains(const ScopedLogCapture& logCapture, const std::string& expectedEntry);

    RuntimeCompareSetupResult prepareRuntimeCompareContext(const std::shared_ptr<ov::Model>& model);

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    bool isTargetDevice = false;
};

InferWithHostCompileTests::ScopedLogCapture::ScopedLogCapture()
    : callback([this](std::string_view s) {
          stream << s << std::endl;
          std::cout << s << std::endl;
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

namespace {

template <typename T>
void dumpValuesAsType(const ov::Tensor& tensor, size_t count, std::ostream& os) {
    const T* data = tensor.data<const T>();
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            os << static_cast<int>(data[i]) << " ";
        } else if constexpr (std::is_same_v<T, ov::float16> || std::is_same_v<T, ov::bfloat16>) {
            os << static_cast<float>(data[i]) << " ";
        } else if constexpr (std::is_same_v<T, bool>) {
            os << (data[i] ? 1 : 0) << " ";
        } else {
            os << data[i] << " ";
        }
    }
}

void dumpTensorValues(const ov::Tensor& tensor, size_t count, std::ostream& os) {
    switch (tensor.get_element_type()) {
    case ov::element::Type_t::boolean:
        dumpValuesAsType<bool>(tensor, count, os);
        break;
    case ov::element::Type_t::bf16:
        dumpValuesAsType<ov::bfloat16>(tensor, count, os);
        break;
    case ov::element::Type_t::f16:
        dumpValuesAsType<ov::float16>(tensor, count, os);
        break;
    case ov::element::Type_t::f32:
        dumpValuesAsType<float>(tensor, count, os);
        break;
    case ov::element::Type_t::f64:
        dumpValuesAsType<double>(tensor, count, os);
        break;
    case ov::element::Type_t::i8:
        dumpValuesAsType<int8_t>(tensor, count, os);
        break;
    case ov::element::Type_t::i16:
        dumpValuesAsType<int16_t>(tensor, count, os);
        break;
    case ov::element::Type_t::i32:
        dumpValuesAsType<int32_t>(tensor, count, os);
        break;
    case ov::element::Type_t::i64:
        dumpValuesAsType<int64_t>(tensor, count, os);
        break;
    case ov::element::Type_t::u8:
        dumpValuesAsType<uint8_t>(tensor, count, os);
        break;
    case ov::element::Type_t::u16:
        dumpValuesAsType<uint16_t>(tensor, count, os);
        break;
    case ov::element::Type_t::u32:
        dumpValuesAsType<uint32_t>(tensor, count, os);
        break;
    case ov::element::Type_t::u64:
        dumpValuesAsType<uint64_t>(tensor, count, os);
        break;
    default: {
        // Fallback to byte-wise dump for unsupported element types.
        const uint8_t* data = tensor.data<const uint8_t>();
        const size_t byteCount = std::min(count * tensor.get_element_type().size(), tensor.get_byte_size());
        for (size_t i = 0; i < byteCount; ++i) {
            os << static_cast<unsigned int>(data[i]) << " ";
        }
        break;
    }
    }
}

}  // namespace

void InferWithHostCompileTests::dumpTensor(const ov::Tensor& tensor, std::string name) {
    std::cout << "Tensor name: " << name << ", shape: " << tensor.get_shape()
              << ", element type: " << tensor.get_element_type() << std::endl;
    size_t count = ov::shape_size(tensor.get_shape());
    count = count > 50 ? 50 : count;
    dumpTensorValues(tensor, count, std::cout);
    std::cout << std::endl;

    // Add a random suffix to avoid collisions when tests run in parallel.
    std::cout << "Dump tensor to file for debugging, tensor name: " << name << std::endl;
    std::string fileName = name + "_" + std::to_string(std::rand()) + ".txt";
    std::ofstream outFile(fileName);
    if (outFile.is_open()) {
        outFile << "Tensor name: " << name << ", shape: " << tensor.get_shape()
                << ", element type: " << tensor.get_element_type() << std::endl;
        size_t totalCount = ov::shape_size(tensor.get_shape());
        dumpTensorValues(tensor, totalCount, outFile);
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

bool InferWithHostCompileTests::logContains(const ScopedLogCapture& logCapture, const std::string& expectedEntry) {
    return logCapture.str().find(expectedEntry) != std::string::npos;
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
        if (std::string(e.what()).find("Cannot load library") == std::string::npos) {
            result.status = RuntimeCompareStatus::fail;
            result.message =
                std::string("Expected exception message to contain 'Cannot load library', but got: ") + e.what();
            return result;
        }

        result.status = RuntimeCompareStatus::skip;
        result.message = "Cannot load library, skip test.";
        return result;
    }

    result.context.reqReference = result.context.referenceCompiledModel.create_infer_request();
    return result;
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
    ov::Shape shape = {1, 16, 720, 1280};
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
    ov::Shape shape2 = {1, 16, 720, 720};
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

    auto model = createMaxPoolModel();
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
    ov::Shape shape = {1, 16, 720, 720};
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
    ov::Shape shape2 = {1, 16, 720, 1280};
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

    auto model = createMaxPoolModel();
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
    ov::Shape shape = {1, 16, 720, 1280};
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
    // Feeding an imported output tensor should update the command list to the new pointer.
    ASSERT_TRUE(logContains(logCapture, "Update command list with new tensor pointer"))
        << "Expected log to contain 'Update command list with new tensor pointer' for third inference, but got: "
        << logCapture.str();

    logCapture.clear();
    auto zeroContext = core->get_default_context(target_device);
    auto inputHostTensor = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
    auto hostTensorSource = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ASSERT_EQ(hostTensorSource.get_byte_size(), inputHostTensor.get_byte_size())
        << "Source and destination tensors must have identical byte sizes for copy";
    std::memcpy(inputHostTensor.data(), hostTensorSource.data(), hostTensorSource.get_byte_size());
    setInputInferAndCompare(model, reqDynamic1, reqReference1, inputHostTensor, "CompileAndInferWithZeroTensor_fourth");
    // Feeding a context-allocated host tensor should also update the command list to the new pointer.
    ASSERT_TRUE(logContains(logCapture, "Update command list with new tensor pointer"))
        << "Expected log to contain 'Update command list with new tensor pointer' for fourth inference, but got: "
        << logCapture.str();
}

// Compare HostCompile inference results against the Template plugin while also checking command-list reuse behavior.
TEST_P(InferWithHostCompileTests, CompileAndInferWithZeroTensorCompareWithReference) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
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

    // Use a regular host tensor for the initial comparison against the Template plugin.
    ov::Shape shape = {1, 16, 720, 1280};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor,
                            "CompileAndInferWithZeroTensorCompareWithReference_first");

    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    inferAndCompare(model,
                    testContext.reqDynamic,
                    testContext.reqReference,
                    "CompileAndInferWithZeroTensorCompareWithReference_second");
    // Reusing the same input should keep the existing command list intact.
    ASSERT_TRUE(logContains(logCapture, "Reuse command list without update since no tensor change detected"))
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << logCapture.str();
    auto npuOutputTensorSecondRun = testContext.reqDynamic.get_tensor(model->output());

    logCapture.clear();
    ov::InferRequest reqDynamic1 = testContext.compiledModel.create_infer_request();
    OV_ASSERT_NO_THROW(reqDynamic1.infer());
    // A fresh infer request rebuilds runtime state on its first execution.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    ov::InferRequest reqReference1 = testContext.referenceCompiledModel.create_infer_request();
    setInputInferAndCompare(model,
                            reqDynamic1,
                            reqReference1,
                            npuOutputTensorSecondRun,
                            "CompileAndInferWithZeroTensorCompareWithReference_third");

    // Feeding an imported output tensor should update the command list to the new pointer.
    ASSERT_TRUE(logContains(logCapture, "Update command list with new tensor pointer"))
        << "Expected log to contain 'Update command list with new tensor pointer' for third inference, but got: "
        << logCapture.str();
}

// Exercise page-aligned external memory and verify both output correctness and command-list pointer updates.
TEST_P(InferWithHostCompileTests, CompileAndInferWithAlignedTensor) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
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
    ov::Shape shape = {1, 16, 720, 768};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor,
                            "CompileAndInferWithAlignedTensor_first");

    // The first run materializes runtime state for the initial shape.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
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

    setInputInferAndCompare(model,
                            testContext.reqDynamic,
                            testContext.reqReference,
                            inTensor1,
                            "CompileAndInferWithAlignedTensor_second");

    if (::intel_npu::ZeroInitStructsHolder::getInstance()->isExternalMemoryStandardAllocationSupported()) {
        // Importable external memory should switch execution to the new tensor pointer.
        ASSERT_TRUE(logContains(logCapture, "Update command list with new tensor pointer"))
            << "Expected log to contain 'Update command list with new tensor pointer' for second inference, but got: "
            << logCapture.str();
    } else {
        // Without import support, execution falls back to copying into the existing internal allocation.
        ASSERT_TRUE(logContains(logCapture, "Reuse command list without update since no tensor change detected"))
            << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
               "inference, but got: "
            << logCapture.str();
    }
}

TEST_P(InferWithHostCompileTests, CompileAndInferWithRandomSize) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ov::CompiledModel compiledModel;

    // Create log callback function which will store log to string, the set to ov
    ScopedLogCapture logCapture;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));

    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 720, 720};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Reusing the same input should keep the existing command list intact.
    ASSERT_TRUE(logContains(logCapture, "Reuse command list without update since no tensor change detected"))
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for second "
           "inference, but got: "
        << logCapture.str();

    logCapture.clear();
    ov::Tensor inTensor1 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor1));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Shape change should force runtime reconfiguration for the new tensor layout.
    ASSERT_TRUE(logContains(logCapture, "Reuse command list without update since no tensor change detected"))
        << "Expected log to contain 'Reuse command list without update since no tensor change detected' for third "
           "inference, but got: "
        << logCapture.str();

    logCapture.clear();
    ov::Shape shape2 = {1, 16, 720, 1024};
    ov::Tensor inTensor2 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape2, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor2));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Shape change should force runtime reconfiguration for the new tensor layout.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << logCapture.str();

    logCapture.clear();
    ov::Shape shape3 = {1, 16, 720, 360};
    ov::Tensor inTensor3 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape3, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor3));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Shape change should force runtime reconfiguration for the new tensor layout.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << logCapture.str();

    logCapture.clear();
    ov::Shape shape4 = {1, 16, 720, 1280};
    ov::Tensor inTensor4 = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape4, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor4));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Shape change should force runtime reconfiguration for the new tensor layout.
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime' for fourth inference with new shape, but "
           "got: "
        << logCapture.str();
}

// Exercise imported Level Zero tensors and verify both output correctness and command-list pointer updates.
TEST_P(InferWithHostCompileTests, CompileAndInferWithZeroTensorWithoutReference) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    if (!isTargetDevice) {
        GTEST_SKIP() << "Skip test for current device";
    }

    auto model = createMaxPoolModel();
    ScopedLogCapture logCapture;

    core->set_property("NPU", ov::log::level(ov::log::Level::DEBUG));
    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(model, target_device, configuration));

    ov::InferRequest reqDynamic;
    try {
        reqDynamic = compiledModel.create_infer_request();
    } catch (const ov::Exception& e) {
        ASSERT_TRUE(std::string(e.what()).find("Cannot load library") != std::string::npos)
            << "Expected exception message to contain 'Cannot load library', but got: " << e.what();
        GTEST_SKIP() << "Cannot load library, skip test.";
    }

    // create input tensor match the customized models
    ov::Shape shape = {1, 16, 720, 720};
    ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Set new tensor with same shape, it can not be used by runtime directly, local LevelZero tensor are reused
    ASSERT_TRUE(logContains(logCapture, "Reset command list to run with runtime"))
        << "Expected log to contain 'Reset command list to run with runtime', but got: " << logCapture.str();

    logCapture.clear();
    auto zeroContext = core->get_default_context(target_device);
    auto inputHostTensor = zeroContext.create_host_tensor(model->input().get_element_type(), shape);
    auto hostTensorSource = ov::test::utils::create_and_fill_tensor(model->input().get_element_type(), shape, 100, 0);
    ASSERT_EQ(hostTensorSource.get_byte_size(), inputHostTensor.get_byte_size())
        << "Source and destination tensors must have identical byte sizes for copy";
    std::memcpy(inputHostTensor.data(), hostTensorSource.data(), hostTensorSource.get_byte_size());
    OV_ASSERT_NO_THROW(reqDynamic.set_input_tensor(0, inputHostTensor));
    OV_ASSERT_NO_THROW(reqDynamic.infer());
    // Feeding a context-allocated host tensor should also update the command list to the new pointer.
    ASSERT_TRUE(logContains(logCapture, "Update command list with new tensor pointer"))
        << "Expected log to contain 'Update command list with new tensor pointer' for fourth inference, but got: "
        << logCapture.str();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
