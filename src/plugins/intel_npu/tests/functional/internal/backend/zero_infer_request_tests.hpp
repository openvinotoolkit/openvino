// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "../compiler_adapter/zero_init_mock.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "compiled_model.hpp"
#include "driver_compiler_adapter.hpp"
#include "graph.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "plugin_compiler_adapter.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "transformations.hpp"
#include "zero_backend.hpp"
#include "zero_infer_request.hpp"

namespace {
[[maybe_unused]] std::shared_ptr<ov::Model> make_2_input_less_eq(const ov::Shape& input_shape = {2, 16, 16, 16}) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, input_shape);
    param0->get_output_tensor(0).set_names({"tensor_input_0"});
    param0->set_layout("N...");
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, input_shape);
    param1->get_output_tensor(0).set_names({"tensor_input_1"});
    param1->set_layout("N...");
    auto lessEqual = std::make_shared<ov::op::v1::LessEqual>(param0, param1);
    auto result = std::make_shared<ov::op::v0::Result>(lessEqual);
    result->get_output_tensor(0).set_names({"tensor_output"});

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0, param1});
    model->set_friendly_name("TwoInputLessEqual");
    return model;
}

std::shared_ptr<ov::Model> make_2_input_less_eq_non_zero(const ov::Shape& input_shape = {2, 16, 16, 16}) {
    auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, input_shape);
    param0->get_output_tensor(0).set_names({"tensor_input_0"});
    param0->set_layout("N...");
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, input_shape);
    param1->get_output_tensor(0).set_names({"tensor_input_1"});
    param1->set_layout("N...");
    auto nonZero0 = std::make_shared<ov::op::v3::NonZero>(param0);
    auto nonZero1 = std::make_shared<ov::op::v3::NonZero>(param1);
    // NonZero Op returns 2D tensor [rank(data), num_non_zero] where second value is dynamic depending on the input
    // need to workaround this for NPU_BATCH_MODE=PLUGIN using intermediary op e.g. ShapeOf
    auto shapeOf0 = std::make_shared<ov::op::v0::ShapeOf>(nonZero0);
    auto shapeOf1 = std::make_shared<ov::op::v0::ShapeOf>(nonZero1);
    auto add = std::make_shared<ov::op::v1::Add>(shapeOf0, shapeOf1);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    result->get_output_tensor(0).set_names({"tensor_output"});

    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param0, param1});
    model->set_friendly_name("TwoTwoInputNonZeroAddModel");
    return model;
}

}  // namespace

namespace ov {
namespace test {
namespace behavior {

using CompilationParamsAndTensorDataType =
    std::tuple<std::string,                     // Device name
               ov::AnyMap,                      // Config
               ov::element::Type,               // Tensor data type
               bool,                            // With warmup infer
               bool,                            // With reset infer request
               std::pair<uint32_t, uint32_t>>;  // Graph Ext Version and Mutable Command List Version

class ZeroInferRequestTests : public ov::test::behavior::OVPluginTestBase,
                              public testing::WithParamInterface<CompilationParamsAndTensorDataType> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    ov::element::Type element_type;
    bool withWarmUpInfer;
    bool withResetInferRequest;
    uint32_t zeGraphNpuExtVersion;
    uint32_t zeMutableCommandListExtVersion;
    std::unique_ptr<::intel_npu::FilteredConfig> npu_config;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> zeroInitStruct;
    std::shared_ptr<ov::Model> ov_model;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParamsAndTensorDataType>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        ov::element::Type type;
        bool withWarmUpInfer;
        bool withResetInferRequest;
        std::pair<uint32_t, uint32_t> graphExtMutableCommandListExtPair;

        std::tie(targetDevice,
                 configuration,
                 type,
                 withWarmUpInfer,
                 withResetInferRequest,
                 graphExtMutableCommandListExtPair) = obj.param;
        auto [zeGraphNpuExtVersion, zeMutableCommandListExtVersion] = graphExtMutableCommandListExtPair;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
                result << "_";
            }
        }
        if (!type.get_type_name().empty()) {
            result << "tensorDataType=" << type.get_type_name() << "_";
        }

        result << "withWarmUpInfer=" << withWarmUpInfer << "_" << "withResetInferRequest=" << withResetInferRequest
               << "_";
        result << "zeGraphNpuExtVersion=" + std::to_string(ZE_MAJOR_VERSION(zeGraphNpuExtVersion)) + "." +
                      std::to_string(ZE_MINOR_VERSION(zeGraphNpuExtVersion))
               << "_";
        result << "zeMutableCommandListExtVersion=" + std::to_string(ZE_MAJOR_VERSION(zeMutableCommandListExtVersion)) +
                      "." + std::to_string(ZE_MINOR_VERSION(zeMutableCommandListExtVersion));

        return result.str();
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        std::pair<uint32_t, uint32_t> graphExtMutableCommandListExtPair;
        std::tie(target_device,
                 configuration,
                 element_type,
                 withWarmUpInfer,
                 withResetInferRequest,
                 graphExtMutableCommandListExtPair) = this->GetParam();
        zeGraphNpuExtVersion = graphExtMutableCommandListExtPair.first;
        zeMutableCommandListExtVersion = graphExtMutableCommandListExtPair.second;

        auto options = std::make_shared<::intel_npu::OptionsDesc>();
        options->add<::intel_npu::PLATFORM>();
        options->add<::intel_npu::COMPILER_TYPE>();
        options->add<::intel_npu::BATCH_MODE>();
        options->add<::intel_npu::MODEL_SERIALIZER_VERSION>();
        options->add<::intel_npu::ENABLE_CPU_PINNING>();
        npu_config = std::make_unique<::intel_npu::FilteredConfig>(options);
        ::intel_npu::Config::ConfigMap configMap{
            /* { ::intel_npu::PLATFORM::key().data(),
             ov::intel_npu::Platform::standardize(ov::test::utils::getTestsPlatformFromEnvironmentOr(target_device)) } */ };
        npu_config->enable(::intel_npu::PLATFORM::key().data(), true);
        npu_config->enable(::intel_npu::MODEL_SERIALIZER_VERSION::key().data(), true);
        npu_config->enable(::intel_npu::ENABLE_CPU_PINNING::key().data(), true);
        for (const auto& [propertyName, propertyValue] : configuration) {
            configMap[propertyName] = propertyValue.as<std::string>();
            npu_config->enable(propertyName, true);
        }
        npu_config->update(configMap);

        auto zeroInitMock = std::make_shared<::intel_npu::ZeroInitStructsMock>(
            ::intel_npu::test_constants::TARGET_ZE_DRIVER_NPU_EXT_VERSION,
            zeGraphNpuExtVersion,
            ::intel_npu::test_constants::TARGET_ZE_COMMAND_QUEUE_NPU_EXT_VERSION,
            ::intel_npu::test_constants::TARGET_ZE_PROFILING_NPU_EXT_VERSION,
            ::intel_npu::test_constants::TARGET_ZE_CONTEXT_NPU_EXT_VERSION,
            zeMutableCommandListExtVersion);
        zeroInitStruct = std::reinterpret_pointer_cast<::intel_npu::ZeroInitStructsHolder>(zeroInitMock);

        OVPluginTestBase::SetUp();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }
};

}  // namespace behavior
}  // namespace test
}  // namespace ov

TEST_P(ZeroInferRequestTests, BooleanSetTensorSetTensorsWork) {
    // TODO: Also test make_2_input_less_eq model when we have compiler support
    ov_model = make_2_input_less_eq_non_zero();

    // FIXME: OV needs to update ov::util::load_shared_object logic to check if certain module was already loaded in
    // memory
    // Force loading compiler lib via core flow to avoid missing library in the current path
    core->set_property(target_device, ov::intel_npu::compiler_type(npu_config->get<::intel_npu::COMPILER_TYPE>()));
    (void)core->get_property(target_device, ov::supported_properties);

    auto zero_backend = std::make_shared<::intel_npu::ZeroEngineBackend>();
    auto device = intel_npu::utils::getDeviceById(zero_backend,
                                                  ov::test::utils::getDeviceNameID(ov::test::utils::getDeviceName()));

    std::shared_ptr<::intel_npu::ICompilerAdapter> compiler;
    try {
        compiler = npu_config->get<::intel_npu::COMPILER_TYPE>() == ov::intel_npu::CompilerType::DRIVER
                       ? std::dynamic_pointer_cast<::intel_npu::ICompilerAdapter>(
                             std::make_shared<::intel_npu::DriverCompilerAdapter>(zeroInitStruct))
                       : std::dynamic_pointer_cast<::intel_npu::ICompilerAdapter>(
                             std::make_shared<::intel_npu::PluginCompilerAdapter>(zeroInitStruct));
    } catch (...) {
        GTEST_SKIP() << "Couldn't load compiler library";
    }

    // WA for error `[NPU_VCL] Unsupported IR API version! Val: 48.0`
    if (compiler->is_option_supported(::intel_npu::MODEL_SERIALIZER_VERSION::key().data())) {
        npu_config->update({{::intel_npu::MODEL_SERIALIZER_VERSION::key().data(),
                             ::intel_npu::MODEL_SERIALIZER_VERSION::toString(
                                 ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY)}});
    }

    // logic for batch
    auto copy_model = ov_model;
    std::optional<int64_t> batch = std::nullopt;
    if (npu_config->has<::intel_npu::BATCH_MODE>() &&
        npu_config->get<::intel_npu::BATCH_MODE>() == ov::intel_npu::BatchMode::PLUGIN) {
        std::optional<ov::Dimension> originalBatch = std::nullopt;
        auto [batchedModel, successfullyDebatched] = intel_npu::batch_helpers::handlePluginBatching(
            ov_model,
            *npu_config,
            [&](ov::intel_npu::BatchMode mode) {
                npu_config->update({{::intel_npu::BATCH_MODE::key().data(), ::intel_npu::BATCH_MODE::toString(mode)}});
            },
            originalBatch,
            ::intel_npu::Logger::global());
        OPENVINO_ASSERT(successfullyDebatched, "Couldn't debatch test model!");
        batch = originalBatch.value().get_length();
        copy_model = batchedModel;
    }

    auto graph = compiler->compile(copy_model, *npu_config);
    if (batch) {
        graph->set_batch_size(batch.value());
    }

    auto compiledModel = std::make_shared<intel_npu::CompiledModel>(
        ov_model,
        std::make_shared<ov::test::utils::MockPlugin>(),
        device,
        graph,
        *npu_config,
        batch,
        /* encryptionCallbackOpt = */ std::nullopt);  // MockPlugin needed only to avoid throw for nullptr
    OPENVINO_ASSERT(compiledModel->inputs()[0].get_element_type() == element_type);
    OPENVINO_ASSERT(compiledModel->inputs()[1].get_element_type() == element_type);

    auto zero_infer_request =
        std::make_shared<::intel_npu::ZeroInferRequest>(zeroInitStruct, compiledModel, *npu_config);
    auto [importMemoryBatchedTensor,
          importMemoryTensor_1,
          importMemoryTensor_2,
          unalignedBatchedTensor,
          unalignedTensor_1,
          unalignedTensor_2] = ov::test::utils::allocate_tensors(ov_model, ov::element::boolean);

    ov::Tensor ref_import_tensor(importMemoryBatchedTensor.get_element_type(), importMemoryBatchedTensor.get_shape());
    ov::Tensor ref_tensor_unaligned(unalignedBatchedTensor.get_element_type(), unalignedBatchedTensor.get_shape());
    importMemoryBatchedTensor.copy_to(ref_import_tensor);
    unalignedBatchedTensor.copy_to(ref_tensor_unaligned);

    auto reset_infer_cb = [this, &zero_infer_request, &compiledModel]() -> void {
        zero_infer_request =
            std::make_shared<::intel_npu::ZeroInferRequest>(zeroInitStruct, compiledModel, *npu_config);
    };

    if (withWarmUpInfer) {
        zero_infer_request->infer();
    }

    OV_ASSERT_NO_THROW(ov::test::utils::set_tensors_and_infer(
        zero_infer_request,
        /* should_infer = */ false,
        withResetInferRequest,
        compiledModel->inputs()[0],
        std::vector<ov::SoPtr<ov::ITensor>>{ov::get_tensor_impl(importMemoryTensor_1),
                                            ov::get_tensor_impl(unalignedTensor_2)},
        reset_infer_cb));
    OV_ASSERT_NO_THROW(ov::test::utils::set_tensors_and_infer(
        zero_infer_request,
        /* should_infer = */ true,
        withResetInferRequest,
        compiledModel->inputs()[1],
        std::vector<ov::SoPtr<ov::ITensor>>{ov::get_tensor_impl(importMemoryTensor_2),
                                            ov::get_tensor_impl(unalignedTensor_1)},
        reset_infer_cb));

    OV_ASSERT_NO_THROW(ov::test::utils::set_tensor_and_infer(zero_infer_request,
                                                             /* should_infer = */ false,
                                                             withResetInferRequest,
                                                             compiledModel->inputs()[0],
                                                             ov::get_tensor_impl(importMemoryBatchedTensor),
                                                             reset_infer_cb));
    OV_ASSERT_NO_THROW(ov::test::utils::set_tensor_and_infer(zero_infer_request,
                                                             /* should_infer = */ true,
                                                             withResetInferRequest,
                                                             compiledModel->inputs()[1],
                                                             ov::get_tensor_impl(unalignedBatchedTensor),
                                                             reset_infer_cb));

    OV_ASSERT_NO_THROW(ov::test::utils::set_tensors_and_infer(
        zero_infer_request,
        /* should_infer = */ false,
        withResetInferRequest,
        compiledModel->inputs()[0],
        std::vector<ov::SoPtr<ov::ITensor>>{ov::get_tensor_impl(unalignedTensor_1),
                                            ov::get_tensor_impl(importMemoryTensor_1)},
        reset_infer_cb));
    OV_ASSERT_NO_THROW(ov::test::utils::set_tensors_and_infer(
        zero_infer_request,
        /* should_infer = */ true,
        withResetInferRequest,
        compiledModel->inputs()[1],
        std::vector<ov::SoPtr<ov::ITensor>>{ov::get_tensor_impl(unalignedTensor_2),
                                            ov::get_tensor_impl(importMemoryTensor_2)},
        reset_infer_cb));

    // ensure user tensors don't get altered during tests
    OV_ASSERT_NO_THROW(ov::test::utils::compare(ref_import_tensor, importMemoryBatchedTensor, ov::element::boolean));
    OV_ASSERT_NO_THROW(ov::test::utils::compare(ref_tensor_unaligned, importMemoryBatchedTensor, ov::element::boolean));
}
