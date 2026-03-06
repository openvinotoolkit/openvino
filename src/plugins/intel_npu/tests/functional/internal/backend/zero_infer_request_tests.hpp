// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "../compiler_adapter/zero_init_mock.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "compiled_model.hpp"
#include "graph.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/device_helpers.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/utils/utils.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "transformations.hpp"
#include "zero_backend.hpp"
#include "zero_infer_request.hpp"

namespace {
std::shared_ptr<ov::Model> make_2_input_less_eq(ov::Shape input_shape = {2, 16, 16, 16}) {
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
}  // namespace

namespace ov {
namespace test {
namespace behavior {

using CompilationParamsAndTensorDataType =
    std::tuple<std::string,                     // Device name
               ov::AnyMap,                      // Config
               ov::element::Type,               // Tensor data type
               std::pair<uint32_t, uint32_t>>;  // Graph Ext Version and Mutable Command List Version

class ZeroInferRequestTests : public ov::test::behavior::OVPluginTestBase,
                              public testing::WithParamInterface<CompilationParamsAndTensorDataType> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    ov::element::Type element_type;
    uint32_t zeGraphNpuExtVersion;
    uint32_t zeMutableCommandListExtVersion;
    std::unique_ptr<::intel_npu::FilteredConfig> npu_config;
    std::shared_ptr<::intel_npu::ZeroInitStructsHolder> zeroInitStruct;
    std::shared_ptr<ov::Model> ov_model;

    auto allocate_tensors() -> std::tuple</* importMemoryBatched */ ov::Tensor,
                                          /* importMemoryTensor_1 */ ov::Tensor,
                                          /* importMemoryTensor_2 */ ov::Tensor,
                                          /* unalignedBatchedTensor */ ov::Tensor,
                                          /* unalignedTensor_1 */ ov::Tensor,
                                          /* unalignedTensor_2 */ ov::Tensor> {
        auto model_shape = ov_model->get_parameters()[0]->get_shape();
        ov::Coordinate start_coordinate{model_shape};
        ov::Coordinate stop_coordinate{model_shape};
        start_coordinate[0] = 1;
        stop_coordinate[0] = 2;
        ov::Allocator alignedAllocator{::intel_npu::utils::AlignedAllocator{::intel_npu::utils::STANDARD_PAGE_SIZE}};
        ov::Tensor importMemoryBatchedTensor(ov::element::boolean, model_shape, alignedAllocator);
        ov::Tensor importMemoryTensor_1(importMemoryBatchedTensor, ov::Coordinate{0, 0, 0, 0}, start_coordinate);
        ov::Tensor importMemoryTensor_2(importMemoryBatchedTensor, ov::Coordinate{1, 0, 0, 0}, stop_coordinate);
        void* alignedAddr = ::operator new(ov::element::boolean.size() * ov::shape_size(model_shape) + 1,
                                           std::align_val_t(::intel_npu::utils::STANDARD_PAGE_SIZE));
        void* unalignedAddr = static_cast<uint8_t*>(alignedAddr) + 1;
        std::shared_ptr<void> deallocateAddressCallback(alignedAddr, [](void* ptr) {
            ::operator delete(ptr, std::align_val_t(::intel_npu::utils::STANDARD_PAGE_SIZE));
        });
        auto unalignedBatchedTensorImpl =
            ov::get_tensor_impl(ov::Tensor(ov::element::boolean, model_shape, unalignedAddr));
        unalignedBatchedTensorImpl._so = deallocateAddressCallback;
        ov::Tensor unalignedBatchedTensor = ov::make_tensor(unalignedBatchedTensorImpl);
        ov::Tensor unalignedTensor_1(unalignedBatchedTensor, ov::Coordinate{0, 0, 0, 0}, start_coordinate);
        ov::Tensor unalignedTensor_2(unalignedBatchedTensor, ov::Coordinate{1, 0, 0, 0}, stop_coordinate);

        return {importMemoryBatchedTensor,
                importMemoryTensor_1,
                importMemoryTensor_2,
                unalignedBatchedTensor,
                unalignedTensor_1,
                unalignedTensor_2};
    };

    auto set_tensor_and_infer(::intel_npu::ZeroInferRequest& zero_infer_request,
                              const bool should_infer,
                              const ov::Output<const ov::Node>& port,
                              const ov::SoPtr<ov::ITensor>& tensor) -> void {
        zero_infer_request.set_tensor(port, tensor);
        if (should_infer) {
            zero_infer_request.infer();
        }
    };

    auto set_tensors_and_infer(::intel_npu::ZeroInferRequest& zero_infer_request,
                               const bool should_infer,
                               const ov::Output<const ov::Node>& port,
                               const std::vector<ov::SoPtr<ov::ITensor>>& tensors) -> void {
        zero_infer_request.set_tensors(port, tensors);
        if (should_infer) {
            zero_infer_request.infer();
        }
    };

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParamsAndTensorDataType>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        ov::element::Type type;
        std::pair<uint32_t, uint32_t> graphExtMutableCommandListExtPair;
        std::tie(targetDevice, configuration, type, graphExtMutableCommandListExtPair) = obj.param;
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
        std::tie(target_device, configuration, element_type, graphExtMutableCommandListExtPair) = this->GetParam();
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
            {::intel_npu::PLATFORM::key().data(),
             ov::intel_npu::Platform::standardize(ov::test::utils::getTestsPlatformFromEnvironmentOr(target_device))}};
        npu_config->enable(::intel_npu::PLATFORM::key().data(), true);
        npu_config->enable(::intel_npu::MODEL_SERIALIZER_VERSION::key().data(), true);
        npu_config->enable(::intel_npu::ENABLE_CPU_PINNING::key().data(), true);
        for (const auto& [propertyName, propertyValue] : configuration) {
            configMap[propertyName] = propertyValue.as<std::string>();
            npu_config->enable(propertyName, true);
        }
        npu_config->update(configMap);

        auto zeroInitMock = std::make_shared<::intel_npu::ZeroInitStructsMock>(zeGraphNpuExtVersion,
                                                                               TARGET_ZE_DRIVER_NPU_EXT_VERSION,
                                                                               TARGET_ZE_COMMAND_QUEUE_NPU_EXT_VERSION,
                                                                               TARGET_ZE_PROFILING_NPU_EXT_VERSION,
                                                                               TARGET_ZE_CONTEXT_NPU_EXT_VERSION,
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

TEST_P(ZeroInferRequestTests, CanAllocateZeroTensorForSpecialDataTypes) {
    ov_model = make_2_input_less_eq();

    auto zero_backend = std::make_shared<intel_npu::ZeroEngineBackend>();
    auto device = intel_npu::utils::getDeviceById(zero_backend,
                                                  ov::test::utils::getDeviceNameID(ov::test::utils::getDeviceName()));
    intel_npu::CompilerAdapterFactory compilerFactory;
    auto compilerType = npu_config->get<::intel_npu::COMPILER_TYPE>();
    auto compiler = compilerFactory.getCompiler(
        zero_backend,
        compilerType,
        ov::intel_npu::Platform::standardize(ov::test::utils::getTestsPlatformFromEnvironmentOr(target_device)));

    // WA for error `[NPU_VCL] Unsupported IR API version! Val: 48.0`
    npu_config->update(
        {{::intel_npu::MODEL_SERIALIZER_VERSION::key().data(),
          ::intel_npu::MODEL_SERIALIZER_VERSION::toString(ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY)}});

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

    auto compiledModel =
        std::make_shared<intel_npu::CompiledModel>(ov_model,
                                                   std::make_shared<ov::test::utils::MockPlugin>(),
                                                   device,
                                                   graph,
                                                   *npu_config,
                                                   batch);  // MockPlugin needed only to avoid throw for nullptr
    OPENVINO_ASSERT(compiledModel->inputs()[0].get_element_type() == element_type);
    OPENVINO_ASSERT(compiledModel->inputs()[1].get_element_type() == element_type);

    ::intel_npu::ZeroInferRequest zero_infer_request(zeroInitStruct, compiledModel, *npu_config);
    auto [importMemoryBatchedTensor,
          importMemoryTensor_1,
          importMemoryTensor_2,
          unalignedBatchedTensor,
          unalignedTensor_1,
          unalignedTensor_2] = allocate_tensors();

    OV_ASSERT_NO_THROW(
        set_tensors_and_infer(zero_infer_request,
                              /* should_infer = */ false,
                              compiledModel->inputs()[0],
                              {ov::get_tensor_impl(importMemoryTensor_1), ov::get_tensor_impl(unalignedTensor_2)}));
    OV_ASSERT_NO_THROW(
        set_tensors_and_infer(zero_infer_request,
                              /* should_infer = */ true,
                              compiledModel->inputs()[1],
                              {ov::get_tensor_impl(importMemoryTensor_2), ov::get_tensor_impl(unalignedTensor_1)}));

    OV_ASSERT_NO_THROW(set_tensor_and_infer(zero_infer_request,
                                            /* should_infer = */ false,
                                            compiledModel->inputs()[0],
                                            ov::get_tensor_impl(importMemoryBatchedTensor)));
    OV_ASSERT_NO_THROW(set_tensor_and_infer(zero_infer_request,
                                            /* should_infer = */ true,
                                            compiledModel->inputs()[1],
                                            ov::get_tensor_impl(unalignedBatchedTensor)));

    OV_ASSERT_NO_THROW(
        set_tensors_and_infer(zero_infer_request,
                              /* should_infer = */ false,
                              compiledModel->inputs()[0],
                              {ov::get_tensor_impl(unalignedTensor_1), ov::get_tensor_impl(importMemoryTensor_1)}));
    OV_ASSERT_NO_THROW(
        set_tensors_and_infer(zero_infer_request,
                              /* should_infer = */ true,
                              compiledModel->inputs()[1],
                              {ov::get_tensor_impl(unalignedTensor_2), ov::get_tensor_impl(importMemoryTensor_2)}));
}
