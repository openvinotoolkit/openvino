// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

#ifdef _WIN32

#    include <d3d12.h>
#    include <wrl.h>

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {
class DX12SharedHeapHelper {
protected:
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    Microsoft::WRL::ComPtr<ID3D12Heap> heap = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> placed_resources = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> comitted_resource;

    HANDLE shared_mem = nullptr;

    void createDevice() {
        auto res = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(device.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(FAILED(res)) << "D3D12CreateDevice failed.";
    }

    void createHeap(const size_t byte_size) {
        const size_t size = (byte_size + (static_cast<size_t>(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) - 1)) &
                            ~(static_cast<size_t>(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) - 1);

        D3D12_HEAP_DESC desc_heap{};
        desc_heap.SizeInBytes = size;
        desc_heap.Properties.Type = D3D12_HEAP_TYPE_CUSTOM;
        desc_heap.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
        desc_heap.Properties.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        desc_heap.Properties.CreationNodeMask = 1;
        desc_heap.Properties.VisibleNodeMask = 1;
        desc_heap.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc_heap.Flags = D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER | D3D12_HEAP_FLAG_SHARED;
        auto res = device->CreateHeap(&desc_heap, IID_PPV_ARGS(heap.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(FAILED(res)) << "CreateHeap failed.";

        res = device->CreateSharedHandle(heap.Get(), nullptr, GENERIC_ALL, nullptr, &shared_mem);
        ASSERT_FALSE(FAILED(res)) << "CreateSharedHandle failed.";
    }

    void createPlacedResources(const size_t byte_size) {
        D3D12_RESOURCE_DESC desc_resource{};
        desc_resource.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc_resource.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc_resource.Width = byte_size;
        desc_resource.Height = 1;
        desc_resource.DepthOrArraySize = 1;
        desc_resource.MipLevels = 1;
        desc_resource.Format = DXGI_FORMAT_UNKNOWN;
        desc_resource.SampleDesc.Count = 1;
        desc_resource.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc_resource.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        auto res = device->CreatePlacedResource(heap.Get(),
                                                0,
                                                &desc_resource,
                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                nullptr,
                                                IID_PPV_ARGS(placed_resources.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(FAILED(res)) << "CreatePlacedResource failed.";
    }

    void createComittedResources(const size_t byte_size) {
        D3D12_HEAP_PROPERTIES heap_properties{};
        heap_properties.Type = D3D12_HEAP_TYPE_UPLOAD;
        heap_properties.CreationNodeMask = 1;
        heap_properties.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC resource_desc{};
        resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resource_desc.Width = byte_size;
        resource_desc.Height = 1;
        resource_desc.DepthOrArraySize = 1;
        resource_desc.MipLevels = 1;
        resource_desc.SampleDesc.Count = 1;
        resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        auto res = device->CreateCommittedResource(&heap_properties,
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &resource_desc,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr,
                                                   IID_PPV_ARGS(comitted_resource.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(FAILED(res)) << "CreateCommittedResource failed.";
    }

    void createResources(const size_t byte_size) {
        createHeap(byte_size);
        createPlacedResources(byte_size);
        createComittedResources(byte_size);
    }

    void copyResources(const size_t byte_size) {
        Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator;
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList4> command_list;
        Microsoft::WRL::ComPtr<ID3D12Fence> fence;
        uint32_t fence_value = 0;

        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        desc.NodeMask = 0;
        auto res = device->CreateCommandQueue(&desc, IID_PPV_ARGS(command_queue.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(FAILED(res)) << "CreateCommandQueue failed.";

        res = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(FAILED(res)) << "CreateFence failed.";

        res = device.Get()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                   IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(FAILED(res)) << "CreateCommandAllocator failed.";

        res = device->CreateCommandList(0,
                                        D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        command_allocator.Get(),
                                        nullptr,
                                        IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(FAILED(res)) << "CreateCommandList failed.";

        command_list->CopyBufferRegion(placed_resources.Get(), 0, comitted_resource.Get(), 0, byte_size);
        res = command_list->Close();
        ASSERT_FALSE(FAILED(res)) << "Close command list failed.";

        ID3D12CommandList* command_lists[] = {command_list.Get()};
        command_queue->ExecuteCommandLists(ARRAYSIZE(command_lists), command_lists);
        res = command_queue->Signal(fence.Get(), ++fence_value);
        ASSERT_FALSE(FAILED(res)) << "Signal command queue failed.";

        volatile auto event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        res = fence->SetEventOnCompletion(fence_value, event);
        ASSERT_FALSE(FAILED(res)) << "SetEventOnCompletion failed.";
        WaitForSingleObject(event, INFINITE);
    }
};

class DX12RemoteRunTests : public ov::test::behavior::OVPluginTestBase,
                           public testing::WithParamInterface<CompilationParams>,
                           protected DX12SharedHeapHelper {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
        ov_model = getDefaultNGraphFunctionForTheDeviceNPU();

        createDevice();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }
};

TEST_P(DX12RemoteRunTests, CheckRemoteTensorSharedBuf) {
    // Skip test according to plugin specific disabled_test_patterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel compiled_model;
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(tensor.get_shape()));

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    createHeap(byte_size);

    auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape(), shared_mem);

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(DX12RemoteRunTests, CheckRemoteTensorSharedBuChangingTensors) {
    // Skip test according to plugin specific disabled_test_patterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel compiled_model;
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(tensor.get_shape()));

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    createHeap(byte_size);

    auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape(), shared_mem);

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random input tensor
    float* random_buffer_tensor = new float[byte_size / sizeof(float)];
    memset(random_buffer_tensor, 1, byte_size);
    ov::Tensor random_tensor_input{ov::element::f32, tensor.get_shape(), random_buffer_tensor};

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(random_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random output tensor
    auto output_tensor = inference_request.get_output_tensor();
    const auto output_byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(output_tensor.get_shape()));

    float* output_random_buffer_tensor = new float[output_byte_size / sizeof(float)];
    memset(output_random_buffer_tensor, 1, output_byte_size);
    ov::Tensor outputrandom_tensor_input{ov::element::f32, output_tensor.get_shape(), output_random_buffer_tensor};

    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(outputrandom_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    delete[] random_buffer_tensor;
}

TEST_P(DX12RemoteRunTests, CheckOutputDataFromMultipleRuns) {
    // Skip test according to plugin specific disabled_test_patterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::CompiledModel compiled_model;
    ov::InferRequest inference_request;
    float* data;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    auto shape = tensor.get_shape();
    const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(shape));
    tensor = {};

    createResources(byte_size);
    void* mem;
    comitted_resource.Get()->Map(0, nullptr, &mem);
    memset(mem, 99, byte_size);
    comitted_resource.Get()->Unmap(0, nullptr);
    copyResources(byte_size);

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    auto output_tensor = inference_request.get_output_tensor();
    const auto output_byte_size = output_tensor.get_byte_size();
    float* output_data_one = new float[output_byte_size / sizeof(float)];
    ov::Tensor output_data_tensor_one{ov::element::f32, output_tensor.get_shape(), output_data_one};

    auto remote_tensor = context.create_tensor(ov::element::f32, shape, shared_mem);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_data_tensor_one));
    OV_ASSERT_NO_THROW(inference_request.infer());

    float* output_data_two = new float[output_byte_size / sizeof(float)];
    ov::Tensor output_data_tensor_two{ov::element::f32, output_tensor.get_shape(), output_data_two};

    data = new float[byte_size / sizeof(float)];
    memset(data, 99, byte_size);
    ov::Tensor input_data_tensor{ov::element::f32, shape, data};
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_data_tensor_two));
    OV_ASSERT_NO_THROW(inference_request.infer());

    delete[] data;

    EXPECT_NE(output_data_one, output_data_two);
    EXPECT_EQ(memcmp(output_data_one, output_data_two, output_byte_size), 0);

    delete[] output_data_one;
    delete[] output_data_two;
}

class DX12RemoteRunDynamicTests : public OVInferRequestDynamicTests, protected DX12SharedHeapHelper {
protected:
    void SetUp() override {
        OVInferRequestDynamicTests::SetUp();
        createDevice();
    }

    void checkOutputFP16(const ov::Tensor& in, const ov::Tensor& actual) {
        auto net = ie->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
        ov::InferRequest req;
        req = net.create_infer_request();
        auto tensor = req.get_tensor(function->inputs().back().get_any_name());
        tensor.set_shape(in.get_shape());
        for (size_t i = 0; i < in.get_size(); i++) {
            tensor.data<ov::element_type_traits<ov::element::f32>::value_type>()[i] =
                in.data<ov::element_type_traits<ov::element::f32>::value_type>()[i];
        }
        req.infer();
        OVInferRequestDynamicTests::checkOutput(actual, req.get_output_tensor(0));
    }
};

TEST_P(DX12RemoteRunDynamicTests, InferDynamicNetworkRemoteTensor) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    std::vector<ov::Shape> vector_shapes{inOutShapes[0].first, inOutShapes[0].first};
    const std::string inputName = "Parameter_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(1, inOutShapes[1].first[0]),
                         ov::Dimension(1, inOutShapes[1].first[1]),
                         ov::Dimension(1, inOutShapes[1].first[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto context = ie->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto inference_request = ie->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    const std::string outputName = "Relu_2";
    for (auto& shape : vector_shapes) {
        ov::Tensor in_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);

        const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(in_tensor.get_shape()));

        createResources(byte_size);
        void* mem;
        comitted_resource.Get()->Map(0, nullptr, &mem);
        memcpy(mem, in_tensor.data(), byte_size);
        comitted_resource.Get()->Unmap(0, nullptr);
        copyResources(byte_size);

        auto remote_tensor = context.create_tensor(ov::element::f32, in_tensor.get_shape(), shared_mem);

        OV_ASSERT_NO_THROW(req = inference_request.create_infer_request());
        OV_ASSERT_NO_THROW(req.set_tensor(inputName, remote_tensor));
        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(in_tensor, req.get_tensor(outputName)));
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

#endif
