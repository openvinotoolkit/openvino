// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "overload/overload_test_utils_npu.hpp"

#ifdef _WIN32
#    ifdef ENABLE_DX12
#        include <initguid.h>  // it has to be placed before dxcore
#    endif
#endif

#ifdef _WIN32
#    ifdef ENABLE_DX12
#        ifndef NOMINMAX
#            define NOMINMAX
#            define NOMINMAX_DEFINED_CTX_UT
#        endif

#        include <combaseapi.h>
#        include <d3d12.h>
#        include <d3dcommon.h>
#        include <dxcore.h>
#        include <dxcore_interface.h>
#        include <wrl.h>
#        include <wrl/client.h>

#        include "d3dx12_core.h"

#        ifdef NOMINMAX_DEFINED_CTX_UT
#            undef NOMINMAX
#            undef NOMINMAX_DEFINED_CTX_UT
#        endif

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {
class DX12RemoteRunTests : public ov::test::behavior::OVPluginTestBase,
                           public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;
    ov::CompiledModel compiled_model;

    Microsoft::WRL::ComPtr<IDXCoreAdapter> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device9> device;
    Microsoft::WRL::ComPtr<ID3D12Heap> heap = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> placed_resources = nullptr;
    Microsoft::WRL::ComPtr<ID3D12Resource> comitted_resource;

    HANDLE shared_mem = nullptr;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
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

        createAdapter();
        creeateDevice();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        APIBaseTest::TearDown();
    }

    void createAdapter() {
        Microsoft::WRL::ComPtr<IDXCoreAdapterFactory> factory;

        auto res = DXCoreCreateAdapterFactory(IID_PPV_ARGS(factory.GetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "DXCoreCreateAdapterFactory failed.";

        const auto regex = std::regex("^\\bIntel\\b.*?\\bGraphics\\b.*?");
        const GUID guids[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE};

        // create the adapter list
        Microsoft::WRL::ComPtr<IDXCoreAdapterList> adapter_list;
        res = factory->CreateAdapterList(ARRAYSIZE(guids), guids, IID_PPV_ARGS(adapter_list.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateAdapterList failed.";

        // find our adapter
        for (uint32_t iter = 0; iter < adapter_list->GetAdapterCount(); iter++) {
            Microsoft::WRL::ComPtr<IDXCoreAdapter> local_adapter;
            res = adapter_list->GetAdapter(iter, IID_PPV_ARGS(local_adapter.ReleaseAndGetAddressOf()));
            ASSERT_FALSE(res != S_OK) << "GetAdapter failed.";

            size_t driver_desc_size = 0;
            res = local_adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &driver_desc_size);
            ASSERT_FALSE(res != S_OK) << "GetPropertySize failed.";

            std::vector<char> driver_desc(driver_desc_size);
            res =
                local_adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, driver_desc_size, &driver_desc[0]);
            ASSERT_FALSE(res != S_OK) << "GetProperty failed.";

            if (std::regex_match(std::string(driver_desc.data()), regex)) {
                adapter = local_adapter;
                break;
            }
        }

        auto check_adapter = adapter->IsValid();
        if (!check_adapter) {
            OPENVINO_THROW("GPU adapter is not valid");
        }
    }

    void creeateDevice() {
        auto res =
            D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_1_0_CORE, IID_PPV_ARGS(device.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "D3D12CreateDevice failed.";
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
        ASSERT_FALSE(res != S_OK) << "CreateHeap failed.";

        res = device->CreateSharedHandle(heap.Get(), nullptr, GENERIC_ALL, nullptr, &shared_mem);
        ASSERT_FALSE(res != S_OK) << "CreateSharedHandle failed.";
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
        ASSERT_FALSE(res != S_OK) << "CreatePlacedResource failed.";
    }

    void createComittedResources(const size_t byte_size) {
        auto res = device->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
                                                   D3D12_HEAP_FLAG_NONE,
                                                   &CD3DX12_RESOURCE_DESC::Buffer(byte_size),
                                                   D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr,
                                                   IID_PPV_ARGS(comitted_resource.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateCommittedResource failed.";
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
        ASSERT_FALSE(res != S_OK) << "CreateCommandQueue failed.";

        res = device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(fence.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateFence failed.";

        res = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                             IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateCommandAllocator failed.";

        res = device->CreateCommandList(0,
                                        D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                        command_allocator.Get(),
                                        nullptr,
                                        IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf()));
        ASSERT_FALSE(res != S_OK) << "CreateCommandList failed.";

        command_list->CopyBufferRegion(placed_resources.Get(), 0, comitted_resource.Get(), 0, byte_size);
        res = command_list->Close();
        ASSERT_FALSE(res != S_OK) << "Close command list failed.";

        ID3D12CommandList* command_lists[] = {command_list.Get()};
        command_queue->ExecuteCommandLists(ARRAYSIZE(command_lists), command_lists);
        res = command_queue->Signal(fence.Get(), ++fence_value);
        ASSERT_FALSE(res != S_OK) << "Signal command queue failed.";

        volatile auto event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        res = fence->SetEventOnCompletion(fence_value, event);
        ASSERT_FALSE(res != S_OK) << "SetEventOnCompletion failed.";
        WaitForSingleObject(event, INFINITE);
    }
};

TEST_P(DX12RemoteRunTests, CheckRemoteTensorSharedBuf) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(tensor.get_shape()));

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
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(tensor.get_shape()));

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
    const auto output_byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(output_tensor.get_shape()));

    float* output_random_buffer_tensor = new float[output_byte_size / sizeof(float)];
    memset(output_random_buffer_tensor, 1, output_byte_size);
    ov::Tensor outputrandom_tensor_input{ov::element::f32, output_tensor.get_shape(), output_random_buffer_tensor};

    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(outputrandom_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    delete[] random_buffer_tensor;
}

TEST_P(DX12RemoteRunTests, CheckOutputDataFromMultipleRuns) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    float* data;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    auto shape = tensor.get_shape();
    const auto byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(shape));
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

}  // namespace behavior
}  // namespace test
}  // namespace ov

#    endif
#endif
