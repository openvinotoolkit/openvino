// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_OCL_RT

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>

#ifdef _WIN32
#ifdef ENABLE_DX11
#ifndef NOMINMAX
#define NOMINMAX
#define NOMINMAX_DEFINED_SHARED_BUF_TEST
#endif
#include <atlbase.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <dxgidebug.h>
#ifdef NOMINMAX_DEFINED_SHARED_BUF_TEST
#undef NOMINMAX
#undef NOMINMAX_DEFINED_SHARED_BUF_TEST
#endif
#endif
#endif

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

// Keep data unchanged while still forcing an explicit output tensor write path.
std::shared_ptr<ov::Model> make_copy_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});
    auto add = std::make_shared<ov::op::v1::Add>(param, zero);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

#ifdef _WIN32
#ifdef ENABLE_DX11

struct Dx12TestContext {
    CComPtr<IDXGIAdapter1> adapter;
    CComPtr<ID3D12Device> device;
    CComPtr<ID3D12CommandQueue> command_queue;
};

struct Dx12SharedBuffer {
    CComPtr<ID3D12Resource> resource;
    HANDLE shared_handle = nullptr;  // NT handle; caller must CloseHandle when done
};

// RAII DXGI debug scope: enables the D3D12 debug layer (must be constructed before
// any ID3D12Device is created), captures IDXGIInfoQueue messages, and on destruction
// flushes remaining messages and calls ReportLiveObjects.
struct DxgiDebugScope {
    CComPtr<IDXGIInfoQueue> info_queue;

    DxgiDebugScope() {
        // Enable D3D12 debug layer before device creation.
        CComPtr<ID3D12Debug> debug_ctrl;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug_ctrl))))
            debug_ctrl->EnableDebugLayer();

        DXGIGetDebugInterface1(0, IID_PPV_ARGS(&info_queue));
    }

    void flush(const char* label = "") const {
        if (!info_queue)
            return;
        const UINT64 count = info_queue->GetNumStoredMessages(DXGI_DEBUG_ALL);
        for (UINT64 i = 0; i < count; ++i) {
            SIZE_T msg_len = 0;
            info_queue->GetMessage(DXGI_DEBUG_ALL, i, nullptr, &msg_len);
            std::vector<char> buf(msg_len);
            auto* msg = reinterpret_cast<DXGI_INFO_QUEUE_MESSAGE*>(buf.data());
            if (SUCCEEDED(info_queue->GetMessage(DXGI_DEBUG_ALL, i, msg, &msg_len)))
                std::cout << "[DXGI" << (label[0] ? "|" : "") << label << "] " << msg->pDescription << "\n";
        }
        info_queue->ClearStoredMessages(DXGI_DEBUG_ALL);
    }

    ~DxgiDebugScope() {
        flush("teardown");
        CComPtr<IDXGIDebug1> dxgi_debug;
        if (SUCCEEDED(DXGIGetDebugInterface1(0, IID_PPV_ARGS(&dxgi_debug))))
            dxgi_debug->ReportLiveObjects(
                DXGI_DEBUG_ALL,
                static_cast<DXGI_DEBUG_RLO_FLAGS>(DXGI_DEBUG_RLO_SUMMARY | DXGI_DEBUG_RLO_IGNORE_INTERNAL));
    }
};

static bool gpu_wait(ID3D12CommandQueue* command_queue, ID3D12Device* device) {
    ID3D12Fence* raw_fence = nullptr;
    HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&raw_fence));
    if (FAILED(hr)) return false;
    CComPtr<ID3D12Fence> fence(raw_fence);

    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!event) return false;

    const UINT64 fence_value = 1;
    command_queue->Signal(fence, fence_value);
    if (fence->GetCompletedValue() < fence_value) {
        fence->SetEventOnCompletion(fence_value, event);
        WaitForSingleObject(event, INFINITE);
    }
    CloseHandle(event);
    return true;
}

Dx12TestContext create_dx12_test_context() {
    IDXGIFactory4* raw_factory = nullptr;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&raw_factory));
    EXPECT_FALSE(FAILED(hr));
    CComPtr<IDXGIFactory4> factory(raw_factory);
    if (!factory) return {};

    CComPtr<IDXGIAdapter1> intel_adapter;
    const UINT intel_vendor_id = 0x8086;
    UINT adapter_index = 0;
    IDXGIAdapter1* raw_adapter = nullptr;
    while (factory->EnumAdapters1(adapter_index, &raw_adapter) != DXGI_ERROR_NOT_FOUND) {
        CComPtr<IDXGIAdapter1> adapter(raw_adapter);
        DXGI_ADAPTER_DESC1 desc{};
        adapter->GetDesc1(&desc);
        if (desc.VendorId == intel_vendor_id && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
            intel_adapter = adapter;
            break;
        }
        ++adapter_index;
    }
    if (!intel_adapter) return {};

    ID3D12Device* raw_device = nullptr;
    hr = D3D12CreateDevice(intel_adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&raw_device));
    EXPECT_FALSE(FAILED(hr));
    if (FAILED(hr)) return {};
    CComPtr<ID3D12Device> device(raw_device);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* raw_queue = nullptr;
    hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&raw_queue));
    EXPECT_FALSE(FAILED(hr));

    return {intel_adapter, device, CComPtr<ID3D12CommandQueue>(raw_queue)};
}

Dx12SharedBuffer create_dx12_shared_buffer(ID3D12Device* device,
                                            ID3D12CommandQueue* command_queue,
                                            size_t byte_size,
                                            const void* data = nullptr) {
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_RESOURCE_DESC resource_desc{};
    resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resource_desc.Alignment = 0;
    resource_desc.Width = byte_size;
    resource_desc.Height = 1;
    resource_desc.DepthOrArraySize = 1;
    resource_desc.MipLevels = 1;
    resource_desc.Format = DXGI_FORMAT_UNKNOWN;
    resource_desc.SampleDesc.Count = 1;
    resource_desc.SampleDesc.Quality = 0;
    resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resource_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* raw_resource = nullptr;
    HRESULT hr = device->CreateCommittedResource(&heap_props,
                                                  D3D12_HEAP_FLAG_SHARED,
                                                  &resource_desc,
                                                  D3D12_RESOURCE_STATE_COMMON,
                                                  nullptr,
                                                  IID_PPV_ARGS(&raw_resource));
    EXPECT_FALSE(FAILED(hr));
    CComPtr<ID3D12Resource> resource(raw_resource);
    if (!resource) return {};

    HANDLE shared_handle = nullptr;
    hr = device->CreateSharedHandle(resource, nullptr, GENERIC_ALL, nullptr, &shared_handle);
    EXPECT_FALSE(FAILED(hr));
    EXPECT_NE(shared_handle, nullptr);

    if (data && resource) {
        D3D12_HEAP_PROPERTIES upload_heap{};
        upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC upload_desc = resource_desc;
        upload_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

        ID3D12Resource* raw_upload = nullptr;
        hr = device->CreateCommittedResource(&upload_heap,
                                              D3D12_HEAP_FLAG_NONE,
                                              &upload_desc,
                                              D3D12_RESOURCE_STATE_GENERIC_READ,
                                              nullptr,
                                              IID_PPV_ARGS(&raw_upload));
        EXPECT_FALSE(FAILED(hr));
        CComPtr<ID3D12Resource> upload_resource(raw_upload);

        if (upload_resource) {
            void* mapped = nullptr;
            D3D12_RANGE read_range{0, 0};
            upload_resource->Map(0, &read_range, &mapped);
            memcpy(mapped, data, byte_size);
            upload_resource->Unmap(0, nullptr);

            ID3D12CommandAllocator* raw_allocator = nullptr;
            device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&raw_allocator));
            CComPtr<ID3D12CommandAllocator> allocator(raw_allocator);

            ID3D12GraphicsCommandList* raw_cmd_list = nullptr;
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
                                      IID_PPV_ARGS(&raw_cmd_list));
            CComPtr<ID3D12GraphicsCommandList> cmd_list(raw_cmd_list);

            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            barrier.Transition.pResource = resource;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            cmd_list->ResourceBarrier(1, &barrier);

            cmd_list->CopyBufferRegion(resource, 0, upload_resource, 0, byte_size);

            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
            cmd_list->ResourceBarrier(1, &barrier);
            cmd_list->Close();

            ID3D12CommandList* cmd_lists[] = {cmd_list};
            command_queue->ExecuteCommandLists(1, cmd_lists);
            gpu_wait(command_queue, device);
        }
    }

    return {resource, shared_handle};
}

bool CopySharedResourceToFloatVector(ID3D12Device* device,
                                      ID3D12CommandQueue* command_queue,
                                      HANDLE shared_handle,
                                      std::vector<float>& out_data) {
    ID3D12Resource* raw_shared = nullptr;
    HRESULT hr = device->OpenSharedHandle(shared_handle, IID_PPV_ARGS(&raw_shared));
    if (FAILED(hr)) return false;
    CComPtr<ID3D12Resource> shared_resource(raw_shared);

    const UINT64 byte_size = shared_resource->GetDesc().Width;

    D3D12_HEAP_PROPERTIES readback_heap{};
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;

    D3D12_RESOURCE_DESC readback_desc{};
    readback_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readback_desc.Alignment = 0;
    readback_desc.Width = byte_size;
    readback_desc.Height = 1;
    readback_desc.DepthOrArraySize = 1;
    readback_desc.MipLevels = 1;
    readback_desc.Format = DXGI_FORMAT_UNKNOWN;
    readback_desc.SampleDesc.Count = 1;
    readback_desc.SampleDesc.Quality = 0;
    readback_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* raw_readback = nullptr;
    hr = device->CreateCommittedResource(&readback_heap,
                                          D3D12_HEAP_FLAG_NONE,
                                          &readback_desc,
                                          D3D12_RESOURCE_STATE_COPY_DEST,
                                          nullptr,
                                          IID_PPV_ARGS(&raw_readback));
    if (FAILED(hr)) return false;
    CComPtr<ID3D12Resource> readback_resource(raw_readback);

    ID3D12CommandAllocator* raw_allocator = nullptr;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&raw_allocator));
    CComPtr<ID3D12CommandAllocator> allocator(raw_allocator);

    ID3D12GraphicsCommandList* raw_cmd_list = nullptr;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
                                   IID_PPV_ARGS(&raw_cmd_list));
    if (FAILED(hr)) return false;
    CComPtr<ID3D12GraphicsCommandList> cmd_list(raw_cmd_list);

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = shared_resource;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd_list->ResourceBarrier(1, &barrier);

    cmd_list->CopyBufferRegion(readback_resource, 0, shared_resource, 0, byte_size);

    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    cmd_list->ResourceBarrier(1, &barrier);
    cmd_list->Close();

    ID3D12CommandList* cmd_lists[] = {cmd_list};
    command_queue->ExecuteCommandLists(1, cmd_lists);
    gpu_wait(command_queue, device);

    void* mapped = nullptr;
    D3D12_RANGE read_range{0, static_cast<SIZE_T>(byte_size)};
    hr = readback_resource->Map(0, &read_range, &mapped);
    if (FAILED(hr)) return false;

    out_data.resize(static_cast<size_t>(byte_size) / sizeof(float));
    memcpy(out_data.data(), mapped, static_cast<size_t>(byte_size));
    D3D12_RANGE write_range{0, 0};
    readback_resource->Unmap(0, &write_range);
    return true;
}

#endif  // ENABLE_DX11
#endif  // _WIN32

#ifdef _WIN32
#ifdef ENABLE_DX11

TEST(GpuSharedBufferRemoteTensor, smoke_Dx12RemoteInputToRemoteOutputCopyAndCompare) {
    DxgiDebugScope debug_scope;
    ov::Core core;
    const ov::Shape shape{16};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);
    auto dx12 = create_dx12_test_context();
    debug_scope.flush("after create_dx12_test_context");
    if (!dx12.device) {
        FAIL() << "No Intel DXGI adapter found or D3D12 device creation failed";
    }

    std::vector<float> input_init(element_count, 2.0f);
    auto dx_input_shared = create_dx12_shared_buffer(dx12.device, dx12.command_queue,
                                                      byte_size, input_init.data());
    auto dx_output_shared = create_dx12_shared_buffer(dx12.device, dx12.command_queue, byte_size);
    ASSERT_NE(dx_input_shared.shared_handle, nullptr);
    ASSERT_NE(dx_output_shared.shared_handle, nullptr);

    auto ov_ctx = core.create_context("GPU", {}).as<ov::intel_gpu::ocl::ClContext>();

    {
        auto params = ov_ctx.get_params();
        auto it = params.find(ov::intel_gpu::ocl_context.name());
        if (it == params.end()) {
            std::cout << "[INFO] GPU context does not expose ocl_context param\n";
            return;
        }
        auto cl_ctx = static_cast<cl_context>(it->second.as<ov::intel_gpu::ocl::gpu_handle_param>());
        size_t devices_size = 0;
        if (clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, 0, nullptr, &devices_size) != CL_SUCCESS || devices_size < sizeof(cl_device_id)) {
            std::cout << "[INFO] clGetContextInfo(CL_CONTEXT_DEVICES) failed\n";
            return;
        }
        std::vector<cl_device_id> cl_devices(devices_size / sizeof(cl_device_id));
        clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, devices_size, cl_devices.data(), nullptr);
        size_t ext_size = 0;
        clGetDeviceInfo(cl_devices[0], CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
        std::string extensions(ext_size, '\0');
        clGetDeviceInfo(cl_devices[0], CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);        while (!extensions.empty() && extensions.back() == '\0') extensions.pop_back();
        std::cout << "[INFO] CL extensions: [" << extensions << "]\n";
        if (extensions.find("cl_khr_external_memory") == std::string::npos) {
            std::cout << "[INFO] cl_khr_external_memory not supported\n";
            return;
        }
    }

    ov::RemoteTensor remote_input_tensor;
    ov::RemoteTensor remote_output_tensor;
    try {
        remote_input_tensor = ov_ctx.create_tensor(ov::element::f32, shape,
                                                   dx_input_shared.shared_handle,
                                                   ov::intel_gpu::MemType::SHARED_BUF);
        remote_output_tensor = ov_ctx.create_tensor(ov::element::f32, shape,
                                                    dx_output_shared.shared_handle,
                                                    ov::intel_gpu::MemType::SHARED_BUF);
    } catch (const ov::Exception& ex) {
        std::cout << "[INFO] NT handle import not supported on this device: " << ex.what() << "\n";
        return;
    }

    auto model = make_copy_model(shape);
    auto compiled = core.compile_model(model, ov_ctx);
    auto infer_req = compiled.create_infer_request();
    infer_req.set_tensor(compiled.input(), remote_input_tensor);
    infer_req.set_tensor(compiled.output(), remote_output_tensor);

    ov::Tensor host_input(ov::element::f32, shape);
    remote_input_tensor.copy_to(host_input);
    const auto* input_values = host_input.data<const float>();
    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(input_values[i], 2.0f) << "Input mismatch at index " << i;
    }

    infer_req.infer();
    debug_scope.flush("after infer");

    ov::Tensor host_output(ov::element::f32, shape);
    remote_output_tensor.copy_to(host_output);
    const auto* output_values = host_output.data<const float>();

    const bool has_non_zero = std::any_of(output_values, output_values + element_count, [](float v) {
        return v != 0.0f;
    });
    ASSERT_TRUE(has_non_zero)
        << "DX12 explicit remote output binding is not supported in this runtime/device configuration";

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_values[i], 2.0f) << "Mismatch at index " << i;
    }

    CloseHandle(dx_input_shared.shared_handle);
    dx_input_shared.shared_handle = nullptr;
    CloseHandle(dx_output_shared.shared_handle);
    dx_output_shared.shared_handle = nullptr;
}

TEST(GpuSharedBufferRemoteTensor, smoke_Dx12RemoteInputToRemoteOutputDirectHandleCompare) {
    DxgiDebugScope debug_scope;
    ov::Core core;
    const ov::Shape shape{16};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);
    auto dx12 = create_dx12_test_context();
    debug_scope.flush("after create_dx12_test_context");
    if (!dx12.device) {
        FAIL() << "No Intel DXGI adapter found or D3D12 device creation failed";
    }

    std::vector<float> input_init(element_count, 2.0f);
    auto dx_input_shared = create_dx12_shared_buffer(dx12.device, dx12.command_queue,
                                                      byte_size, input_init.data());
    auto dx_output_shared = create_dx12_shared_buffer(dx12.device, dx12.command_queue, byte_size);
    ASSERT_NE(dx_input_shared.shared_handle, nullptr);
    ASSERT_NE(dx_output_shared.shared_handle, nullptr);

    auto ov_ctx = core.create_context("GPU", {}).as<ov::intel_gpu::ocl::ClContext>();

    {
        auto params = ov_ctx.get_params();
        auto it = params.find(ov::intel_gpu::ocl_context.name());
        if (it == params.end()) {
            std::cout << "[INFO] GPU context does not expose ocl_context param\n";
            return;
        }
        auto cl_ctx = static_cast<cl_context>(it->second.as<ov::intel_gpu::ocl::gpu_handle_param>());
        size_t devices_size = 0;
        if (clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, 0, nullptr, &devices_size) != CL_SUCCESS || devices_size < sizeof(cl_device_id)) {
            std::cout << "[INFO] clGetContextInfo(CL_CONTEXT_DEVICES) failed\n";
            return;
        }
        std::vector<cl_device_id> cl_devices(devices_size / sizeof(cl_device_id));
        clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, devices_size, cl_devices.data(), nullptr);
        size_t ext_size = 0;
        clGetDeviceInfo(cl_devices[0], CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
        std::string extensions(ext_size, '\0');
        clGetDeviceInfo(cl_devices[0], CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
        while (!extensions.empty() && extensions.back() == '\0') extensions.pop_back();
        std::cout << "[INFO] CL extensions: [" << extensions << "]\n";
        if (extensions.find("cl_khr_external_memory_win32") == std::string::npos) {
            std::cout << "[INFO] cl_khr_external_memory_win32 not supported\n";
            return;
        }
    }

    {
        auto remote_input_tensor = ov_ctx.create_tensor(ov::element::f32, shape,
                                                         dx_input_shared.shared_handle,
                                                         ov::intel_gpu::MemType::SHARED_BUF);
        auto remote_output_tensor = ov_ctx.create_tensor(ov::element::f32, shape,
                                                          dx_output_shared.shared_handle,
                                                          ov::intel_gpu::MemType::SHARED_BUF);

        auto model = make_copy_model(shape);
        auto compiled = core.compile_model(model, ov_ctx);
        auto infer_req = compiled.create_infer_request();
        infer_req.set_tensor(compiled.input(), remote_input_tensor);
        infer_req.set_tensor(compiled.output(), remote_output_tensor);
        infer_req.infer();
        debug_scope.flush("after infer");
    }  // Release remote tensors, infer_req, and compiled model before reading DX12 buffer directly.

    std::vector<float> output_host;
    ASSERT_TRUE(CopySharedResourceToFloatVector(dx12.device, dx12.command_queue,
                                                 dx_output_shared.shared_handle, output_host))
        << "Failed to read DX12 shared buffer";
    ASSERT_EQ(output_host.size(), element_count);

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_host[i], 2.0f) << "Mismatch at index " << i;
    }

    CloseHandle(dx_input_shared.shared_handle);
    dx_input_shared.shared_handle = nullptr;
    CloseHandle(dx_output_shared.shared_handle);
    dx_output_shared.shared_handle = nullptr;
}

#endif  // ENABLE_DX11
#endif  // _WIN32

}  // namespace

#endif  // OV_GPU_WITH_OCL_RT
