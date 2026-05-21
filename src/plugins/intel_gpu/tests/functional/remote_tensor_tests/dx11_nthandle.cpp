// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#if defined(OV_GPU_WITH_OCL_RT) && defined(_WIN32) && defined(ENABLE_DX11)
#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#define NOMINMAX_DEFINED_SHARED_BUF_TEST
#endif
#include <atlbase.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#ifdef NOMINMAX_DEFINED_SHARED_BUF_TEST
#undef NOMINMAX
#undef NOMINMAX_DEFINED_SHARED_BUF_TEST
#endif
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/dx.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {
bool get_context_device_luid(cl_context cl_ctx, std::array<unsigned char, CL_LUID_SIZE_KHR>& cl_luid) {
    size_t devices_size = 0;
    if (clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, 0, nullptr, &devices_size) != CL_SUCCESS ||
        devices_size < sizeof(cl_device_id)) {
        return false;
    }

    std::vector<cl_device_id> cl_devices(devices_size / sizeof(cl_device_id));
    if (clGetContextInfo(cl_ctx, CL_CONTEXT_DEVICES, devices_size, cl_devices.data(), nullptr) != CL_SUCCESS ||
        cl_devices.empty()) {
        return false;
    }

    cl_bool cl_luid_valid = CL_FALSE;
    if (clGetDeviceInfo(cl_devices[0], CL_DEVICE_LUID_VALID_KHR, sizeof(cl_luid_valid), &cl_luid_valid, nullptr) !=
            CL_SUCCESS ||
        cl_luid_valid != CL_TRUE) {
        return false;
    }

    return clGetDeviceInfo(cl_devices[0], CL_DEVICE_LUID_KHR, cl_luid.size(), cl_luid.data(), nullptr) == CL_SUCCESS;
}

// Keep data unchanged while still forcing an explicit output tensor write path.
std::shared_ptr<ov::Model> make_copy_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});
    auto add = std::make_shared<ov::op::v1::Add>(param, zero);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}


struct Dx11TestContext {
    CComPtr<ID3D11Device> device;
    CComPtr<ID3D11DeviceContext> device_ctx;
};

struct Dx11SharedBuffer {
    CComPtr<ID3D11Texture2D> buffer;
    HANDLE shared_handle = nullptr;
};

void close_nt_handle(HANDLE& handle) {
    if (handle != nullptr) {
        CloseHandle(handle);
        handle = nullptr;
    }
}

struct NtHandleGuard {
    HANDLE& handle;

    ~NtHandleGuard() {
        close_nt_handle(handle);
    }
};

Dx11TestContext create_dx11_test_context(const std::array<unsigned char, CL_LUID_SIZE_KHR>& target_luid) {
    IDXGIFactory* raw_factory = nullptr;
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&raw_factory));
    EXPECT_FALSE(FAILED(hr));
    if (FAILED(hr)) {
        return {};
    }
    CComPtr<IDXGIFactory> factory(raw_factory);
    if (!factory) {
        return {};
    }

    UINT adapter_index = 0;
    IDXGIAdapter* raw_adapter = nullptr;
    while (factory->EnumAdapters(adapter_index, &raw_adapter) != DXGI_ERROR_NOT_FOUND) {
        CComPtr<IDXGIAdapter> adapter(raw_adapter);
        DXGI_ADAPTER_DESC desc{};
        adapter->GetDesc(&desc);

        std::array<unsigned char, CL_LUID_SIZE_KHR> adapter_luid{};
        memcpy(adapter_luid.data(), &desc.AdapterLuid, sizeof(desc.AdapterLuid));
        if (memcmp(adapter_luid.data(), target_luid.data(), target_luid.size()) != 0) {
            ++adapter_index;
            continue;
        }

        D3D_FEATURE_LEVEL feature_levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL feature_level;
        ID3D11Device* raw_device = nullptr;
        ID3D11DeviceContext* raw_ctx = nullptr;
        hr = D3D11CreateDevice(adapter,
                               D3D_DRIVER_TYPE_UNKNOWN,
                               nullptr,
                               0,
                               feature_levels,
                               ARRAYSIZE(feature_levels),
                               D3D11_SDK_VERSION,
                               &raw_device,
                               &feature_level,
                               &raw_ctx);
        if (FAILED(hr)) {
            return {};
        }

        return {CComPtr<ID3D11Device>(raw_device), CComPtr<ID3D11DeviceContext>(raw_ctx)};
    }

    return {};
}

Dx11SharedBuffer create_dx11_shared_buffer(ID3D11Device* device, size_t byte_size, const void* data = nullptr) {
    // D3D11 does not allow SHARED_NTHANDLE on ID3D11Buffer; use an R32_FLOAT 4x4 Texture2D as backing storage.
    const UINT element_count = static_cast<UINT>(byte_size / sizeof(float));
    const UINT tex_width = 4;
    const UINT tex_height = element_count / tex_width;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = tex_width;
    desc.Height = tex_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    // Keep UAV-capable shared buffer; CPU writes are done via UpdateSubresource.
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;

    D3D11_SUBRESOURCE_DATA init_data{};
    init_data.pSysMem = data;
    init_data.SysMemPitch = tex_width * static_cast<UINT>(sizeof(float));
    init_data.SysMemSlicePitch = init_data.SysMemPitch * tex_height;

    ID3D11Texture2D* raw_texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&desc, data ? &init_data : nullptr, &raw_texture);

    if (FAILED(hr)) {
        return {};
    }
    CComPtr<ID3D11Texture2D> shared_texture(raw_texture);

    HANDLE shared_handle = nullptr;
    CComPtr<IDXGIResource1> dxgi_resource;
    hr = shared_texture->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void**>(&dxgi_resource));
    if (FAILED(hr)) {
        return {};
    }
    if (dxgi_resource) {
        hr = dxgi_resource->CreateSharedHandle(nullptr,
                                               DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                               nullptr,
                                               &shared_handle);
    }
    if (FAILED(hr)) {
        return {};
    }
    if (shared_handle == nullptr) {
        return {};
    }

    return {shared_texture, shared_handle};
}

CComPtr<ID3D11Texture2D> open_dx11_shared_buffer(ID3D11Device* device, HANDLE shared_handle) {
    CComPtr<ID3D11Device1> device1;
    HRESULT hr = device->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&device1));
    EXPECT_FALSE(FAILED(hr));
    ID3D11Texture2D* raw_opened_texture = nullptr;
    hr = device1->OpenSharedResource1(shared_handle,
                                      __uuidof(ID3D11Texture2D),
                                      reinterpret_cast<void**>(&raw_opened_texture));
    if(FAILED(hr)) {
        return {};
    }
    return CComPtr<ID3D11Texture2D>(raw_opened_texture);
}

TEST(GpuSharedBufferRemoteTensor, smoke_Dx11RemoteInputToRemoteOutputCopyAndCompare) {
#ifndef CL_VERSION_3_0
    GTEST_SKIP() << "OpenCL version 3.0 is required for external memory sharing"; 
#endif
    //test work on 32.101.7076 - not tried with older driver
    ov::Core core;
    const ov::Shape shape{16};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);

    // Declare GPU device number
    const std::string selected_gpu_id = "0";
    const std::string selected_gpu_device = "GPU." + selected_gpu_id;

    // Get OpenCL context for the selected GPU
    auto candidate_ctx = core.get_default_context(selected_gpu_device).as<ov::intel_gpu::ocl::ClContext>();
    auto params = candidate_ctx.get_params();
    auto it = params.find(ov::intel_gpu::ocl_context.name());
    if (it == params.end()) {
        GTEST_SKIP() << "Failed to get OpenCL context for " << selected_gpu_device;
    }

    // Extract LUID from OpenCL context
    auto cl_ctx = static_cast<cl_context>(it->second.as<ov::intel_gpu::ocl::gpu_handle_param>());
    std::array<unsigned char, CL_LUID_SIZE_KHR> cl_luid{};
    if (!get_context_device_luid(cl_ctx, cl_luid)) {
        GTEST_SKIP() << "Failed to get LUID for " << selected_gpu_device;
    }

    // Create DX11 context for the selected GPU's LUID
    Dx11TestContext dx11 = create_dx11_test_context(cl_luid);
    if (!dx11.device) {
        GTEST_SKIP() << "Failed to create DX11 context for " << selected_gpu_device;
    }

    std::vector<float> input_init(element_count, 2.0f);
    auto dx_input_shared = create_dx11_shared_buffer(dx11.device, byte_size, input_init.data());
    NtHandleGuard input_handle_guard{dx_input_shared.shared_handle};
    std::vector<float> output_init(element_count, 0.0f);
    auto dx_output_shared = create_dx11_shared_buffer(dx11.device, byte_size, output_init.data());
    NtHandleGuard output_handle_guard{dx_output_shared.shared_handle};

    auto dx_input_buffer = open_dx11_shared_buffer(dx11.device,
                                                   dx_input_shared.shared_handle);
    if (dx_input_buffer == nullptr) {
        GTEST_SKIP() << "Failed to open shared input buffer in DX11 context for " << selected_gpu_device;
    }

    auto dx_output_buffer = open_dx11_shared_buffer(dx11.device,
                                                    dx_output_shared.shared_handle);
    if (dx_output_buffer == nullptr) {
        GTEST_SKIP() << "Failed to open shared output buffer in DX11 context for " << selected_gpu_device;
    }

    // Initialize opened shared input texture explicitly to avoid driver-dependent init visibility.
    const UINT row_pitch = 4u * static_cast<UINT>(sizeof(float));  // 4 floats per row
    dx11.device_ctx->UpdateSubresource(dx_input_buffer,
                                       0,
                                       nullptr,
                                       input_init.data(),
                                       row_pitch,
                                       static_cast<UINT>(byte_size));
    dx11.device_ctx->Flush();

    auto d3d_ctx = ov::intel_gpu::ocl::D3DContext(core, dx11.device);

    auto remote_input_tensor = d3d_ctx.create_tensor(ov::element::f32,
                                                     shape,
                                                     dx_input_shared.shared_handle,
                                                     ov::intel_gpu::MemType::SHARED_BUF);
    auto remote_output_tensor = d3d_ctx.create_tensor(ov::element::f32,
                                                      shape,
                                                      dx_output_shared.shared_handle,
                                                      ov::intel_gpu::MemType::SHARED_BUF);

    auto model = make_copy_model(shape);
    auto compiled = core.compile_model(model, d3d_ctx);
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
    ov::Tensor host_output(ov::element::f32, shape);
    remote_output_tensor.copy_to(host_output);
    const auto* output_values = host_output.data<const float>();
    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_values[i], 2.0f) << "Mismatch at index " << i;
    }
}
}  // namespace
#endif
