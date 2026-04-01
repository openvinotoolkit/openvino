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
#include <d3d11.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#ifdef NOMINMAX_DEFINED_SHARED_BUF_TEST
#undef NOMINMAX
#undef NOMINMAX_DEFINED_SHARED_BUF_TEST
#endif
#endif
#endif

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/dx.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

constexpr size_t kDx11SharedBufferAlignment = 16;

size_t align_to(size_t size, size_t alignment) {
    return (size % alignment == 0) ? size : size - (size % alignment) + alignment;
}

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
struct Dx11TestContext {
    CComPtr<ID3D11Device> device;
    CComPtr<ID3D11DeviceContext> device_ctx;
};

struct Dx11SharedBuffer {
    CComPtr<ID3D11Buffer> buffer;
    HANDLE shared_handle = nullptr;
    bool is_nt_handle = false;
};

Dx11TestContext create_dx11_test_context() {
    IDXGIFactory* raw_factory = nullptr;
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&raw_factory));
    EXPECT_FALSE(FAILED(hr));
    CComPtr<IDXGIFactory> factory(raw_factory);

    CComPtr<IDXGIAdapter> intel_adapter;
    const unsigned int ref_intel_vendor_id = 0x8086;
    UINT adapter_index = 0;
    IDXGIAdapter* raw_adapter = nullptr;
    while (factory->EnumAdapters(adapter_index, &raw_adapter) != DXGI_ERROR_NOT_FOUND) {
        CComPtr<IDXGIAdapter> adapter(raw_adapter);
        DXGI_ADAPTER_DESC desc{};
        adapter->GetDesc(&desc);
        if (desc.VendorId == ref_intel_vendor_id) {
            intel_adapter = adapter;
            break;
        }
        ++adapter_index;
    }

    if (!intel_adapter) {
        return {};
    }

    D3D_FEATURE_LEVEL feature_levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL feature_level;
    ID3D11Device* raw_device = nullptr;
    ID3D11DeviceContext* raw_ctx = nullptr;
    hr = D3D11CreateDevice(intel_adapter,
                           D3D_DRIVER_TYPE_UNKNOWN,
                           nullptr,
                           0,
                           feature_levels,
                           ARRAYSIZE(feature_levels),
                           D3D11_SDK_VERSION,
                           &raw_device,
                           &feature_level,
                           &raw_ctx);
    EXPECT_FALSE(FAILED(hr));

    return {CComPtr<ID3D11Device>(raw_device), CComPtr<ID3D11DeviceContext>(raw_ctx)};
}

Dx11SharedBuffer create_dx11_shared_buffer(ID3D11Device* device, size_t byte_size, const void* data = nullptr) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(align_to(byte_size, kDx11SharedBufferAlignment));
    desc.Usage = D3D11_USAGE_DEFAULT;
    // Keep UAV-capable shared buffer; CPU writes are done via UpdateSubresource.
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    D3D11_SUBRESOURCE_DATA init_data{};
    init_data.pSysMem = data;

    ID3D11Buffer* raw_buffer = nullptr;
    HRESULT hr = device->CreateBuffer(&desc, data ? &init_data : nullptr, &raw_buffer);
    EXPECT_FALSE(FAILED(hr));
    CComPtr<ID3D11Buffer> shared_buffer(raw_buffer);

    HANDLE shared_handle = nullptr;
    CComPtr<IDXGIResource> dxgi_resource;
    hr = shared_buffer->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void**>(&dxgi_resource));
    EXPECT_FALSE(FAILED(hr));
    if (dxgi_resource) {
        hr = dxgi_resource->GetSharedHandle(&shared_handle);
    }
    EXPECT_FALSE(FAILED(hr));
    EXPECT_NE(shared_handle, nullptr);

    return {shared_buffer, shared_handle, false};
}

CComPtr<ID3D11Buffer> open_dx11_shared_buffer(ID3D11Device* device, HANDLE shared_handle, bool is_nt_handle) {
    ID3D11Buffer* raw_opened_buffer = nullptr;
    HRESULT hr = E_FAIL;

    if (is_nt_handle) {
        CComPtr<ID3D11Device1> device1;
        hr = device->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&device1));
        EXPECT_FALSE(FAILED(hr));
        if (!FAILED(hr) && device1) {
            hr = device1->OpenSharedResource1(shared_handle,
                                              __uuidof(ID3D11Buffer),
                                              reinterpret_cast<void**>(&raw_opened_buffer));
        }
    } else {
        hr = device->OpenSharedResource(shared_handle,
                                        __uuidof(ID3D11Buffer),
                                        reinterpret_cast<void**>(&raw_opened_buffer));
    }

    EXPECT_FALSE(FAILED(hr));
    return CComPtr<ID3D11Buffer>(raw_opened_buffer);
}
#endif
#endif

#ifdef _WIN32
#ifdef ENABLE_DX11

TEST(GpuSharedBufferRemoteTensor, smoke_Dx11RemoteInputToRemoteOutputCopyAndCompare) {
    ov::Core core;
    const ov::Shape shape{16};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);
    auto dx11 = create_dx11_test_context();
    if (!dx11.device) {
        FAIL() << "No Intel DXGI adapter found";
    }

    std::vector<float> input_init(element_count, 2.0f);
    auto dx_input_shared = create_dx11_shared_buffer(dx11.device, byte_size, input_init.data());
    auto dx_output_shared = create_dx11_shared_buffer(dx11.device, byte_size);

    auto dx_input_buffer = open_dx11_shared_buffer(dx11.device,
                                                   dx_input_shared.shared_handle,
                                                   dx_input_shared.is_nt_handle);
    ASSERT_NE(dx_input_buffer, nullptr);

    auto dx_output_buffer = open_dx11_shared_buffer(dx11.device,
                                                    dx_output_shared.shared_handle,
                                                    dx_output_shared.is_nt_handle);
    ASSERT_NE(dx_output_buffer, nullptr);

    // Initialize opened shared input buffer explicitly to avoid driver-dependent init visibility.
    dx11.device_ctx->UpdateSubresource(dx_input_buffer,
                                       0,
                                       nullptr,
                                       input_init.data(),
                                       static_cast<UINT>(byte_size),
                                       0);
    dx11.device_ctx->Flush();

    auto d3d_ctx = ov::intel_gpu::ocl::D3DContext(core, dx11.device);

    auto remote_input_tensor = d3d_ctx.create_tensor(ov::element::f32, shape, dx_input_buffer);
    auto remote_output_tensor = d3d_ctx.create_tensor(ov::element::f32, shape, dx_output_buffer);

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

    const bool has_non_zero = std::any_of(output_values, output_values + element_count, [](float v) {
        return v != 0.0f;
    });
    ASSERT_TRUE(has_non_zero)
        << "DX11 explicit remote output binding is not supported in this runtime/device configuration";

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_values[i], 2.0f) << "Mismatch at index " << i;
    }

}

TEST(GpuSharedBufferRemoteTensor, smoke_Dx11RemoteInputToRemoteOutputDirectHandleCompare) {
    ov::Core core;
    const ov::Shape shape{16};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);
    auto dx11 = create_dx11_test_context();
    if (!dx11.device) {
        FAIL() << "No Intel DXGI adapter found";
    }

    std::vector<float> input_init(element_count, 2.0f);
    auto dx_input_shared = create_dx11_shared_buffer(dx11.device, byte_size, input_init.data());
    auto dx_output_shared = create_dx11_shared_buffer(dx11.device, byte_size);

    auto dx_input_buffer = open_dx11_shared_buffer(dx11.device,
                                                   dx_input_shared.shared_handle,
                                                   dx_input_shared.is_nt_handle);
    ASSERT_NE(dx_input_buffer, nullptr);

    auto dx_output_buffer = open_dx11_shared_buffer(dx11.device,
                                                    dx_output_shared.shared_handle,
                                                    dx_output_shared.is_nt_handle);
    ASSERT_NE(dx_output_buffer, nullptr);

    // Initialize opened shared input buffer explicitly to avoid driver-dependent init visibility.
    dx11.device_ctx->UpdateSubresource(dx_input_buffer,
                                       0,
                                       nullptr,
                                       input_init.data(),
                                       static_cast<UINT>(byte_size),
                                       0);
    dx11.device_ctx->Flush();

    auto d3d_ctx = ov::intel_gpu::ocl::D3DContext(core, dx11.device);

    {
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
        infer_req.infer();
    }  // Release remote tensors, infer_req, and compiled model before reading DX11 buffer directly.

    // Read output directly from DX11 handle without using ov::Tensor copy.
    // DEFAULT buffers are not CPU-mappable, so copy into a staging buffer then map.
    std::vector<float> output_host(element_count);
    D3D11_BUFFER_DESC staging_desc = {};
    dx_output_buffer->GetDesc(&staging_desc);
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    CComPtr<ID3D11Buffer> staging_buffer;
    ID3D11Buffer* raw_staging_buffer = nullptr;
    HRESULT hr_staging = dx11.device->CreateBuffer(&staging_desc, nullptr, &raw_staging_buffer);
    ASSERT_FALSE(FAILED(hr_staging)) << "Failed to create staging buffer";
    staging_buffer = raw_staging_buffer;

    dx11.device_ctx->CopyResource(staging_buffer, dx_output_buffer);
    dx11.device_ctx->Flush();
    // Bardziej niezawodny sposób na upewnienie się, że GPU skończyło kopiowanie
    D3D11_QUERY_DESC queryDesc = { D3D11_QUERY_EVENT, 0 };
    CComPtr<ID3D11Query> query;
    dx11.device->CreateQuery(&queryDesc, &query);
    dx11.device_ctx->End(query);
    while (dx11.device_ctx->GetData(query, NULL, 0, 0) == S_FALSE) { /* Wait */ }
    D3D11_MAPPED_SUBRESOURCE staging_mapped = {};
    HRESULT hr_map = dx11.device_ctx->Map(staging_buffer, 0, D3D11_MAP_READ, 0, &staging_mapped);
    ASSERT_FALSE(FAILED(hr_map)) << "Failed to map staging buffer";

    memcpy(output_host.data(), staging_mapped.pData, byte_size);
    dx11.device_ctx->Unmap(staging_buffer, 0);

    const float* readback_values = output_host.data();

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(readback_values[i], 2.0f) << "Mismatch at index " << i;
    }
}

#endif  // ENABLE_DX11
#endif  // _WIN32

}  // namespace

#endif  // OV_GPU_WITH_OCL_RT
