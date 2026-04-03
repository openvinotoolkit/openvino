// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_OCL_RT

#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <chrono>
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

    return {shared_buffer, shared_handle};
}

struct Dx11SharedTexture {
    CComPtr<ID3D11Texture2D> texture;
    HANDLE nt_handle = nullptr;
};

// Creates a 1-row R32_FLOAT ID3D11Texture2D backed by a Windows NT kernel handle.
// D3D11_RESOURCE_MISC_SHARED_NTHANDLE is valid for ID3D11Texture2D (unlike ID3D11Buffer).
// NT handles must be CloseHandle'd by the caller.
Dx11SharedTexture create_dx11_nt_shared_texture(ID3D11Device* device,
                                                UINT element_count,
                                                const float* data = nullptr) {
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = element_count;
    desc.Height = 1;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

    D3D11_SUBRESOURCE_DATA init_data{};
    init_data.pSysMem = data;
    init_data.SysMemPitch = element_count * sizeof(float);

    ID3D11Texture2D* raw_tex = nullptr;
    HRESULT hr = device->CreateTexture2D(&desc, data ? &init_data : nullptr, &raw_tex);
    if (FAILED(hr)) {
        return {};
    }
    CComPtr<ID3D11Texture2D> texture(raw_tex);

    CComPtr<IDXGIResource1> dxgi_resource1;
    hr = texture->QueryInterface(__uuidof(IDXGIResource1), reinterpret_cast<void**>(&dxgi_resource1));
    EXPECT_FALSE(FAILED(hr));
    if (!dxgi_resource1) return {};

    HANDLE nt_handle = nullptr;
    hr = dxgi_resource1->CreateSharedHandle(
        nullptr,
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
        nullptr,
        &nt_handle);
    EXPECT_FALSE(FAILED(hr));
    EXPECT_NE(nt_handle, nullptr);

    return {texture, nt_handle};
}

CComPtr<ID3D11Texture2D> open_dx11_nt_shared_texture(ID3D11Device* device, HANDLE nt_handle) {
    CComPtr<ID3D11Device1> device1;
    HRESULT hr = device->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&device1));
    EXPECT_FALSE(FAILED(hr));
    if (!device1) return {};

    ID3D11Texture2D* raw_tex = nullptr;
    hr = device1->OpenSharedResource1(nt_handle, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&raw_tex));
    EXPECT_FALSE(FAILED(hr));
    return CComPtr<ID3D11Texture2D>(raw_tex);
}

CComPtr<ID3D11Buffer> open_dx11_shared_buffer(ID3D11Device* device, HANDLE shared_handle) {
    ID3D11Buffer* raw_opened_buffer = nullptr;
    HRESULT hr = device->OpenSharedResource(shared_handle,
                                            __uuidof(ID3D11Buffer),
                                            reinterpret_cast<void**>(&raw_opened_buffer));
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
                                                   dx_input_shared.shared_handle);
    ASSERT_NE(dx_input_buffer, nullptr);

    auto dx_output_buffer = open_dx11_shared_buffer(dx11.device,
                                                    dx_output_shared.shared_handle);
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



// Tests the Windows NT kernel handle (IDXGIResource1::CreateSharedHandle) round-trip on a
// DXGI_FORMAT_R32_FLOAT ID3D11Texture2D.  D3D11_RESOURCE_MISC_SHARED_NTHANDLE is only valid
// for 2D surfaces, never for ID3D11Buffer (CREATEBUFFER_INVALIDMISCFLAGS error #68).
// The test verifies:
//   1. NT handle creation succeeds on a Texture2D.
//   2. Data written at creation time is readable back via the re-opened NT handle.
//   3. The NT handle remains valid and must be explicitly CloseHandle'd.
// OpenVINO inference through NT-handle-backed resources is architecturally unsupported because
// the GPU plugin's DX_BUFFER/clCreateFromD3D11BufferKHR path requires ID3D11Buffer (no NT
// handles), while the VA_SURFACE/clCreateFromD3D11Texture2DKHR path requires is_image_2d()
// layout (NV12/video formats, not float32).  Inference correctness with DX shared buffers is
// covered by smoke_Dx11RemoteInputToRemoteOutputCopyAndCompare.
TEST(GpuSharedBufferRemoteTensor11, smoke_Dx11NtHandleTexture2DRoundTrip) {
    const size_t element_count = 16;
    const size_t byte_size = element_count * sizeof(float);
    auto dx11 = create_dx11_test_context();
    if (!dx11.device) {
        FAIL() << "No Intel DXGI adapter found";
    }

    std::vector<float> input_data(element_count);
    for (size_t i = 0; i < element_count; ++i) input_data[i] = static_cast<float>(i) + 1.0f;

    // Create the shared texture (NT handle).
    auto shared_tex = create_dx11_nt_shared_texture(dx11.device,
                                                    static_cast<UINT>(element_count),
                                                    input_data.data());
    if (!shared_tex.nt_handle) {
        GTEST_SKIP_("NT handle creation for ID3D11Texture2D failed on this driver");
    }

    // Open the texture via its NT handle (simulates cross-device / cross-process access).
    auto opened_tex = open_dx11_nt_shared_texture(dx11.device, shared_tex.nt_handle);
    ASSERT_NE(opened_tex, nullptr) << "OpenSharedResource1 failed for NT handle";

    // Create a CPU-readable staging texture and copy the shared texture into it.
    D3D11_TEXTURE2D_DESC staging_desc{};
    opened_tex->GetDesc(&staging_desc);
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;

    ID3D11Texture2D* raw_staging = nullptr;
    HRESULT hr = dx11.device->CreateTexture2D(&staging_desc, nullptr, &raw_staging);
    ASSERT_FALSE(FAILED(hr)) << "Failed to create staging texture";
    CComPtr<ID3D11Texture2D> staging(raw_staging);

    dx11.device_ctx->CopyResource(staging, opened_tex);

    // GPU sync via D3D11 event query.
    D3D11_QUERY_DESC query_desc = {D3D11_QUERY_EVENT, 0};
    CComPtr<ID3D11Query> query;
    dx11.device->CreateQuery(&query_desc, &query);
    dx11.device_ctx->End(query);
    while (dx11.device_ctx->GetData(query, nullptr, 0, 0) == S_FALSE) {}

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = dx11.device_ctx->Map(staging, 0, D3D11_MAP_READ, 0, &mapped);
    ASSERT_FALSE(FAILED(hr)) << "Failed to map staging texture";

    std::vector<float> readback(element_count, 0.0f);
    SIZE_T bytesRead = 0;
    BOOL ok = ReadProcessMemory(GetCurrentProcess(),
                                mapped.pData,
                                readback.data(),
                                byte_size,
                                &bytesRead);
    if (ok) {
        std::cout << "Odczytano wartosc[0]: " << readback[0]
                  << " Liczba odczytanych bajtow: " << bytesRead << std::endl;
    } else {
        ADD_FAILURE() << "ReadProcessMemory zawiodl. Blad: " << GetLastError();
    }
    dx11.device_ctx->Unmap(staging, 0);

    // NT handles must be closed by the caller (unlike legacy DXGI handles).
    CloseHandle(shared_tex.nt_handle);

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(readback[i], input_data[i]) << "NT handle data mismatch at index " << i;
    }
}

#endif  // ENABLE_DX11
#endif  // _WIN32

}  // namespace

#endif  // OV_GPU_WITH_OCL_RT