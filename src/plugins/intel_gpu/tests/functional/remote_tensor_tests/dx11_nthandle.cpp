// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_OCL_RT

#include <array>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <gtest/gtest.h>
#include <chrono>
#include <sstream>
#include <vector>
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
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

constexpr size_t kDx11SharedBufferAlignment = 16;

size_t align_to(size_t size, size_t alignment) {
    return (size % alignment == 0) ? size : size - (size % alignment) + alignment;
}

std::string format_luid_bytes(const unsigned char* data, size_t size) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (size_t index = 0; index < size; ++index) {
        stream << std::setw(2) << static_cast<unsigned int>(data[index]);
    }
    return stream.str();
}

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
        EXPECT_FALSE(FAILED(hr));
        if (FAILED(hr)) {
            return {};
        }

        return {CComPtr<ID3D11Device>(raw_device), CComPtr<ID3D11DeviceContext>(raw_ctx)};
    }

    return {};
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

    // Declare GPU device number
    const std::string selected_gpu_id = "0";
    const std::string selected_gpu_device = "GPU." + selected_gpu_id;
    std::cout << "[INFO] Selected GPU device: " << selected_gpu_device << "\n";

    // Get OpenCL context for the selected GPU
    auto candidate_ctx = core.get_default_context(selected_gpu_device).as<ov::intel_gpu::ocl::ClContext>();
    auto params = candidate_ctx.get_params();
    auto it = params.find(ov::intel_gpu::ocl_context.name());
    if (it == params.end()) {
        FAIL() << "Failed to get OpenCL context for " << selected_gpu_device;
    }

    // Extract LUID from OpenCL context
    auto cl_ctx = static_cast<cl_context>(it->second.as<ov::intel_gpu::ocl::gpu_handle_param>());
    std::array<unsigned char, CL_LUID_SIZE_KHR> cl_luid{};
    if (!get_context_device_luid(cl_ctx, cl_luid)) {
        FAIL() << "Failed to get LUID for " << selected_gpu_device;
    }

    std::cout << "[INFO] " << selected_gpu_device << " OpenCL LUID: "
              << format_luid_bytes(cl_luid.data(), cl_luid.size()) << "\n";

    // Create DX11 context for the selected GPU's LUID
    Dx11TestContext dx11 = create_dx11_test_context(cl_luid);
    if (!dx11.device) {
        FAIL() << "Failed to create DX11 context for " << selected_gpu_device;
    }

    std::vector<float> input_init(element_count, 2.0f);
    auto dx_input_shared = create_dx11_shared_buffer(dx11.device, byte_size, input_init.data());
    NtHandleGuard input_handle_guard{dx_input_shared.shared_handle};
    std::vector<float> output_init(element_count, 0.0f);
    auto dx_output_shared = create_dx11_shared_buffer(dx11.device, byte_size, output_init.data());
    NtHandleGuard output_handle_guard{dx_output_shared.shared_handle};

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



#endif  // ENABLE_DX11
#endif  // _WIN32

}  // namespace

#endif  // OV_GPU_WITH_OCL_RT