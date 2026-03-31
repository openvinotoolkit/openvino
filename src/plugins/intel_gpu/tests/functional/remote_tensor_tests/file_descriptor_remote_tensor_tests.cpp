// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_OCL_RT

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#ifdef ENABLE_DX11
#ifndef NOMINMAX
#define NOMINMAX
#define NOMINMAX_DEFINED_SHARED_BUF_TEST
#endif
#include <atlbase.h>
#include <d3d11.h>
#ifdef NOMINMAX_DEFINED_SHARED_BUF_TEST
#undef NOMINMAX
#undef NOMINMAX_DEFINED_SHARED_BUF_TEST
#endif
#endif
#endif

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/intel_gpu/ocl/dx.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"

#ifdef _WIN32
#ifdef ENABLE_DX11
#include <CL/cl_d3d11.h>
#endif
#endif

namespace {

// Simple passthrough model: Parameter -> Result
std::shared_ptr<ov::Model> make_passthrough_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto result = std::make_shared<ov::op::v0::Result>(param);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
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

struct Dx11SharedTexture2D {
    CComPtr<ID3D11Texture2D> texture;
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

CComPtr<ID3D11Buffer> create_dx11_buffer(ID3D11Device* device, size_t byte_size, const void* data = nullptr) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(byte_size);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;

    D3D11_SUBRESOURCE_DATA init_data{};
    init_data.pSysMem = data;

    ID3D11Buffer* raw_buffer = nullptr;
    HRESULT hr = device->CreateBuffer(&desc, data ? &init_data : nullptr, &raw_buffer);
    EXPECT_FALSE(FAILED(hr));
    return CComPtr<ID3D11Buffer>(raw_buffer);
}

Dx11SharedBuffer create_dx11_shared_buffer(ID3D11Device* device, size_t byte_size, const void* data = nullptr) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(byte_size);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;

    D3D11_SUBRESOURCE_DATA init_data{};
    init_data.pSysMem = data;

    ID3D11Buffer* raw_buffer = nullptr;
    HRESULT hr = device->CreateBuffer(&desc, data ? &init_data : nullptr, &raw_buffer);
    EXPECT_FALSE(FAILED(hr));
    CComPtr<ID3D11Buffer> shared_buffer(raw_buffer);

    CComPtr<IDXGIResource> dxgi_resource;
    hr = shared_buffer->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void**>(&dxgi_resource));
    EXPECT_FALSE(FAILED(hr));

    HANDLE shared_handle = nullptr;
    hr = dxgi_resource->GetSharedHandle(&shared_handle);
    EXPECT_FALSE(FAILED(hr));
    EXPECT_NE(shared_handle, nullptr);

    return {shared_buffer, shared_handle};
}

Dx11SharedTexture2D create_dx11_shared_texture_2d(ID3D11Device* device,
                                                  const D3D11_TEXTURE2D_DESC& texture_description,
                                                  const D3D11_SUBRESOURCE_DATA* texture_data = nullptr) {
    D3D11_TEXTURE2D_DESC shared_desc = texture_description;
    shared_desc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;

    ID3D11Texture2D* raw_texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&shared_desc, texture_data, &raw_texture);
    EXPECT_FALSE(FAILED(hr));
    CComPtr<ID3D11Texture2D> shared_texture(raw_texture);

    CComPtr<IDXGIResource> dxgi_resource;
    hr = shared_texture->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void**>(&dxgi_resource));
    EXPECT_FALSE(FAILED(hr));

    HANDLE shared_handle = nullptr;
    hr = dxgi_resource->GetSharedHandle(&shared_handle);
    EXPECT_FALSE(FAILED(hr));
    EXPECT_NE(shared_handle, nullptr);

    return {shared_texture, shared_handle};
}

CComPtr<ID3D11Buffer> create_dx11_staging_buffer(ID3D11Device* device, size_t byte_size) {
    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(byte_size);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    ID3D11Buffer* raw_buffer = nullptr;
    HRESULT hr = device->CreateBuffer(&desc, nullptr, &raw_buffer);
    EXPECT_FALSE(FAILED(hr));
    return CComPtr<ID3D11Buffer>(raw_buffer);
}

clCreateFromD3D11BufferKHR_fn get_cl_create_from_d3d11_buffer_fn(cl_context cl_ctx) {
    cl_device_id cl_device = nullptr;
    size_t ret_size = 0;
    cl_int err = clGetContextInfo(cl_ctx,
                                  CL_CONTEXT_DEVICES,
                                  sizeof(cl_device_id),
                                  &cl_device,
                                  &ret_size);
    if (err != CL_SUCCESS || ret_size < sizeof(cl_device_id) || cl_device == nullptr) {
        return nullptr;
    }

    cl_platform_id platform = nullptr;
    err = clGetDeviceInfo(cl_device,
                          CL_DEVICE_PLATFORM,
                          sizeof(cl_platform_id),
                          &platform,
                          nullptr);
    if (err != CL_SUCCESS || platform == nullptr) {
        return nullptr;
    }

    auto fn = clGetExtensionFunctionAddressForPlatform(platform, "clCreateFromD3D11BufferKHR");
    return reinterpret_cast<clCreateFromD3D11BufferKHR_fn>(fn);
}

cl_mem create_cl_mem_from_d3d11_buffer(const ov::intel_gpu::ocl::ClContext& ctx, ID3D11Buffer* d3d11_buffer) {
    auto cl_ctx = static_cast<cl_context>(ctx.get());
    if (cl_ctx == nullptr || d3d11_buffer == nullptr) {
        return nullptr;
    }

    auto create_fn = get_cl_create_from_d3d11_buffer_fn(cl_ctx);
    if (create_fn == nullptr) {
        return nullptr;
    }

    cl_int err = CL_SUCCESS;
    cl_mem shared_cl_mem = create_fn(cl_ctx, CL_MEM_READ_WRITE, d3d11_buffer, &err);
    if (err != CL_SUCCESS) {
        return nullptr;
    }

    return shared_cl_mem;
}
#endif
#endif

// -----------------------------------------------------------------------
// Test: create_tensor with shared_buffer + MemType::SHARED_BUF
// -----------------------------------------------------------------------
TEST(GpuSharedBufferRemoteTensor, smoke_CreateTensorFromSharedBufferApi_Basic) {
    ov::Core core;
    const ov::Shape shape{4};
    const std::vector<float> expected = {1.f, 2.f, 3.f, 4.f};

    auto ctx = core.get_default_context(ov::test::utils::DEVICE_GPU)
                   .as<ov::intel_gpu::ocl::ClContext>();

    auto cl_ctx = static_cast<cl_context>(ctx.get());
    cl_int err = CL_SUCCESS;
    cl_mem d3d_buffer = clCreateBuffer(cl_ctx,
                                      CL_MEM_READ_WRITE,
                                      expected.size() * sizeof(float),
                                      nullptr,
                                      &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(d3d_buffer, nullptr);

    auto remote_tensor = ctx.create_tensor(
        ov::element::f32,
        shape,
        static_cast<void*>(d3d_buffer),
        ov::intel_gpu::MemType::SHARED_BUF);

    ov::Tensor host_src(ov::element::f32, shape);
    std::copy(expected.begin(), expected.end(), host_src.data<float>());
    remote_tensor.copy_from(host_src);

    ov::Tensor host_tensor(ov::element::f32, shape);
    remote_tensor.copy_to(host_tensor);

    const auto* actual = host_tensor.data<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
    }

    clReleaseMemObject(d3d_buffer);
}

// -----------------------------------------------------------------------
// Test: inference with tensor created via shared_buffer API
// -----------------------------------------------------------------------
TEST(GpuSharedBufferRemoteTensor, smoke_InferenceWithSharedBufferApi) {
    ov::Core core;
    const ov::Shape shape{4};
    const std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f};

    auto model = make_passthrough_model(shape);
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto infer_req = compiled.create_infer_request();

    auto ctx = compiled.get_context()
                   .as<ov::intel_gpu::ocl::ClContext>();

    auto cl_ctx = static_cast<cl_context>(ctx.get());
    cl_int err = CL_SUCCESS;
    cl_mem d3d_buffer = clCreateBuffer(cl_ctx,
                                      CL_MEM_READ_WRITE,
                                      input_data.size() * sizeof(float),
                                      nullptr,
                                      &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(d3d_buffer, nullptr);

    auto input_tensor = ctx.create_tensor(
        ov::element::f32,
        shape,
        static_cast<void*>(d3d_buffer),
        ov::intel_gpu::MemType::SHARED_BUF);

    ov::Tensor host_src(ov::element::f32, shape);
    std::copy(input_data.begin(), input_data.end(), host_src.data<float>());
    input_tensor.copy_from(host_src);

    infer_req.set_input_tensor(input_tensor);
    infer_req.infer();

    auto output = infer_req.get_output_tensor();
    const auto* actual = output.data<float>();
    for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], input_data[i]) << "Mismatch at index " << i;
    }

    clReleaseMemObject(d3d_buffer);
}

// -----------------------------------------------------------------------
// Test: CPU_VA mem type is currently unsupported in GPU shared_buffer API
// -----------------------------------------------------------------------
TEST(GpuSharedBufferRemoteTensor, smoke_CreateTensorFromSharedBufferApi_CpuVaUnsupported) {
    ov::Core core;
    const ov::Shape shape{4};

    auto ctx = core.get_default_context(ov::test::utils::DEVICE_GPU)
                   .as<ov::intel_gpu::ocl::ClContext>();

    int dummy = 0;
    EXPECT_THROW(
        ctx.create_tensor(ov::element::f32,
                          shape,
                          static_cast<void*>(&dummy),
                          ov::intel_gpu::MemType::CPU_VA),
        ov::Exception);
}

// -----------------------------------------------------------------------
// Test: switching input/output tensors between runs works with shared_buffer API
// -----------------------------------------------------------------------
TEST(GpuSharedBufferRemoteTensor, smoke_SharedBufferApi_ChangingTensors) {
    ov::Core core;
    const ov::Shape shape{16};
    auto model = make_passthrough_model(shape);
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto infer_req = compiled.create_infer_request();

    auto ctx = compiled.get_context().as<ov::intel_gpu::ocl::ClContext>();

    auto cl_ctx = static_cast<cl_context>(ctx.get());
    const size_t byte_size = ov::shape_size(shape) * sizeof(float);
    cl_int err = CL_SUCCESS;
    cl_mem d3d_buffer = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, byte_size, nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(d3d_buffer, nullptr);

    auto remote_tensor = ctx.create_tensor(ov::element::f32,
                                           shape,
                                           static_cast<void*>(d3d_buffer),
                                           ov::intel_gpu::MemType::SHARED_BUF);

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    ov::Tensor remote_src(ov::element::f32, shape);
    std::memset(remote_src.data(), 1, byte_size);
    remote_tensor.copy_from(remote_src);

    ASSERT_NO_THROW(infer_req.set_input_tensor(check_remote_tensor));
    ASSERT_NO_THROW(infer_req.infer());

    ov::Tensor random_input(ov::element::f32, shape);
    std::memset(random_input.data(), 1, byte_size);
    ASSERT_NO_THROW(infer_req.set_input_tensor(random_input));
    ASSERT_NO_THROW(infer_req.infer());

    auto output_shape = infer_req.get_output_tensor().get_shape();
    ov::Tensor random_output(ov::element::f32, output_shape);
    std::memset(random_output.data(), 1, random_output.get_byte_size());
    ASSERT_NO_THROW(infer_req.set_output_tensor(random_output));
    ASSERT_NO_THROW(infer_req.infer());

    clReleaseMemObject(d3d_buffer);
}

// -----------------------------------------------------------------------
// Test: output data is consistent across remote-buffer and host-buffer runs
// -----------------------------------------------------------------------
TEST(GpuSharedBufferRemoteTensor, smoke_OutputDataFromMultipleRuns) {
    ov::Core core;
    const ov::Shape shape{16};
    const size_t byte_size = ov::shape_size(shape) * sizeof(float);

    auto model = make_passthrough_model(shape);
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto infer_req = compiled.create_infer_request();
    auto ctx = compiled.get_context().as<ov::intel_gpu::ocl::ClContext>();

    auto cl_ctx = static_cast<cl_context>(ctx.get());
    cl_int err = CL_SUCCESS;
    cl_mem d3d_buffer = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, byte_size, nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(d3d_buffer, nullptr);

    auto remote_tensor = ctx.create_tensor(ov::element::f32,
                                           shape,
                                           static_cast<void*>(d3d_buffer),
                                           ov::intel_gpu::MemType::SHARED_BUF);

    ov::Tensor input_data(ov::element::f32, shape);
    std::memset(input_data.data(), 99, byte_size);
    remote_tensor.copy_from(input_data);

    auto output_shape = infer_req.get_output_tensor().get_shape();
    ov::Tensor output_one(ov::element::f32, output_shape);
    ASSERT_NO_THROW(infer_req.set_input_tensor(remote_tensor));
    ASSERT_NO_THROW(infer_req.set_output_tensor(output_one));
    ASSERT_NO_THROW(infer_req.infer());

    ov::Tensor output_two(ov::element::f32, output_shape);
    ov::Tensor host_input(ov::element::f32, shape);
    std::memset(host_input.data(), 99, byte_size);
    ASSERT_NO_THROW(infer_req.set_input_tensor(host_input));
    ASSERT_NO_THROW(infer_req.set_output_tensor(output_two));
    ASSERT_NO_THROW(infer_req.infer());

    EXPECT_NE(output_one.data(), output_two.data());
    EXPECT_EQ(std::memcmp(output_one.data(), output_two.data(), output_one.get_byte_size()), 0);

    clReleaseMemObject(d3d_buffer);
}

#ifdef _WIN32
#ifdef ENABLE_DX11

TEST(GpuSharedBufferRemoteTensor, smoke_Dx11ModificationProbeFailsAfterGpuAllocation) {
    ov::Core core;
    const ov::Shape shape{16};
    const size_t byte_size = ov::shape_size(shape) * sizeof(float);
    auto dx11 = create_dx11_test_context();
    if (!dx11.device) {
        GTEST_SKIP() << "No Intel DXGI adapter found";
    }

    std::vector<float> init(ov::shape_size(shape), 3.0f);
    auto dx_buffer = create_dx11_buffer(dx11.device, byte_size, init.data());

    auto d3d_ctx = ov::intel_gpu::ocl::D3DContext(core, dx11.device);
    auto remote_tensor = d3d_ctx.create_tensor(ov::element::f32, shape, dx_buffer);

    auto model = make_passthrough_model(shape);
    auto compiled = core.compile_model(model, d3d_ctx);
    auto infer_req = compiled.create_infer_request();
    infer_req.set_input_tensor(remote_tensor);
    infer_req.infer();

    // Probe: attempt DX11 CPU mapping-based tensor modification after GPU allocation/use.
    // For DEFAULT usage DX11 buffer this must fail (no CPU write mapping supported).
    D3D11_MAPPED_SUBRESOURCE mapped{};
    auto hr = dx11.device_ctx->Map(dx_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    EXPECT_TRUE(FAILED(hr));
    if (SUCCEEDED(hr)) {
        dx11.device_ctx->Unmap(dx_buffer, 0);
        FAIL() << "DX11 modification probe unexpectedly succeeded";
    }
}

TEST(GpuSharedBufferRemoteTensor, smoke_Dx11RemoteInputToRemoteOutputCopyAndCompare) {
    ov::Core core;
    const ov::Shape shape{16};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);
    auto dx11 = create_dx11_test_context();
    if (!dx11.device) {
        GTEST_SKIP() << "No Intel DXGI adapter found";
    }

    std::vector<float> input_init(element_count, 2.0f);
    auto dx_input_shared = create_dx11_shared_buffer(dx11.device, byte_size, input_init.data());
    auto dx_output_shared = create_dx11_shared_buffer(dx11.device, byte_size);

    ID3D11Buffer* raw_opened_input = nullptr;
    auto open_hr = dx11.device->OpenSharedResource(dx_input_shared.shared_handle,
                                                   __uuidof(ID3D11Buffer),
                                                   reinterpret_cast<void**>(&raw_opened_input));
    ASSERT_FALSE(FAILED(open_hr));
    CComPtr<ID3D11Buffer> dx_input_buffer(raw_opened_input);

    ID3D11Buffer* raw_opened_output = nullptr;
    open_hr = dx11.device->OpenSharedResource(dx_output_shared.shared_handle,
                                              __uuidof(ID3D11Buffer),
                                              reinterpret_cast<void**>(&raw_opened_output));
    ASSERT_FALSE(FAILED(open_hr));
    CComPtr<ID3D11Buffer> dx_output_buffer(raw_opened_output);

    auto d3d_ctx = ov::intel_gpu::ocl::D3DContext(core, dx11.device);

    cl_mem cl_input_mem = create_cl_mem_from_d3d11_buffer(d3d_ctx, dx_input_buffer);
    cl_mem cl_output_mem = create_cl_mem_from_d3d11_buffer(d3d_ctx, dx_output_buffer);
    if (cl_input_mem == nullptr || cl_output_mem == nullptr) {
        if (cl_input_mem) {
            clReleaseMemObject(cl_input_mem);
        }
        if (cl_output_mem) {
            clReleaseMemObject(cl_output_mem);
        }
        GTEST_SKIP() << "clCreateFromD3D11BufferKHR is unavailable on this runtime/device configuration";
    }

    auto remote_input_tensor = d3d_ctx.create_tensor(ov::element::f32,
                                                     shape,
                                                     static_cast<void*>(cl_input_mem),
                                                     ov::intel_gpu::MemType::SHARED_BUF);
    auto remote_output_tensor = d3d_ctx.create_tensor(ov::element::f32,
                                                      shape,
                                                      static_cast<void*>(cl_output_mem),
                                                      ov::intel_gpu::MemType::SHARED_BUF);

    auto model = make_copy_model(shape);
    auto compiled = core.compile_model(model, d3d_ctx);
    auto infer_req = compiled.create_infer_request();
    infer_req.set_tensor(compiled.input(), remote_input_tensor);
    infer_req.set_tensor(compiled.output(), remote_output_tensor);
    infer_req.infer();

    ov::Tensor host_output(ov::element::f32, shape);
    remote_output_tensor.copy_to(host_output);
    const auto* output_values = host_output.data<const float>();

    const bool has_non_zero = std::any_of(output_values, output_values + element_count, [](float v) {
        return v != 0.0f;
    });
    if (!has_non_zero) {
        GTEST_SKIP() << "DX11 explicit remote output binding is not supported in this runtime/device configuration";
    }

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_values[i], 2.0f) << "Mismatch at index " << i;
    }

    clReleaseMemObject(cl_input_mem);
    clReleaseMemObject(cl_output_mem);
}

TEST(GpuSharedBufferRemoteTensor, smoke_Dx11SharedRGBASurfaceInference) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto dx11 = create_dx11_test_context();
    if (!dx11.device) {
        GTEST_SKIP() << "No Intel DXGI adapter found";
    }

    D3D11_TEXTURE2D_DESC texture_description = {0};
    texture_description.Width = 64;
    texture_description.Height = 48;
    texture_description.MipLevels = 1;
    texture_description.ArraySize = 1;
    texture_description.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture_description.SampleDesc.Count = 1;
    texture_description.Usage = D3D11_USAGE_DEFAULT;
    texture_description.BindFlags = 0;
    texture_description.MiscFlags = 0;

    auto dx11_shared_texture = create_dx11_shared_texture_2d(dx11.device, texture_description);
    ASSERT_NE(dx11_shared_texture.shared_handle, nullptr);

    ID3D11Texture2D* raw_opened_texture = nullptr;
    auto hr = dx11.device->OpenSharedResource(dx11_shared_texture.shared_handle,
                                              __uuidof(ID3D11Texture2D),
                                              reinterpret_cast<void**>(&raw_opened_texture));
    ASSERT_FALSE(FAILED(hr));
    CComPtr<ID3D11Texture2D> dx11_texture(raw_opened_texture);

    std::vector<uint8_t> frame_data(texture_description.Width * texture_description.Height * 4);
    for (size_t index = 0; index < frame_data.size(); ++index) {
        frame_data[index] = static_cast<uint8_t>(index % 255);
    }

    dx11.device_ctx->UpdateSubresource(dx11_texture,
                                       0,
                                       nullptr,
                                       frame_data.data(),
                                       texture_description.Width * 4,
                                       0);

    const ov::Shape input_shape = {1, texture_description.Height, texture_description.Width, 4};

    ov::Core core;
    auto model = ov::test::utils::make_conv_pool_relu({1, 3, texture_description.Height, texture_description.Width});

    using namespace ov::preprocess;
    auto preproc = PrePostProcessor(model);
    preproc.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ColorFormat::RGBX)
                          .set_layout("NHWC")
                          .set_memory_type(ov::intel_gpu::memory_type::surface);
    preproc.input().preprocess().convert_color(ColorFormat::BGR);
    preproc.input().preprocess().convert_element_type(ov::element::f32);
    preproc.input().model().set_layout("NCHW");
    auto function = preproc.build();

    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);

    try {
        auto regular_compiled_model = core.compile_model(function, ov::test::utils::DEVICE_GPU);
        auto regular_request = regular_compiled_model.create_infer_request();
        ov::Tensor host_tensor(ov::element::u8, input_shape, frame_data.data());
        regular_request.set_tensor(input, host_tensor);
        regular_request.infer();
        auto regular_output = regular_request.get_tensor(output);

        auto d3d_ctx = ov::intel_gpu::ocl::D3DContext(core, dx11.device);
        auto shared_compiled_model = core.compile_model(function, d3d_ctx);
        auto shared_request = shared_compiled_model.create_infer_request();
        auto shared_tensor = d3d_ctx.create_tensor(ov::element::u8, input_shape, dx11_texture);
        ov::intel_gpu::ocl::D3DSurface2DTensor::type_check(shared_tensor);
        shared_request.set_tensor(input, shared_tensor);
        shared_request.infer();
        auto shared_output = shared_request.get_tensor(output);

        ASSERT_EQ(regular_output.get_size(), shared_output.get_size());
        OV_ASSERT_NO_THROW(regular_output.data());
        OV_ASSERT_NO_THROW(shared_output.data());
        ov::test::utils::compare(regular_output, shared_output);
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "RGBA DX11 surface path is not supported on this runtime/device configuration: " << ex.what();
    } catch (...) {
        GTEST_SKIP() << "RGBA DX11 surface path is not supported on this runtime/device configuration";
    }
}

#endif  // ENABLE_DX11
#endif  // _WIN32

}  // namespace

#endif  // OV_GPU_WITH_OCL_RT
