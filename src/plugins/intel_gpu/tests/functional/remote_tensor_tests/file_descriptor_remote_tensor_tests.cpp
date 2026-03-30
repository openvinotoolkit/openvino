// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_OCL_RT

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
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace {

// Simple passthrough model: Parameter -> Result
std::shared_ptr<ov::Model> make_passthrough_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto result = std::make_shared<ov::op::v0::Result>(param);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

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
    cl_mem cl_buffer = clCreateBuffer(cl_ctx,
                                      CL_MEM_READ_WRITE,
                                      expected.size() * sizeof(float),
                                      nullptr,
                                      &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(cl_buffer, nullptr);

    auto remote_tensor = ctx.create_tensor(
        ov::element::f32,
        shape,
        static_cast<void*>(cl_buffer),
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

    clReleaseMemObject(cl_buffer);
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
    cl_mem cl_buffer = clCreateBuffer(cl_ctx,
                                      CL_MEM_READ_WRITE,
                                      input_data.size() * sizeof(float),
                                      nullptr,
                                      &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(cl_buffer, nullptr);

    auto input_tensor = ctx.create_tensor(
        ov::element::f32,
        shape,
        static_cast<void*>(cl_buffer),
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

    clReleaseMemObject(cl_buffer);
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
    cl_mem cl_buffer = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, byte_size, nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(cl_buffer, nullptr);

    auto remote_tensor = ctx.create_tensor(ov::element::f32,
                                           shape,
                                           static_cast<void*>(cl_buffer),
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

    clReleaseMemObject(cl_buffer);
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
    cl_mem cl_buffer = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, byte_size, nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);
    ASSERT_NE(cl_buffer, nullptr);

    auto remote_tensor = ctx.create_tensor(ov::element::f32,
                                           shape,
                                           static_cast<void*>(cl_buffer),
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

    clReleaseMemObject(cl_buffer);
}

#ifdef _WIN32
#ifdef ENABLE_DX11

TEST(GpuSharedBufferRemoteTensor, smoke_Dx11ModificationProbeFailsAfterGpuAllocation) {
    ov::Core core;
    const ov::Shape shape{16};
    const size_t byte_size = ov::shape_size(shape) * sizeof(float);

    IDXGIFactory* raw_factory = nullptr;
    HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&raw_factory));
    ASSERT_FALSE(FAILED(hr));
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
        GTEST_SKIP() << "No Intel DXGI adapter found";
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
    ASSERT_FALSE(FAILED(hr));

    CComPtr<ID3D11Device> device(raw_device);
    CComPtr<ID3D11DeviceContext> device_ctx(raw_ctx);

    std::vector<float> init(ov::shape_size(shape), 3.0f);
    D3D11_BUFFER_DESC buf_desc{};
    buf_desc.ByteWidth = static_cast<UINT>(byte_size);
    buf_desc.Usage = D3D11_USAGE_DEFAULT;
    buf_desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    buf_desc.CPUAccessFlags = 0;
    buf_desc.MiscFlags = 0;
    D3D11_SUBRESOURCE_DATA init_data{};
    init_data.pSysMem = init.data();

    ID3D11Buffer* raw_buffer = nullptr;
    hr = device->CreateBuffer(&buf_desc, &init_data, &raw_buffer);
    ASSERT_FALSE(FAILED(hr));
    CComPtr<ID3D11Buffer> dx_buffer(raw_buffer);

    auto d3d_ctx = ov::intel_gpu::ocl::D3DContext(core, device);
    auto remote_tensor = d3d_ctx.create_tensor(ov::element::f32, shape, dx_buffer);

    auto model = make_passthrough_model(shape);
    auto compiled = core.compile_model(model, d3d_ctx);
    auto infer_req = compiled.create_infer_request();
    infer_req.set_input_tensor(remote_tensor);
    infer_req.infer();

    // Probe: attempt DX11 CPU mapping-based tensor modification after GPU allocation/use.
    // For DEFAULT usage DX11 buffer this must fail (no CPU write mapping supported).
    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = device_ctx->Map(dx_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    EXPECT_TRUE(FAILED(hr));
    if (SUCCEEDED(hr)) {
        device_ctx->Unmap(dx_buffer, 0);
        FAIL() << "DX11 modification probe unexpectedly succeeded";
    }
}

#endif  // ENABLE_DX11
#endif  // _WIN32

}  // namespace

#endif  // OV_GPU_WITH_OCL_RT
