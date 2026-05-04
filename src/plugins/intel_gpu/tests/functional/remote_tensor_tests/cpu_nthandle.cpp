// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Empirical probe: attempt to use a Windows NT-style HANDLE created from CPU
// memory as a SHARED_BUF source for ov::intel_gpu::ocl::ClContext::create_tensor
// and run inference. Per OpenCL cl_khr_external_memory and
// VK_EXT_external_memory_host (Issue 7) and DX12 shared-heaps spec, this is
// expected to be unsupported. The test exercises three CPU-side allocation
// schemes and records each outcome; it does not assert a specific failure
// (driver behavior may differ), but it does assert that no inference path
// silently succeeds with semantically invalid input.

#if defined(OV_GPU_WITH_OCL_RT) && defined(_WIN32)

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#define NOMINMAX_DEFINED_CPU_NTHANDLE_TEST
#endif
#include <windows.h>
#ifdef NOMINMAX_DEFINED_CPU_NTHANDLE_TEST
#undef NOMINMAX
#undef NOMINMAX_DEFINED_CPU_NTHANDLE_TEST
#endif

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

std::shared_ptr<ov::Model> make_identity_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});
    auto add = std::make_shared<ov::op::v1::Add>(param, zero);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

// Attempts to create a SHARED_BUF remote tensor and run a no-op inference.
// Returns true if both creation and inference succeed and output equals input.
bool try_inference_with_handle(ov::Core& core,
                                ov::intel_gpu::ocl::ClContext& ov_ctx,
                                HANDLE handle,
                                const ov::Shape& shape,
                                const std::vector<float>& expected_input,
                                const std::string& label) {
    if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
        std::cout << "[INFO] " << label << ": no handle to test\n";
        return false;
    }

    ov::RemoteTensor remote_tensor;
    try {
        remote_tensor = ov_ctx.create_tensor(ov::element::f32, shape, handle,
                                              ov::intel_gpu::MemType::SHARED_BUF);
    } catch (const std::exception& ex) {
        std::cout << "[INFO] " << label << ": create_tensor rejected handle: " << ex.what() << "\n";
        return false;
    }
    std::cout << "[INFO] " << label << ": create_tensor accepted handle (unexpected for CPU memory)\n";

    try {
        auto model = make_identity_model(shape);
        auto compiled = core.compile_model(model, ov_ctx);
        auto infer_req = compiled.create_infer_request();
        infer_req.set_tensor(compiled.input(), remote_tensor);
        infer_req.set_tensor(compiled.output(), remote_tensor);
        infer_req.infer();
    } catch (const std::exception& ex) {
        std::cout << "[INFO] " << label << ": inference failed: " << ex.what() << "\n";
        return false;
    }

    ov::Tensor host_output(ov::element::f32, shape);
    try {
        remote_tensor.copy_to(host_output);
    } catch (const std::exception& ex) {
        std::cout << "[INFO] " << label << ": copy_to failed: " << ex.what() << "\n";
        return false;
    }

    const auto* output_values = host_output.data<const float>();
    const size_t element_count = expected_input.size();
    for (size_t i = 0; i < element_count; ++i) {
        if (output_values[i] != expected_input[i]) {
            std::cout << "[INFO] " << label << ": output mismatch at index " << i
                      << " (got " << output_values[i] << ", expected " << expected_input[i] << ")\n";
            return false;
        }
    }
    std::cout << "[INFO] " << label << ": inference succeeded with matching output\n";
    return true;
}

}  // namespace

// Allocates CPU memory and tries to construct a Windows HANDLE that represents
// it via three different mechanisms, then attempts inference for each.
// All three paths are expected to fail because cl_khr_external_memory accepts
// only NT handles produced by D3D11/D3D12/Vulkan exports referring to a DXGK
// allocation; CPU-only allocations are not registered with DXGK and cannot be
// imported as cl_mem regardless of how the HANDLE was created.
TEST(GpuSharedBufferRemoteTensor, smoke_CpuMemoryAsNtHandleForInference) {
    ov::Core core;
    const ov::Shape shape{1024};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);

    const std::string selected_gpu_device = "GPU.0";
    std::unique_ptr<ov::intel_gpu::ocl::ClContext> ov_ctx_ptr;
    try {
        ov_ctx_ptr = std::make_unique<ov::intel_gpu::ocl::ClContext>(
            core.get_default_context(selected_gpu_device).as<ov::intel_gpu::ocl::ClContext>());
    } catch (const std::exception& ex) {
        GTEST_SKIP() << "Failed to obtain ClContext for " << selected_gpu_device << ": " << ex.what();
    }
    auto& ov_ctx = *ov_ctx_ptr;

    std::vector<float> input_data(element_count, 7.0f);

    bool any_succeeded = false;

    // -----------------------------------------------------------------------
    // Path 1: NT handle to a pagefile-backed section object created via
    // CreateFileMapping. The mapped view is normal CPU virtual memory; the
    // returned handle is a real NT handle to a section object, *not* to a
    // DXGK allocation.
    // -----------------------------------------------------------------------
    {
        const SIZE_T total_bytes = static_cast<SIZE_T>(byte_size);
        HANDLE section_handle = CreateFileMappingW(INVALID_HANDLE_VALUE,
                                                    nullptr,
                                                    PAGE_READWRITE,
                                                    0,
                                                    static_cast<DWORD>(total_bytes),
                                                    nullptr);
        if (section_handle == nullptr) {
            std::cout << "[INFO] Path1 (CreateFileMapping): failed, GetLastError=" << GetLastError() << "\n";
        } else {
            void* view = MapViewOfFile(section_handle, FILE_MAP_ALL_ACCESS, 0, 0, total_bytes);
            if (view == nullptr) {
                std::cout << "[INFO] Path1 (CreateFileMapping): MapViewOfFile failed, GetLastError="
                          << GetLastError() << "\n";
            } else {
                memcpy(view, input_data.data(), byte_size);
                FlushViewOfFile(view, byte_size);

                if (try_inference_with_handle(core, ov_ctx, section_handle, shape, input_data,
                                               "Path1 (CreateFileMapping section)")) {
                    any_succeeded = true;
                }
                UnmapViewOfFile(view);
            }
            CloseHandle(section_handle);
        }
    }

    // -----------------------------------------------------------------------
    // Path 2: raw `new[]` CPU buffer. There is no native API to obtain an NT
    // handle for a heap allocation, so we duplicate the current process pseudo
    // handle as a stand-in. The handle does not refer to the buffer in any
    // meaningful way; this exercises the literal interpretation of "create a
    // Windows handle from a `new` allocation".
    // -----------------------------------------------------------------------
    {
        std::unique_ptr<float[]> raw_buffer(new float[element_count]);
        std::copy(input_data.begin(), input_data.end(), raw_buffer.get());

        HANDLE proc_pseudo = GetCurrentProcess();
        HANDLE duplicated = nullptr;
        if (!DuplicateHandle(proc_pseudo, proc_pseudo, proc_pseudo, &duplicated,
                              0, FALSE, DUPLICATE_SAME_ACCESS)) {
            std::cout << "[INFO] Path2 (new[] + DuplicateHandle): DuplicateHandle failed, GetLastError="
                      << GetLastError() << "\n";
        } else {
            if (try_inference_with_handle(core, ov_ctx, duplicated, shape, input_data,
                                           "Path2 (new[] + DuplicateHandle)")) {
                any_succeeded = true;
            }
            CloseHandle(duplicated);
        }
    }

    // -----------------------------------------------------------------------
    // Path 3: literal pointer-as-HANDLE. Reinterprets a raw `new[]` pointer as
    // a HANDLE value. This is the most direct interpretation of "use the CPU
    // allocation as a Windows handle".
    // -----------------------------------------------------------------------
    {
        std::unique_ptr<float[]> raw_buffer(new float[element_count]);
        std::copy(input_data.begin(), input_data.end(), raw_buffer.get());

        HANDLE pointer_as_handle = reinterpret_cast<HANDLE>(raw_buffer.get());
        // No CloseHandle: this is not a real kernel handle; closing it would
        // either be a no-op (HANDLE not in process handle table) or attempt to
        // free unrelated kernel state.
        if (try_inference_with_handle(core, ov_ctx, pointer_as_handle, shape, input_data,
                                       "Path3 (raw pointer reinterpret_cast<HANDLE>)")) {
            any_succeeded = true;
        }
    }

    EXPECT_FALSE(any_succeeded)
        << "Unexpected success: a CPU-only allocation was accepted as SHARED_BUF and produced "
           "matching inference output. This contradicts the OpenCL/Vulkan/DX12 external memory "
           "contract and should be investigated.";
}

#endif  // OV_GPU_WITH_OCL_RT && _WIN32
