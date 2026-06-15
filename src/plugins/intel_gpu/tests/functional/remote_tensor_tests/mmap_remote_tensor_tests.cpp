// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(OV_GPU_WITH_OCL_RT) && defined(__linux__)

#include <gtest/gtest.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

namespace {

std::shared_ptr<ov::Model> make_copy_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0.0f});
    auto add = std::make_shared<ov::op::v1::Add>(param, zero);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

struct MmapGuard {
    void* ptr = MAP_FAILED;
    size_t bytes = 0;

    ~MmapGuard() {
        if (ptr != MAP_FAILED) {
            munmap(ptr, bytes);
        }
    }
};

struct FdGuard {
    int fd = -1;

    FdGuard() = default;
    explicit FdGuard(int f) : fd(f) {}
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
    FdGuard(FdGuard&& other) noexcept : fd(other.fd) {
        other.fd = -1;
    }
    FdGuard& operator=(FdGuard&& other) noexcept {
        if (this != &other) {
            if (fd >= 0) {
                close(fd);
            }
            fd = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    ~FdGuard() {
        if (fd >= 0) {
            close(fd);
        }
    }
};

struct TempFileGuard {
    std::string path;

    TempFileGuard() = default;
    explicit TempFileGuard(std::string p) : path(std::move(p)) {}
    TempFileGuard(const TempFileGuard&) = delete;
    TempFileGuard& operator=(const TempFileGuard&) = delete;
    TempFileGuard(TempFileGuard&& other) noexcept : path(std::move(other.path)) {
        other.path.clear();
    }
    TempFileGuard& operator=(TempFileGuard&& other) noexcept {
        if (this != &other) {
            if (!path.empty()) {
                unlink(path.c_str());
            }
            path = std::move(other.path);
            other.path.clear();
        }
        return *this;
    }

    ~TempFileGuard() {
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }
};

std::pair<FdGuard, TempFileGuard> create_temp_file(size_t byte_size) {
    std::string tmpl = "/tmp/ov_gpu_mmap_remote_tensor_XXXXXX";
    std::vector<char> path_buf(tmpl.begin(), tmpl.end());
    path_buf.push_back('\0');

    int fd = mkstemp(path_buf.data());
    if (fd < 0) {
        OPENVINO_THROW("mkstemp failed: ", std::strerror(errno));
    }

    if (ftruncate(fd, static_cast<off_t>(byte_size)) != 0) {
        close(fd);
        OPENVINO_THROW("ftruncate failed: ", std::strerror(errno));
    }

    return {FdGuard(fd), TempFileGuard(std::string(path_buf.data()))};
}

TEST(GpuMmapedMemoryRemoteTensor, smoke_MmapRemoteInputToRemoteOutputCopyAndCompare) {
    ov::Core core;
    const ov::Shape shape{16};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);

    std::string target_device = ov::test::utils::DEVICE_GPU;
    auto ctx = core.get_default_context(target_device).as<ov::intel_gpu::ocl::ClContext>();

    void* input_ptr = mmap(nullptr, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(input_ptr, MAP_FAILED) << "Failed to mmap input buffer";
    MmapGuard input_guard{input_ptr, byte_size};

    void* output_ptr = mmap(nullptr, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(output_ptr, MAP_FAILED) << "Failed to mmap output buffer";
    MmapGuard output_guard{output_ptr, byte_size};

    std::fill_n(static_cast<float*>(input_ptr), element_count, 2.0f);
    std::fill_n(static_cast<float*>(output_ptr), element_count, 0.0f);

    auto remote_input_tensor = ctx.create_tensor(ov::element::f32,
                                                 shape,
                                                 input_ptr,
                                                 ov::intel_gpu::MemType::cpu_pointer);
    auto remote_output_tensor = ctx.create_tensor(ov::element::f32,
                                                  shape,
                                                  output_ptr,
                                                  ov::intel_gpu::MemType::cpu_pointer);

    auto model = make_copy_model(shape);
    auto compiled = core.compile_model(model, ctx);
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

TEST(GpuMmapedMemoryRemoteTensor, smoke_MallocBasedRemoteInputToRemoteOutputCopyAndCompare) {
    ov::Core core;
    const ov::Shape shape{1024};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);

    std::string target_device = ov::test::utils::DEVICE_GPU;
    auto ctx = core.get_default_context(target_device).as<ov::intel_gpu::ocl::ClContext>();

    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    ASSERT_GT(page_size, 0);

    std::unique_ptr<void, decltype(&std::free)> raw_input_ptr(std::malloc(byte_size + page_size), &std::free);
    ASSERT_NE(raw_input_ptr.get(), nullptr) << "Failed to allocate input buffer with malloc";
    std::unique_ptr<void, decltype(&std::free)> raw_output_ptr(std::malloc(byte_size + page_size), &std::free);
    ASSERT_NE(raw_output_ptr.get(), nullptr) << "Failed to allocate output buffer with malloc";

    auto align_to_page = [page_size](void* ptr) {
        auto addr = reinterpret_cast<uintptr_t>(ptr);
        auto aligned_addr = (addr + page_size - 1) & ~(page_size - 1);
        return reinterpret_cast<void*>(aligned_addr);
    };

    void* input_ptr = align_to_page(raw_input_ptr.get());
    void* output_ptr = align_to_page(raw_output_ptr.get());

    std::fill_n(static_cast<float*>(input_ptr), element_count, 2.0f);
    std::fill_n(static_cast<float*>(output_ptr), element_count, 0.0f);

    auto remote_input_tensor = ctx.create_tensor(ov::element::f32,
                                                 shape,
                                                 input_ptr,
                                                 ov::intel_gpu::MemType::cpu_pointer);
    auto remote_output_tensor = ctx.create_tensor(ov::element::f32,
                                                  shape,
                                                  output_ptr,
                                                  ov::intel_gpu::MemType::cpu_pointer);

    auto model = make_copy_model(shape);
    auto compiled = core.compile_model(model, ctx);
    auto infer_req = compiled.create_infer_request();
    infer_req.set_tensor(compiled.input(), remote_input_tensor);
    infer_req.set_tensor(compiled.output(), remote_output_tensor);

    infer_req.infer();

    ov::Tensor host_output(ov::element::f32, shape);
    remote_output_tensor.copy_to(host_output);
    const auto* output_values = host_output.data<const float>();
    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_values[i], 2.0f) << "Mismatch at index " << i;
    }
}

TEST(GpuMmapedMemoryRemoteTensor, smoke_MmapRemoteInputToRemoteOutputCopyAndCompare_FileBacked) {
    ov::Core core;
    const ov::Shape shape{4096};
    const size_t element_count = ov::shape_size(shape);
    const size_t byte_size = element_count * sizeof(float);

    std::string target_device = ov::test::utils::DEVICE_GPU;
    auto ctx = core.get_default_context(target_device).as<ov::intel_gpu::ocl::ClContext>();

    auto [input_fd, input_file] = create_temp_file(byte_size);
    auto [output_fd, output_file] = create_temp_file(byte_size);

    void* input_ptr = mmap(nullptr, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, input_fd.fd, 0);
    ASSERT_NE(input_ptr, MAP_FAILED) << "Failed to mmap input file";
    MmapGuard input_guard{input_ptr, byte_size};

    void* output_ptr = mmap(nullptr, byte_size, PROT_READ | PROT_WRITE, MAP_SHARED, output_fd.fd, 0);
    ASSERT_NE(output_ptr, MAP_FAILED) << "Failed to mmap output file";
    MmapGuard output_guard{output_ptr, byte_size};

    std::fill_n(static_cast<float*>(input_ptr), element_count, 2.0f);
    std::fill_n(static_cast<float*>(output_ptr), element_count, 0.0f);
    ASSERT_EQ(msync(input_ptr, byte_size, MS_SYNC), 0) << "msync(input) failed";
    ASSERT_EQ(msync(output_ptr, byte_size, MS_SYNC), 0) << "msync(output-init) failed";

    auto remote_input_tensor = ctx.create_tensor(ov::element::f32,
                                                 shape,
                                                 input_ptr,
                                                 ov::intel_gpu::MemType::cpu_pointer);
    auto remote_output_tensor = ctx.create_tensor(ov::element::f32,
                                                  shape,
                                                  output_ptr,
                                                  ov::intel_gpu::MemType::cpu_pointer);

    auto model = make_copy_model(shape);
    auto compiled = core.compile_model(model, ctx);
    auto infer_req = compiled.create_infer_request();
    infer_req.set_tensor(compiled.input(), remote_input_tensor);
    infer_req.set_tensor(compiled.output(), remote_output_tensor);

    infer_req.infer();

    ASSERT_EQ(msync(output_ptr, byte_size, MS_SYNC), 0) << "msync failed";

    std::vector<float> output_from_file(element_count, 0.0f);
    ssize_t output_read = pread(output_fd.fd, output_from_file.data(), byte_size, 0);
    ASSERT_NE(output_read, static_cast<ssize_t>(-1)) << "Failed to read output file: " << std::strerror(errno);
    ASSERT_EQ(static_cast<size_t>(output_read), byte_size) << "Failed to read output file";

    for (size_t i = 0; i < element_count; ++i) {
        EXPECT_FLOAT_EQ(output_from_file[i], 2.0f) << "Mismatch at index " << i;
    }
}

}  // namespace

#endif
