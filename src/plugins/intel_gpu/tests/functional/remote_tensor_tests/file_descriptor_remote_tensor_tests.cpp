// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_OCL_RT

#include <fstream>
#include <filesystem>
#include <numeric>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace {

// Helper: write binary data to a temp file, return path
std::filesystem::path write_temp_binary_file(const std::vector<float>& data) {
    auto path = std::filesystem::temp_directory_path() / "ov_gpu_fd_test.bin";
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return path;
}

// Simple passthrough model: Parameter -> Result
std::shared_ptr<ov::Model> make_passthrough_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto result = std::make_shared<ov::op::v0::Result>(param);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

// -----------------------------------------------------------------------
// Test: create_tensor with file_descriptor, data is loaded and readable
// -----------------------------------------------------------------------
TEST(FileDescriptorRemoteTensor, smoke_CreateTensorFromFile_USMHost) {
    ov::Core core;
    const ov::Shape shape{4};
    const std::vector<float> expected = {1.f, 2.f, 3.f, 4.f};
    auto path = write_temp_binary_file(expected);

    auto ctx = core.get_default_context(ov::test::utils::DEVICE_GPU)
                   .as<ov::intel_gpu::ocl::ClContext>();

    // Create tensor backed by USM host memory, loaded from file
    auto remote_tensor = ctx.create_tensor(
        ov::element::f32,
        shape,
        {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER),
         ov::intel_gpu::file_descriptor(ov::intel_gpu::FileDescriptor{path})});

    // Copy back to host and verify
    ov::Tensor host_tensor(ov::element::f32, shape);
    remote_tensor.copy_to(host_tensor);

    const auto* actual = host_tensor.data<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
    }

    std::filesystem::remove(path);
}

// -----------------------------------------------------------------------
// Test: file_descriptor with non-zero offset
// -----------------------------------------------------------------------
TEST(FileDescriptorRemoteTensor, smoke_CreateTensorFromFile_WithOffset) {
    ov::Core core;
    const ov::Shape shape{2};
    // File has 4 floats; we read from offset 2*sizeof(float) → {3.f, 4.f}
    const std::vector<float> file_data = {1.f, 2.f, 3.f, 4.f};
    const std::vector<float> expected = {3.f, 4.f};
    auto path = write_temp_binary_file(file_data);

    auto ctx = core.get_default_context(ov::test::utils::DEVICE_GPU)
                   .as<ov::intel_gpu::ocl::ClContext>();

    auto remote_tensor = ctx.create_tensor(
        ov::element::f32,
        shape,
        {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER),
         ov::intel_gpu::file_descriptor(
             ov::intel_gpu::FileDescriptor{path, 2 * sizeof(float)})});

    ov::Tensor host_tensor(ov::element::f32, shape);
    remote_tensor.copy_to(host_tensor);

    const auto* actual = host_tensor.data<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
    }

    std::filesystem::remove(path);
}

// -----------------------------------------------------------------------
// Test: file_descriptor passed at context level, not tensor level
// -----------------------------------------------------------------------
TEST(FileDescriptorRemoteTensor, smoke_CreateTensorFromFile_ContextLevelDescriptor) {
    ov::Core core;
    const ov::Shape shape{4};
    const std::vector<float> expected = {5.f, 6.f, 7.f, 8.f};
    auto path = write_temp_binary_file(expected);

    // Pass file_descriptor in context properties
    auto ctx = core.create_context(
        ov::test::utils::DEVICE_GPU,
        {ov::intel_gpu::context_type(ov::intel_gpu::ContextType::OCL),
         ov::intel_gpu::ocl_context(
             core.get_default_context(ov::test::utils::DEVICE_GPU)
                 .get_params()
                 .at(ov::intel_gpu::ocl_context.name())
                 .as<ov::intel_gpu::gpu_handle_param>()),
         ov::intel_gpu::file_descriptor(ov::intel_gpu::FileDescriptor{path})});

    auto remote_tensor = ctx.create_tensor(
        ov::element::f32,
        shape,
        {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER)});

    ov::Tensor host_tensor(ov::element::f32, shape);
    remote_tensor.copy_to(host_tensor);

    const auto* actual = host_tensor.data<float>();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
    }

    std::filesystem::remove(path);
}

// -----------------------------------------------------------------------
// Test: inference with tensor loaded from file
// -----------------------------------------------------------------------
TEST(FileDescriptorRemoteTensor, smoke_InferenceWithFileTensor) {
    ov::Core core;
    const ov::Shape shape{4};
    const std::vector<float> input_data = {1.f, 2.f, 3.f, 4.f};
    auto path = write_temp_binary_file(input_data);

    auto model = make_passthrough_model(shape);
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto infer_req = compiled.create_infer_request();

    auto ctx = compiled.get_context().as<ov::intel_gpu::ocl::ClContext>();

    auto input_tensor = ctx.create_tensor(
        ov::element::f32,
        shape,
        {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER),
         ov::intel_gpu::file_descriptor(ov::intel_gpu::FileDescriptor{path})});

    infer_req.set_input_tensor(input_tensor);
    infer_req.infer();

    auto output = infer_req.get_output_tensor();
    const auto* actual = output.data<float>();
    for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_FLOAT_EQ(actual[i], input_data[i]) << "Mismatch at index " << i;
    }

    std::filesystem::remove(path);
}

// -----------------------------------------------------------------------
// Test: offset beyond file end throws
// -----------------------------------------------------------------------
TEST(FileDescriptorRemoteTensor, smoke_OffsetBeyondFileEnd_Throws) {
    ov::Core core;
    const ov::Shape shape{4};
    const std::vector<float> file_data = {1.f, 2.f};
    auto path = write_temp_binary_file(file_data);

    auto ctx = core.get_default_context(ov::test::utils::DEVICE_GPU)
                   .as<ov::intel_gpu::ocl::ClContext>();

    EXPECT_THROW(
        ctx.create_tensor(
            ov::element::f32,
            shape,
            {ov::intel_gpu::shared_mem_type(ov::intel_gpu::SharedMemType::USM_HOST_BUFFER),
             ov::intel_gpu::file_descriptor(
                 ov::intel_gpu::FileDescriptor{path, 999999})}),
        ov::Exception);

    std::filesystem::remove(path);
}

// -----------------------------------------------------------------------
// Test: empty path throws
// -----------------------------------------------------------------------
TEST(FileDescriptorRemoteTensor, smoke_EmptyPath_Throws) {
    EXPECT_THROW(ov::intel_gpu::FileDescriptor{""},
                 ov::Exception);
}

}  // namespace

#endif  // OV_GPU_WITH_OCL_RT
