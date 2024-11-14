// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/ov_behavior_test_utils.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "overload/overload_test_utils_npu.hpp"

#ifdef __linux__
#    include <linux/version.h>
#    if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
#        include <fcntl.h>
#        include <linux/dma-heap.h>
#        include <sys/ioctl.h>
#        include <sys/mman.h>

#        include <filesystem>

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {
class DmaBufRemoteRunTests : public ov::test::behavior::OVPluginTestBase,
                             public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;
    ov::CompiledModel compiled_model;
    int _fd_dma_heap = -1;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        targetDevice = ov::test::utils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
        ov_model = getDefaultNGraphFunctionForTheDeviceNPU();
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }

        if (_fd_dma_heap != -1) {
            close(_fd_dma_heap);
        }

        APIBaseTest::TearDown();
    }

    int32_t getFdDmaHeap(const size_t byte_size) {
        if (!std::filesystem::exists("/dev/dma_heap/system")) {
            OPENVINO_THROW("Cannot open /dev/dma_heap/system file.");
        }

        _fd_dma_heap = open("/dev/dma_heap/system", O_RDWR);
        if (_fd_dma_heap == -1) {
            OPENVINO_THROW("Cannot open /dev/dma_heap/system.");
        }

        static const std::size_t alignment = 4096;
        size_t size = byte_size + alignment - (byte_size % alignment);
        struct dma_heap_allocation_data heapAlloc = {
            .len = size,  // this length should be alligned to the page size
            .fd = 0,
            .fd_flags = O_RDWR | O_CLOEXEC,
            .heap_flags = 0,
        };

        auto ret = ioctl(_fd_dma_heap, DMA_HEAP_IOCTL_ALLOC, &heapAlloc);
        if (ret != 0) {
            OPENVINO_THROW("Cannot initialize DMA heap");
        }

        return static_cast<int32_t>(heapAlloc.fd);
    }
};

TEST_P(DmaBufRemoteRunTests, CheckRemoteTensorSharedBuf) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(tensor.get_shape()));

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    auto fd_heap = getFdDmaHeap(byte_size);

    auto mmap_ret = mmap(NULL, byte_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd_heap, 0);
    if (mmap_ret == MAP_FAILED) {
        ASSERT_FALSE(true) << "mmap failed.";
    }

    auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape(), fd_heap);

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    close(fd_heap);
}

TEST_P(DmaBufRemoteRunTests, CheckRemoteTensorSharedBuChangingTensors) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(tensor.get_shape()));

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    auto fd_heap = getFdDmaHeap(byte_size);

    auto mmap_ret = mmap(NULL, byte_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd_heap, 0);
    if (mmap_ret == MAP_FAILED) {
        ASSERT_FALSE(true) << "mmap failed.";
    }

    auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape(), fd_heap);

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random input tensor
    float* random_buffer_tensor = new float[byte_size / sizeof(float)];
    memset(random_buffer_tensor, 1, byte_size);
    ov::Tensor random_tensor_input{ov::element::f32, tensor.get_shape(), random_buffer_tensor};

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(random_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random output tensor
    auto output_tensor = inference_request.get_output_tensor();
    const auto output_byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(output_tensor.get_shape()));

    float* output_random_buffer_tensor = new float[output_byte_size / sizeof(float)];
    memset(output_random_buffer_tensor, 1, output_byte_size);
    ov::Tensor outputrandom_tensor_input{ov::element::f32, output_tensor.get_shape(), output_random_buffer_tensor};

    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(outputrandom_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    delete[] random_buffer_tensor;
    delete[] output_random_buffer_tensor;
    close(fd_heap);
}

TEST_P(DmaBufRemoteRunTests, CheckOutputDataFromMultipleRuns) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::InferRequest inference_request;
    float* data;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    auto shape = tensor.get_shape();
    const auto byte_size = ov::element::get_memory_size(ov::element::f32, shape_size(shape));
    tensor = {};

    auto fd_heap = getFdDmaHeap(byte_size);

    auto mmap_ret = mmap(NULL, byte_size, PROT_WRITE | PROT_READ, MAP_SHARED, fd_heap, 0);
    if (mmap_ret == MAP_FAILED) {
        ASSERT_FALSE(true) << "mmap failed.";
    }

    memset(mmap_ret, 99, byte_size);

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    auto output_tensor = inference_request.get_output_tensor();
    const auto output_byte_size = output_tensor.get_byte_size();
    float* output_data_one = new float[output_byte_size / sizeof(float)];
    ov::Tensor output_data_tensor_one{ov::element::f32, output_tensor.get_shape(), output_data_one};

    auto remote_tensor = context.create_tensor(ov::element::f32, shape, fd_heap);
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_data_tensor_one));
    OV_ASSERT_NO_THROW(inference_request.infer());

    float* output_data_two = new float[output_byte_size / sizeof(float)];
    ov::Tensor output_data_tensor_two{ov::element::f32, output_tensor.get_shape(), output_data_two};

    data = new float[byte_size / sizeof(float)];
    memset(data, 99, byte_size);
    ov::Tensor input_data_tensor{ov::element::f32, shape, data};
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_data_tensor_two));
    OV_ASSERT_NO_THROW(inference_request.infer());

    EXPECT_NE(output_data_one, output_data_two);
    EXPECT_EQ(memcmp(output_data_one, output_data_two, output_byte_size), 0);

    delete[] data;
    delete[] output_data_one;
    delete[] output_data_two;

    close(fd_heap);
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

#    endif
#endif
