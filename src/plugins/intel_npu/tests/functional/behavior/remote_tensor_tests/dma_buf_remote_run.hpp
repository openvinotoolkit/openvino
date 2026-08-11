// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "ov_infer_request/infer_request_dynamic_utils.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

#ifdef __linux__
#    include <linux/version.h>
#    if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 10, 0)
#        include <fcntl.h>
#        include <linux/dma-heap.h>
#        include <sys/ioctl.h>
#        include <sys/mman.h>
#        include <unistd.h>

#        include <cstdint>
#        include <cstring>
#        include <filesystem>
#        include <map>
#        include <string>
#        include <vector>

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

namespace ov {
namespace test {
namespace behavior {

// RAII owner of a DMA-BUF system-heap allocation: the heap device fd, the exported buffer fd and the
// mmap-ed CPU view of it. Construction either yields a fully mapped buffer or throws, so the object is
// never observable half-initialized.
class DmaHeapBuffer {
public:
    explicit DmaHeapBuffer(const size_t byte_size) : _byte_size(roundUpToPageSize(byte_size)) {
        if (!std::filesystem::exists(DEVICE_PATH)) {
            OPENVINO_THROW("Cannot open ", DEVICE_PATH, " file.");
        }

        _fd_dma_heap = open(DEVICE_PATH, O_RDWR);
        if (_fd_dma_heap == -1) {
            OPENVINO_THROW("Cannot open ", DEVICE_PATH, ".");
        }

        dma_heap_allocation_data heap_alloc = {
            .len = _byte_size,  // the DMA heap only accepts page-aligned allocations
            .fd = 0,
            .fd_flags = O_RDWR | O_CLOEXEC,
            .heap_flags = 0,
        };
        if (ioctl(_fd_dma_heap, DMA_HEAP_IOCTL_ALLOC, &heap_alloc) != 0) {
            close(_fd_dma_heap);
            OPENVINO_THROW("Cannot initialize DMA heap.");
        }
        _fd = static_cast<int32_t>(heap_alloc.fd);

        _data = mmap(nullptr, _byte_size, PROT_WRITE | PROT_READ, MAP_SHARED, _fd, 0);
        if (_data == MAP_FAILED) {
            close(_fd);
            close(_fd_dma_heap);
            OPENVINO_THROW("mmap failed.");
        }
    }

    DmaHeapBuffer(const DmaHeapBuffer&) = delete;
    DmaHeapBuffer& operator=(const DmaHeapBuffer&) = delete;

    ~DmaHeapBuffer() {
        munmap(_data, _byte_size);
        close(_fd);
        close(_fd_dma_heap);
    }

    int32_t fd() const {
        return _fd;
    }

    void* data() const {
        return _data;
    }

private:
    static constexpr const char* DEVICE_PATH = "/dev/dma_heap/system";
    static constexpr size_t PAGE_ALIGNMENT = 4096;

    static size_t roundUpToPageSize(const size_t byte_size) {
        return (byte_size + PAGE_ALIGNMENT - 1) & ~(PAGE_ALIGNMENT - 1);
    }

    size_t _byte_size;
    int _fd_dma_heap = -1;
    int32_t _fd = -1;
    void* _data = nullptr;
};

class DmaBufRemoteRunTests : public ov::test::behavior::OVPluginTestBase,
                             public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> ov_model;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
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

        APIBaseTest::TearDown();
    }
};

TEST_P(DmaBufRemoteRunTests, CheckRemoteTensorSharedBuf) {
    // Skip test according to plugin specific disabled_test_patterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel compiled_model;
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(tensor.get_shape()));

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    DmaHeapBuffer buffer(byte_size);

    auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape(), buffer.fd());

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(DmaBufRemoteRunTests, CheckRemoteTensorSharedBufChangingTensors) {
    // Skip test according to plugin specific disabled_test_patterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::CompiledModel compiled_model;
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(tensor.get_shape()));

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    DmaHeapBuffer buffer(byte_size);

    auto remote_tensor = context.create_tensor(ov::element::f32, tensor.get_shape(), buffer.fd());

    ov::Tensor check_remote_tensor;
    ASSERT_NO_THROW(check_remote_tensor = remote_tensor);
    ASSERT_THROW(check_remote_tensor.data(), ov::Exception);

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(check_remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random input tensor
    std::vector<float> random_buffer_tensor(byte_size / sizeof(float));
    memset(random_buffer_tensor.data(), 1, byte_size);
    ov::Tensor random_tensor_input{ov::element::f32, tensor.get_shape(), random_buffer_tensor.data()};

    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(random_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // set random output tensor
    auto output_tensor = inference_request.get_output_tensor();
    const auto output_byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(output_tensor.get_shape()));

    std::vector<float> output_random_buffer_tensor(output_byte_size / sizeof(float));
    memset(output_random_buffer_tensor.data(), 1, output_byte_size);
    ov::Tensor output_random_tensor_input{ov::element::f32,
                                          output_tensor.get_shape(),
                                          output_random_buffer_tensor.data()};

    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_random_tensor_input));
    OV_ASSERT_NO_THROW(inference_request.infer());
}

TEST_P(DmaBufRemoteRunTests, CheckOutputDataFromMultipleRuns) {
    // Skip test according to plugin specific disabled_test_patterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ov::CompiledModel compiled_model;
    ov::InferRequest inference_request;

    OV_ASSERT_NO_THROW(compiled_model = core->compile_model(ov_model, target_device, configuration));
    OV_ASSERT_NO_THROW(inference_request = compiled_model.create_infer_request());
    auto tensor = inference_request.get_input_tensor();

    const auto shape = tensor.get_shape();
    const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(shape));
    tensor = {};

    DmaHeapBuffer buffer(byte_size);
    memset(buffer.data(), 99, byte_size);

    auto context = core->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();

    auto output_tensor = inference_request.get_output_tensor();
    const auto output_byte_size = output_tensor.get_byte_size();
    std::vector<float> output_data_one(output_byte_size / sizeof(float));
    ov::Tensor output_data_tensor_one{ov::element::f32, output_tensor.get_shape(), output_data_one.data()};

    auto remote_tensor = context.create_tensor(ov::element::f32, shape, buffer.fd());
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(remote_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_data_tensor_one));
    OV_ASSERT_NO_THROW(inference_request.infer());

    // the same input data, this time passed through a host tensor, must yield the same output
    std::vector<float> output_data_two(output_byte_size / sizeof(float));
    ov::Tensor output_data_tensor_two{ov::element::f32, output_tensor.get_shape(), output_data_two.data()};

    std::vector<float> data(byte_size / sizeof(float));
    memset(data.data(), 99, byte_size);
    ov::Tensor input_data_tensor{ov::element::f32, shape, data.data()};
    OV_ASSERT_NO_THROW(inference_request.set_input_tensor(input_data_tensor));
    OV_ASSERT_NO_THROW(inference_request.set_output_tensor(output_data_tensor_two));
    OV_ASSERT_NO_THROW(inference_request.infer());

    EXPECT_NE(output_data_one.data(), output_data_two.data());
    EXPECT_EQ(memcmp(output_data_one.data(), output_data_two.data(), output_byte_size), 0);
}

// Runs a dynamically-shaped model with a DMA-BUF backed remote tensor as input. A distinct type -- not an
// alias -- so that gtest registers it as its own suite and it can be instantiated independently.
class DmaBufRemoteRunDynamicTests : public InferRequestDynamicTests {};

TEST_P(DmaBufRemoteRunDynamicTests, InferDynamicNetworkRemoteTensor) {
    // Skip test according to plugin specific disabled_test_patterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const std::string inputName = "Parameter_1";
    const std::string outputName = "Relu_2";

    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(1, inOutShapes[1].first[0]),
                         ov::Dimension(1, inOutShapes[1].first[1]),
                         ov::Dimension(1, inOutShapes[1].first[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto context = ie->get_default_context(target_device).as<ov::intel_npu::level_zero::ZeroContext>();
    auto compiled_model = ie->compile_model(function, target_device, configuration);

    // infer the same shape twice to exercise reuse of the compiled model across remote tensors
    const std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].first};
    for (auto& shape : vectorShapes) {
        ov::Tensor in_tensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);

        const auto byte_size = ov::util::get_memory_size(ov::element::f32, shape_size(shape));

        DmaHeapBuffer buffer(byte_size);
        memcpy(buffer.data(), in_tensor.data(), byte_size);

        auto remote_tensor = context.create_tensor(ov::element::f32, shape, buffer.fd());

        ov::InferRequest req;
        OV_ASSERT_NO_THROW(req = compiled_model.create_infer_request());
        OV_ASSERT_NO_THROW(req.set_tensor(inputName, remote_tensor));
        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(in_tensor, req.get_tensor(outputName)));
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov

#    endif
#endif
