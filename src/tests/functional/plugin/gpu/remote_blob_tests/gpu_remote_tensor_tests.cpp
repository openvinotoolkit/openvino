// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/core.hpp"

#include <gpu/gpu_config.hpp>
#include <remote_blob_tests/remote_blob_helpers.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/plugin_cache.hpp>
#include "base/ov_behavior_test_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "transformations/utils/utils.hpp"

using namespace ::testing;

class OVRemoteTensor_Test : public CommonTestUtils::TestsCommon {
protected:
    std::shared_ptr<ngraph::Function> fn_ptr;

    void SetUp() override {
        fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    }
};

std::vector<bool> ov_with_auto_batching {true, false};
enum class RemoteTensorSharingType {
    USER_CL_TENSOR = 0,
    PLUGIN_CL_TENSOR = 1,
    USER_USM_HOST_TENSOR = 2,
    USER_USM_DEVICE_TENSOR = 3,
    PLUGIN_USM_HOST_TENSOR = 4,
    PLUGIN_USM_DEVICE_TENSOR = 5,
    PLUGIN_HOST_TENSOR = 6
};

std::ostream& operator<<(std::ostream& stream, RemoteTensorSharingType sharing_type) {
    switch (sharing_type) {
    case RemoteTensorSharingType::USER_CL_TENSOR:  stream << "USER_CL_TENSOR"; break;
    case RemoteTensorSharingType::PLUGIN_CL_TENSOR: stream << "PLUGIN_CL_TENSOR"; break;
    case RemoteTensorSharingType::USER_USM_HOST_TENSOR: stream << "USER_USM_HOST_TENSOR"; break;
    case RemoteTensorSharingType::USER_USM_DEVICE_TENSOR: stream << "USER_USM_DEVICE_TENSOR"; break;
    case RemoteTensorSharingType::PLUGIN_USM_HOST_TENSOR: stream << "PLUGIN_USM_HOST_TENSOR"; break;
    case RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR: stream << "PLUGIN_USM_DEVICE_TENSOR"; break;
    case RemoteTensorSharingType::PLUGIN_HOST_TENSOR: stream << "PLUGIN_HOST_TENSOR"; break;
    }

    return stream;
}

using RemoteTensorSharingTestOptionsParams = std::tuple<RemoteTensorSharingType, bool /*auto-batching*/>;

class OVRemoteTensorInputBlob_Test : public OVRemoteTensor_Test,
        public testing::WithParamInterface<RemoteTensorSharingTestOptionsParams> {
protected:
    std::shared_ptr<ngraph::Function> fn_ptr;
    std::string deviceName;
    ov::AnyMap config;

public:
    void SetUp() override {
        deviceName = CommonTestUtils::DEVICE_GPU;
        RemoteTensorSharingType sharing_type;
        bool with_auto_batching;
        std::tie(sharing_type, with_auto_batching) = this->GetParam();
        if (with_auto_batching) {
            config =
                    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                     // immediate timeout to avoid increasing the test time
                     ov::auto_batch_timeout(0)
                    };
        }
        fn_ptr = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(with_auto_batching ? CommonTestUtils::DEVICE_BATCH : deviceName);
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RemoteTensorSharingTestOptionsParams>& obj) {
        RemoteTensorSharingType sharing_type;
        bool with_auto_batching;
        std::tie(sharing_type, with_auto_batching) = obj.param;

        std::ostringstream result;
        result << "OVRemoteTensorInputBlob_Test_";
        result << sharing_type;
        if (with_auto_batching)
            result << "_WITH_AUTO_BATCHING";
        return result.str();
    }
};

TEST_P(OVRemoteTensorInputBlob_Test, smoke_cantCreateBlobWithInvalidSize) {
    RemoteTensorSharingType sharing_type;
    bool with_auto_batching;
    std::tie(sharing_type, with_auto_batching) = GetParam();
    if (with_auto_batching)
        GTEST_SKIP();

    if (sharing_type == RemoteTensorSharingType::PLUGIN_CL_TENSOR ||
        sharing_type == RemoteTensorSharingType::PLUGIN_USM_HOST_TENSOR ||
        sharing_type == RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR ||
        sharing_type == RemoteTensorSharingType::PLUGIN_HOST_TENSOR)
        GTEST_SKIP();

    auto ie = ov::Core();
    auto cldnn_context = ie.get_default_context(deviceName).as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    ov::Shape invalid_shape = {1, 20, 30, 40};

    auto imSize = ov::shape_size(ov::Shape({1, 2, 3, 4}));

    switch (sharing_type) {
        case RemoteTensorSharingType::USER_CL_TENSOR: {
            cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imSize, NULL, &err);
            ASSERT_ANY_THROW(cldnn_context.create_tensor(ov::element::i8, invalid_shape, shared_buffer));
            break;
        }
        case RemoteTensorSharingType::USER_USM_DEVICE_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            void* shared_buffer = ocl_instance->allocate_usm_device_buffer(imSize);
            ASSERT_ANY_THROW(cldnn_context.create_tensor(ov::element::i8, invalid_shape, shared_buffer));
            ocl_instance->free_mem(shared_buffer);
            break;
        }
        case RemoteTensorSharingType::USER_USM_HOST_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            void* shared_buffer = ocl_instance->allocate_usm_host_buffer(imSize);
            ASSERT_ANY_THROW(cldnn_context.create_tensor(ov::element::i8, invalid_shape, shared_buffer));
            ocl_instance->free_mem(shared_buffer);
            break;
        }
        default: break;
    }
}

TEST_P(OVRemoteTensorInputBlob_Test, smoke_canInputRemoteTensor) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto ie = ov::Core();

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr);
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto function = p.build();
    RemoteTensorSharingType sharing_type;
    bool with_auto_batching;
    std::tie(sharing_type, with_auto_batching) = GetParam();

    // auto-batching relies on availability of the lock() for the tensor (and the *USM_DEVICE is not lockable)
    if (with_auto_batching
            && (RemoteTensorSharingType::USER_USM_DEVICE_TENSOR == sharing_type
                    || RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR == sharing_type))
        GTEST_SKIP();

    auto exec_net = ie.compile_model(function, deviceName, config);

    // regular inference
    auto inf_req_regular = exec_net.create_infer_request();
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());

    inf_req_regular.set_tensor(input, fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(output);

    // inference using remote tensor
    auto inf_req_shared = exec_net.create_infer_request();
    auto cldnn_context = exec_net.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    auto imSize = ov::shape_size(input->get_shape());

    switch (sharing_type) {
        case RemoteTensorSharingType::USER_CL_TENSOR: {
            cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imSize, NULL, &err);
            {
                void* buffer = fakeImageData.data();
                ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
            }

            auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape(), shared_buffer);
            inf_req_shared.set_tensor(input, cldnn_tensor);
            inf_req_shared.infer();

            break;
        }
        case RemoteTensorSharingType::USER_USM_DEVICE_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            void* shared_buffer = ocl_instance->allocate_usm_device_buffer(imSize);
            {
                void* buffer = fakeImageData.data();
                err = ocl_instance->memcpy(ocl_instance->_queue, shared_buffer, buffer, imSize, true, nullptr, nullptr);
                if (err != CL_SUCCESS)
                    FAIL() << "Failed to copy data from host buffer to USM device";
            }

            auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape(), shared_buffer);
            inf_req_shared.set_tensor(input, cldnn_tensor);
            inf_req_shared.infer();

            ocl_instance->free_mem(shared_buffer);

            break;
        }
        case RemoteTensorSharingType::USER_USM_HOST_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            void* shared_buffer = ocl_instance->allocate_usm_host_buffer(imSize);
            {
                void* buffer = fakeImageData.data();
                std::memcpy(shared_buffer, buffer, imSize);
            }

            auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape(), shared_buffer);
            inf_req_shared.set_tensor(input, cldnn_tensor);
            inf_req_shared.infer();

            ocl_instance->free_mem(shared_buffer);

            break;
        }
        case RemoteTensorSharingType::PLUGIN_CL_TENSOR: {
            auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape());
            ASSERT_TRUE(cldnn_tensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
            auto cl_tensor = cldnn_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();
            {
                cl::Buffer shared_buffer = cl_tensor;
                void* buffer = fakeImageData.data();
                ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
            }
            inf_req_shared.set_tensor(input, cldnn_tensor);
            inf_req_shared.infer();
            break;
        }
        case RemoteTensorSharingType::PLUGIN_USM_HOST_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            auto cldnn_tensor = cldnn_context.create_usm_host_tensor(input->get_element_type(), input->get_shape());
            ASSERT_TRUE(cldnn_tensor.is<ov::intel_gpu::ocl::USMTensor>());
            {
                auto cl_tensor = cldnn_tensor.as<ov::intel_gpu::ocl::USMTensor>();
                void* shared_buffer = cl_tensor.get();
                ASSERT_EQ(ocl_instance->get_allocation_type(shared_buffer), CL_MEM_TYPE_HOST_INTEL);
                void* buffer = fakeImageData.data();
                std::memcpy(shared_buffer, buffer, imSize);
            }

            inf_req_shared.set_tensor(input, cldnn_tensor);
            inf_req_shared.infer();

            break;
        }
        case RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            auto cldnn_tensor = cldnn_context.create_usm_device_tensor(input->get_element_type(), input->get_shape());
            ASSERT_TRUE(cldnn_tensor.is<ov::intel_gpu::ocl::USMTensor>());
            {
                auto cl_tensor = cldnn_tensor.as<ov::intel_gpu::ocl::USMTensor>();
                void* shared_buffer = cl_tensor.get();
                ASSERT_EQ(ocl_instance->get_allocation_type(shared_buffer), CL_MEM_TYPE_DEVICE_INTEL);
                void* buffer = fakeImageData.data();
                err = ocl_instance->memcpy(ocl_instance->_queue, shared_buffer, buffer, imSize, true, nullptr, nullptr);
                if (err != CL_SUCCESS)
                    FAIL() << "Failed to copy data from host buffer to USM device";
            }

            inf_req_shared.set_tensor(input, cldnn_tensor);
            inf_req_shared.infer();

            break;
        }
        case RemoteTensorSharingType::PLUGIN_HOST_TENSOR: {
            auto cldnn_tensor = cldnn_context.create_host_tensor(input->get_element_type(), input->get_shape());
            {
                ASSERT_NO_THROW(cldnn_tensor.data());
                void* shared_buffer = cldnn_tensor.data();
                if (ocl_instance->supports_usm())
                    ASSERT_EQ(ocl_instance->get_allocation_type(shared_buffer), CL_MEM_TYPE_HOST_INTEL);
                void* buffer = fakeImageData.data();
                std::memcpy(shared_buffer, buffer, imSize);
            }

            inf_req_shared.set_tensor(input, cldnn_tensor);
            inf_req_shared.infer();

            break;
        }
    }

    auto output_tensor_shared = inf_req_shared.get_tensor(output);

    // compare results
    {
        ASSERT_EQ(output->get_element_type(), ov::element::f32);
        ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        ASSERT_NO_THROW(output_tensor_regular.data());
        ASSERT_NO_THROW(output_tensor_shared.data());
        FuncTestUtils::compare_tensor(output_tensor_regular, output_tensor_shared, thr);
    }
}

INSTANTIATE_TEST_SUITE_P(
    smoke_GPU,
    OVRemoteTensorInputBlob_Test,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<RemoteTensorSharingType>{RemoteTensorSharingType::USER_CL_TENSOR,
                                                                 RemoteTensorSharingType::PLUGIN_CL_TENSOR,
                                                                 RemoteTensorSharingType::USER_USM_HOST_TENSOR,
                                                                 RemoteTensorSharingType::USER_USM_DEVICE_TENSOR,
                                                                 RemoteTensorSharingType::PLUGIN_USM_HOST_TENSOR,
                                                                 RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR,
                                                                 RemoteTensorSharingType::PLUGIN_HOST_TENSOR}),
        ::testing::ValuesIn(ov_with_auto_batching)),
        OVRemoteTensorInputBlob_Test::getTestCaseName);

class OVRemoteTensor_TestsWithContext : public OVRemoteTensor_Test, public testing::WithParamInterface<bool> {
protected:
    std::shared_ptr<ngraph::Function> fn_ptr;
    std::string deviceName;
    ov::AnyMap config;

public:
    void SetUp() override {
        fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
        deviceName = CommonTestUtils::DEVICE_GPU;
        auto with_auto_batching = this->GetParam();
        if (with_auto_batching) {
            config =
                    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                            // immediate timeout to avoid increasing the test time
                     ov::auto_batch_timeout(0)
                    };
        }
        fn_ptr = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(with_auto_batching ? CommonTestUtils::DEVICE_BATCH : deviceName);
    }
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        auto with_auto_batch = obj.param;
        return std::string("RemoteTensor_Test") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
    }
};

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserContext) {
    auto ie = ov::Core();

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr);
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);
    auto function = p.build();

    auto exec_net_regular = ie.compile_model(function, deviceName);
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);

    // regular inference
    auto inf_req_regular = exec_net_regular.create_infer_request();
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
    inf_req_regular.set_tensor(input, fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // inference using remote tensor
    auto ocl_instance = std::make_shared<OpenCL>();

    auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_context.get());
    auto exec_net_shared = ie.compile_model(function, remote_context, config);
    auto inf_req_shared = exec_net_shared.create_infer_request();
    inf_req_shared.set_tensor(input, fakeImageData);

    inf_req_shared.infer();
    auto output_tensor_shared = inf_req_shared.get_tensor(output);

    // compare results
    {
        ASSERT_EQ(output->get_element_type(), ov::element::f32);
        ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        ASSERT_NO_THROW(output_tensor_regular.data());
        ASSERT_NO_THROW(output_tensor_shared.data());
        FuncTestUtils::compare_tensor(output_tensor_regular, output_tensor_shared, thr);
    }
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserContextWithMultipleDevices) {
    auto ie = ov::Core();

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr);
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);
    auto function = p.build();

    auto exec_net_regular = ie.compile_model(function, deviceName);
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);

    // regular inference
    auto inf_req_regular = exec_net_regular.create_infer_request();
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
    inf_req_regular.set_tensor(input, fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // inference using remote tensor

    auto ocl_instance_tmp = std::make_shared<OpenCL>();
    cl::Context multi_device_ctx({ocl_instance_tmp->_device, ocl_instance_tmp->_device});
    auto ocl_instance = std::make_shared<OpenCL>(multi_device_ctx.get());

    auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_context.get(), 1);

    ASSERT_EQ(remote_context.get_device_name(), "GPU.0");
    auto exec_net_shared = ie.compile_model(function, remote_context, config);
    auto inf_req_shared = exec_net_shared.create_infer_request();
    inf_req_shared.set_tensor(input, fakeImageData);

    inf_req_shared.infer();
    auto output_tensor_shared = inf_req_shared.get_tensor(output);

    // compare results
    {
        ASSERT_EQ(output->get_element_type(), ov::element::f32);
        ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        ASSERT_NO_THROW(output_tensor_regular.data());
        ASSERT_NO_THROW(output_tensor_shared.data());
        FuncTestUtils::compare_tensor(output_tensor_regular, output_tensor_shared, thr);
    }
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserQueue_out_of_order) {
    auto ie = ov::Core();

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr);
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);
    auto function = p.build();

    auto exec_net_regular = ie.compile_model(function, deviceName);
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);

    // regular inference
    auto inf_req_regular = exec_net_regular.create_infer_request();
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
    inf_req_regular.set_tensor(input, fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    auto in_size = ov::shape_size(input->get_output_shape(0)) * input->get_output_element_type(0).size();
    auto out_size = ov::shape_size(output->get_output_shape(0)) * output->get_output_element_type(0).size();

    // inference using remote tensor
    auto ocl_instance = std::make_shared<OpenCL>();
    cl_int err;

    // Allocate shared buffers for input and output data which will be set to infer request
    cl::Buffer shared_input_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size, NULL, &err);
    cl::Buffer shared_output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);

    auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_queue.get());
    auto exec_net_shared = ie.compile_model(function, remote_context); // no auto-batching support, so no config is passed
    auto gpu_context = exec_net_shared.get_context().as<ov::intel_gpu::ocl::ClContext>();

    auto gpu_in_tensor = gpu_context.create_tensor(input->get_output_element_type(0), input->get_output_shape(0), shared_input_buffer);
    auto gpu_out_tensor = gpu_context.create_tensor(output->get_output_element_type(0), output->get_output_shape(0), shared_output_buffer);
    auto out_tensor = FuncTestUtils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

    auto inf_req_shared = exec_net_shared.create_infer_request();
    inf_req_shared.set_tensor(input, gpu_in_tensor);
    inf_req_shared.set_tensor(output, gpu_out_tensor);

    // 1. Pre-processing. Enqueue non-blocking copy from host ptr to shared device input buffer and barrier to ensure that copy is finished before
    // inference primitives starts execution
    {
        void* buffer = fakeImageData.data();
        ocl_instance->_queue.enqueueWriteBuffer(shared_input_buffer, false, 0, in_size, buffer);
        ocl_instance->_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
    }

    // 2. Enqueue inference primitives. With shared queue this call ensures that all kernels are scheduled to the corresponding queue
    // before giving the control back
    inf_req_shared.start_async();

    // 3. Post-processing. Enqueue copy from shared blob with inference result to another output blob
    // Enqueue barrier with empty wait list is needed to ensure that previous kernels are finished before copying the data. It's needed here since we
    // create OOO queue.
    // Note: inf_req_shared.wait() can be dropped in some cases, but if plugin-side post-processing is required,
    // then the result may be incorrect without Wait().
    {
        ocl_instance->_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
        ocl_instance->_queue.enqueueReadBuffer(shared_output_buffer, false, 0, out_size, out_tensor.data(), nullptr, nullptr);
    }

    // 4. Wait for infer request and post-processing completion
    ocl_instance->_queue.finish();

    // compare results
    {
        ASSERT_EQ(output->get_element_type(), ov::element::f32);
        ASSERT_EQ(output_tensor_regular.get_size(), out_tensor.get_size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        ASSERT_NO_THROW(output_tensor_regular.data());
        FuncTestUtils::compare_tensor(output_tensor_regular, out_tensor, thr);
    }
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserQueue_in_order) {
    auto ie = ov::Core();

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr);
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);
    auto function = p.build();

    auto exec_net_regular = ie.compile_model(function, deviceName);
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);

    // regular inference
    auto inf_req_regular = exec_net_regular.create_infer_request();
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
    inf_req_regular.set_tensor(input, fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    auto in_size = ov::shape_size(input->get_output_shape(0)) * input->get_output_element_type(0).size();
    auto out_size = ov::shape_size(output->get_output_shape(0)) * output->get_output_element_type(0).size();

    // inference using remote tensor
    auto ocl_instance = std::make_shared<OpenCL>();
    ocl_instance->_queue = cl::CommandQueue(ocl_instance->_context, ocl_instance->_device);
    cl_int err;

    // Allocate shared buffers for input and output data which will be set to infer request
    cl::Buffer shared_input_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size, NULL, &err);
    cl::Buffer shared_output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);

    auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_queue.get());
    auto exec_net_shared = ie.compile_model(function, remote_context); // no auto-batching support, so no config is passed
    auto gpu_context = exec_net_shared.get_context().as<ov::intel_gpu::ocl::ClContext>();

    auto gpu_in_tensor = gpu_context.create_tensor(input->get_output_element_type(0), input->get_output_shape(0), shared_input_buffer);
    auto gpu_out_tensor = gpu_context.create_tensor(output->get_output_element_type(0), output->get_output_shape(0), shared_output_buffer);
    auto out_tensor = FuncTestUtils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

    auto inf_req_shared = exec_net_shared.create_infer_request();
    inf_req_shared.set_tensor(input, gpu_in_tensor);
    inf_req_shared.set_tensor(output, gpu_out_tensor);

    // 1. Pre-processing. Enqueue non-blocking copy from host ptr to shared device input buffer
    {
        void* buffer = fakeImageData.data();
        ocl_instance->_queue.enqueueWriteBuffer(shared_input_buffer, false, 0, in_size, buffer);
    }

    // 2. Enqueue inference primitives. With shared queue this call ensures that all kernels are scheduled to the corresponding queue
    // before giving the control back
    inf_req_shared.start_async();

    // 3. Post-processing. Enqueue copy from shared blob with inference result to another output blob
    // Note: inf_req_shared.Wait() can be dropped in some cases, but if plugin-side post-processing is required,
    // then the result may be incorrect without Wait().
    {
        ocl_instance->_queue.enqueueReadBuffer(shared_output_buffer, false, 0, out_size, out_tensor.data(), nullptr, nullptr);
    }

    // 4. Wait for infer request and post-processing completion
    ocl_instance->_queue.finish();

    // compare results
    {
        ASSERT_EQ(output->get_element_type(), ov::element::f32);
        ASSERT_EQ(output_tensor_regular.get_size(), out_tensor.get_size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        ASSERT_NO_THROW(output_tensor_regular.data());
        FuncTestUtils::compare_tensor(output_tensor_regular, out_tensor, thr);
    }
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canCreateManyTensorsOnSameMem) {
    bool with_auto_batching = GetParam();
    if (with_auto_batching)
        GTEST_SKIP();

    auto ocl_instance = std::make_shared<OpenCL>();
    if (!ocl_instance->supports_usm())
        GTEST_SKIP();

    auto input = fn_ptr->get_parameters().at(0);
    ov::Shape input_shape = input->get_shape();
    auto imSize = ov::shape_size(input_shape);
    void* usm_ptr = usm_ptr = ocl_instance->allocate_usm_host_buffer(imSize*sizeof(float));

    auto ie = ov::Core();
    auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_context.get());
    auto model = ie.compile_model(fn_ptr, remote_context, config);
    auto infer_request = model.create_infer_request();
    for (int i = 0; i < 10; ++i) {
        auto input = model.input();
        auto cl_context = model.get_context().as<ov::intel_gpu::ocl::ClContext>();
        ov::RemoteTensor input_tensor = cl_context.create_tensor(
            input.get_element_type(), input.get_shape(), usm_ptr);
        infer_request.set_tensor(input.get_any_name(), input_tensor);
        infer_request.start_async();
        infer_request.wait();
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVRemoteTensor_TestsWithContext, ::testing::ValuesIn(ov_with_auto_batching),
                         OVRemoteTensor_TestsWithContext::getTestCaseName);

TEST_F(OVRemoteTensor_Test, NV12toBGR_image_ConvertTranspose) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 8;
    const int width = 8;

    // ------------------------------------------------------
    // Prepare input data
    ov::Tensor fake_image_data_y = FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 1, height, width}, 50, 0, 1);
    ov::Tensor fake_image_data_uv = FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 2, height / 2, width / 2}, 256, 0, 1);

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvertTranspose({1, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(GPU_CONFIG_KEY(SURFACE));
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);
    auto param_input_uv = fn_ptr_remote->get_parameters().at(1);

    auto exec_net_b = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
    auto inf_req_remote = exec_net_b.create_infer_request();

    auto cldnn_context = exec_net_b.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context.get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    cl_image_format image_format;
    cl_image_desc image_desc = { 0 };
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    cl_mem nv12_image_plane_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    ASSERT_EQ(err, 0);

    image_format.image_channel_order = CL_RG;
    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;
    cl_mem nv12_image_plane_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    ASSERT_EQ(err, 0);

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)height, 1 };
    size_t uv_region[3] = { (size_t)width / 2, (size_t)height / 2, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y,
        true, origin, y_region, 0, 0, fake_image_data_y.data(), 0, NULL, NULL);
    ASSERT_EQ(err, 0);

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv,
        true, origin, uv_region, 0, 0, fake_image_data_uv.data(), 0, NULL, NULL);
    ASSERT_EQ(err, 0);

    cl::Image2D img_y = cl::Image2D(nv12_image_plane_y);
    cl::Image2D img_uv = cl::Image2D(nv12_image_plane_uv);

    auto tensor_remote_y = cldnn_context.create_tensor(param_input_y->get_element_type(), fake_image_data_y.get_shape(), img_y);
    auto tensor_remote_uv = cldnn_context.create_tensor(param_input_uv->get_element_type(), fake_image_data_uv.get_shape(), img_uv);

    inf_req_remote.set_tensor(*param_input_y->output(0).get_tensor().get_names().begin(), tensor_remote_y);
    inf_req_remote.set_tensor(*param_input_uv->output(0).get_tensor().get_names().begin(), tensor_remote_uv);

    inf_req_remote.infer();

    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ngraph::builder::subgraph::makeConvertTranspose({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(GPU_CONFIG_KEY(BUFFER));
    p_reg.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto exec_net_regular = ie.compile_model(function_regular, CommonTestUtils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();
    inf_req_regular.set_tensor(param_input_y, fake_image_data_y);
    inf_req_regular.set_tensor(param_input_uv, fake_image_data_uv);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // ------------------------------------------------------
    // compare results
    ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
    ASSERT_NO_THROW(output_tensor_regular.data());
    ASSERT_NO_THROW(output_tensor_shared.data());
    float thr = 0.1;
    FuncTestUtils::compare_tensor(output_tensor_shared, output_tensor_regular, thr);
}

TEST_F(OVRemoteTensor_Test, NV12toBGR_image) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    ov::Tensor fake_image_data_y = FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 1, height, width}, 50, 0, 1);
    ov::Tensor fake_image_data_uv = FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 2, height / 2, width / 2}, 256, 0, 1);

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(GPU_CONFIG_KEY(SURFACE));
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);
    auto param_input_uv = fn_ptr_remote->get_parameters().at(1);

    auto exec_net_b = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
    auto inf_req_remote = exec_net_b.create_infer_request();

    auto cldnn_context = exec_net_b.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context.get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    cl_image_format image_format;
    cl_image_desc image_desc = { 0 };
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    cl_mem nv12_image_plane_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    ASSERT_EQ(err, 0);

    image_format.image_channel_order = CL_RG;
    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;
    cl_mem nv12_image_plane_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    ASSERT_EQ(err, 0);

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)height, 1 };
    size_t uv_region[3] = { (size_t)width / 2, (size_t)height / 2, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y,
        true, origin, y_region, 0, 0, fake_image_data_y.data(), 0, NULL, NULL);
    ASSERT_EQ(err, 0);

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv,
        true, origin, uv_region, 0, 0, fake_image_data_uv.data(), 0, NULL, NULL);
    ASSERT_EQ(err, 0);

    cl::Image2D img_y = cl::Image2D(nv12_image_plane_y);
    cl::Image2D img_uv = cl::Image2D(nv12_image_plane_uv);

    auto tensor_remote_y = cldnn_context.create_tensor(param_input_y->get_element_type(), fake_image_data_y.get_shape(), img_y);
    auto tensor_remote_uv = cldnn_context.create_tensor(param_input_uv->get_element_type(), fake_image_data_uv.get_shape(), img_uv);

    inf_req_remote.set_tensor(*param_input_y->output(0).get_tensor().get_names().begin(), tensor_remote_y);
    inf_req_remote.set_tensor(*param_input_uv->output(0).get_tensor().get_names().begin(), tensor_remote_uv);

    inf_req_remote.infer();

    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(GPU_CONFIG_KEY(BUFFER));
    p_reg.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto exec_net_regular = ie.compile_model(function_regular, CommonTestUtils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();
    inf_req_regular.set_tensor(param_input_y, fake_image_data_y);
    inf_req_regular.set_tensor(param_input_uv, fake_image_data_uv);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // ------------------------------------------------------
    // compare results
    ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
    ASSERT_NO_THROW(output_tensor_regular.data());
    ASSERT_NO_THROW(output_tensor_shared.data());
    float thr = 0.1;
    FuncTestUtils::compare_tensor(output_tensor_shared, output_tensor_regular, thr);
}

TEST_F(OVRemoteTensor_Test, NV12toBGR_buffer) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    ov::Tensor fake_image_data_y = FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 1, height, width}, 50, 0, 1);
    ov::Tensor fake_image_data_uv = FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 2, height / 2, width / 2}, 256, 0, 1);

    auto ie = ov::Core();

    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(GPU_CONFIG_KEY(BUFFER));
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = function->get_parameters().at(0);
    auto param_input_uv = function->get_parameters().at(1);
    auto output = function->get_results().at(0);

    // ------------------------------------------------------
    // inference using remote tensor
    auto ocl_instance = std::make_shared<OpenCL>();
    ocl_instance->_queue = cl::CommandQueue(ocl_instance->_context, ocl_instance->_device);

    auto in_size_y = ov::shape_size(param_input_y->get_output_shape(0)) * param_input_y->get_output_element_type(0).size();
    auto in_size_uv = ov::shape_size(param_input_uv->get_output_shape(0)) * param_input_uv->get_output_element_type(0).size();
    auto out_size = ov::shape_size(output->get_output_shape(0)) * output->get_output_element_type(0).size();

    cl_int err;
    cl::Buffer shared_input_y_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size_y, NULL, &err);
    cl::Buffer shared_input_uv_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size_uv, NULL, &err);
    cl::Buffer shared_output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);

    auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_queue.get());
    auto exec_net_shared = ie.compile_model(function, remote_context);
    auto gpu_context = exec_net_shared.get_context().as<ov::intel_gpu::ocl::ClContext>();

    auto gpu_in_y_tensor = gpu_context.create_tensor(param_input_y->get_output_element_type(0), fake_image_data_y.get_shape(), shared_input_y_buffer);
    auto gpu_in_uv_tensor = gpu_context.create_tensor(param_input_uv->get_output_element_type(0), fake_image_data_uv.get_shape(), shared_input_uv_buffer);
    auto gpu_out_tensor = gpu_context.create_tensor(output->get_output_element_type(0), output->get_output_shape(0), shared_output_buffer);
    auto out_tensor = FuncTestUtils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

    auto inf_req_shared = exec_net_shared.create_infer_request();
    inf_req_shared.set_tensor(param_input_y, gpu_in_y_tensor);
    inf_req_shared.set_tensor(param_input_uv, gpu_in_uv_tensor);
    inf_req_shared.set_tensor(output, gpu_out_tensor);

    void* buffer_y = fake_image_data_y.data();
    void* buffer_uv = fake_image_data_uv.data();
    ocl_instance->_queue.enqueueWriteBuffer(shared_input_y_buffer, false, 0, in_size_y, buffer_y);
    ocl_instance->_queue.enqueueWriteBuffer(shared_input_uv_buffer, false, 0, in_size_uv, buffer_uv);

    inf_req_shared.start_async();

    ocl_instance->_queue.enqueueReadBuffer(shared_output_buffer, false, 0, out_size, out_tensor.data(), nullptr, nullptr);
    ocl_instance->_queue.finish();

    // ------------------------------------------------------
    // regular inference
    auto exec_net_regular = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();
    inf_req_regular.set_tensor(param_input_y, fake_image_data_y);
    inf_req_regular.set_tensor(param_input_uv, fake_image_data_uv);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // ------------------------------------------------------
    // compare results
    ASSERT_EQ(output_tensor_regular.get_size(), out_tensor.get_size());
    ASSERT_NO_THROW(output_tensor_regular.data());
    ASSERT_NO_THROW(out_tensor.data());
    float thr = 0.1;
    FuncTestUtils::compare_tensor(out_tensor, output_tensor_regular, thr);
}

class OVRemoteTensorBatched_Test : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<size_t> {
    void SetUp() override {
        num_batch = this->GetParam();
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::size_t> &obj) {
        return "num_batch_" + std::to_string(obj.param);
    }

protected:
    size_t num_batch;
    std::vector<std::shared_ptr<ngraph::Function>> fn_ptrs;
};

TEST_P(OVRemoteTensorBatched_Test, NV12toBGR_image) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    std::vector<ov::Tensor> fake_image_data_y, fake_image_data_uv;
    for (int i = 0; i < num_batch; i++) {
        fake_image_data_y.push_back(FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 1, height, width}, 50, 0, 1, i));
        fake_image_data_uv.push_back(FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 2, height / 2, width / 2}, 256, 0, 1, i));
    }

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({num_batch, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(GPU_CONFIG_KEY(SURFACE));
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);
    auto param_input_uv = fn_ptr_remote->get_parameters().at(1);

    auto exec_net_b = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
    auto inf_req_remote = exec_net_b.create_infer_request();

    auto cldnn_context = exec_net_b.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context.get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    std::vector<cl_mem> nv12_image_plane_y, nv12_image_plane_uv;
    std::vector<cl::Image2D> img_y, img_uv;
    std::vector<ov::Tensor> tensor_remote_y, tensor_remote_uv;

    for (size_t i = 0; i < num_batch; ++i) {
        cl_image_format image_format;
        cl_image_desc image_desc = { 0 };
        image_format.image_channel_order = CL_R;
        image_format.image_channel_data_type = CL_UNORM_INT8;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        image_desc.image_width = width;
        image_desc.image_height = height;
        nv12_image_plane_y.emplace_back(clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err));
        ASSERT_EQ(err, 0);

        image_format.image_channel_order = CL_RG;
        image_desc.image_width = width / 2;
        image_desc.image_height = height / 2;
        nv12_image_plane_uv.emplace_back(clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err));
        ASSERT_EQ(err, 0);

        size_t origin[3] = { 0, 0, 0 };
        size_t y_region[3] = { (size_t)width, (size_t)height, 1 };
        size_t uv_region[3] = { (size_t)width / 2, (size_t)height / 2, 1 };

        err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y[i],
            true, origin, y_region, 0, 0, fake_image_data_y[i].data(), 0, NULL, NULL);
        ASSERT_EQ(err, 0);

        err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv[i],
            true, origin, uv_region, 0, 0, fake_image_data_uv[i].data(), 0, NULL, NULL);
        ASSERT_EQ(err, 0);

        img_y.emplace_back(nv12_image_plane_y[i]);
        img_uv.emplace_back(nv12_image_plane_uv[i]);

        tensor_remote_y.emplace_back(cldnn_context.create_tensor(param_input_y->get_element_type(), fake_image_data_y[i].get_shape(), img_y[i]));
        tensor_remote_uv.emplace_back(cldnn_context.create_tensor(param_input_uv->get_element_type(), fake_image_data_uv[i].get_shape(), img_uv[i]));
    }

    for (size_t i = 0; i < 5; ++i) {    // to test repeating set_tensors/infer functionality
        inf_req_remote.set_tensors(*param_input_y->output(0).get_tensor().get_names().begin(), tensor_remote_y);
        inf_req_remote.set_tensors(*param_input_uv->output(0).get_tensor().get_names().begin(), tensor_remote_uv);

        inf_req_remote.infer();
    }

    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));
    ASSERT_NO_THROW(output_tensor_shared.data());

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(GPU_CONFIG_KEY(BUFFER));
    p_reg.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto param_input_y_reg = fn_ptr_regular->get_parameters().at(0);
    auto param_input_uv_reg = fn_ptr_regular->get_parameters().at(1);

    auto exec_net_regular = ie.compile_model(function_regular, CommonTestUtils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();

    for (size_t i = 0; i < num_batch; ++i) {
        inf_req_regular.set_tensor(param_input_y_reg, fake_image_data_y[i]);
        inf_req_regular.set_tensor(param_input_uv_reg, fake_image_data_uv[i]);
        inf_req_regular.infer();
        auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

        ASSERT_EQ(output_tensor_regular.get_size() * num_batch, output_tensor_shared.get_size());
        float thr = 0.1;

        FuncTestUtils::compareRawBuffers<float>(static_cast<float*>(output_tensor_shared.data()) + i * output_tensor_regular.get_size(),
                                                static_cast<float*>(output_tensor_regular.data()),
                                                output_tensor_regular.get_size(), output_tensor_regular.get_size(), thr);
    }
}

TEST_P(OVRemoteTensorBatched_Test, NV12toBGR_buffer) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    std::vector<ov::Tensor> fake_image_data_y, fake_image_data_uv;
    for (size_t i = 0; i < num_batch * 2; ++i) {
        fake_image_data_y.push_back(FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 1, height, width}, 50, 0, 1, i));
        fake_image_data_uv.push_back(FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 2, height / 2, width / 2}, 256, 0, 1, i));
    }

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({num_batch, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(GPU_CONFIG_KEY(BUFFER));
    p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);
    auto param_input_uv = fn_ptr_remote->get_parameters().at(1);
    auto output = function->get_results().at(0);

    auto ocl_instance = std::make_shared<OpenCL>();
    ocl_instance->_queue = cl::CommandQueue(ocl_instance->_context, ocl_instance->_device);
    cl_int err;

    auto in_size_y = ov::shape_size(param_input_y->get_output_shape(0)) * param_input_y->get_output_element_type(0).size();
    auto in_size_uv = ov::shape_size(param_input_uv->get_output_shape(0)) * param_input_uv->get_output_element_type(0).size();
    auto out_size = ov::shape_size(output->get_output_shape(0)) * output->get_output_element_type(0).size();

    auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_queue.get());
    auto exec_net_shared = ie.compile_model(function, remote_context);
    auto gpu_context = exec_net_shared.get_context().as<ov::intel_gpu::ocl::ClContext>();

    std::vector<cl::Buffer> shared_input_y_buffer, shared_input_uv_buffer;
    std::vector<ov::Tensor> gpu_in_y_tensor, gpu_in_uv_tensor;

    for (size_t i = 0; i < num_batch; ++i) {
        shared_input_y_buffer.emplace_back(cl::Buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size_y, NULL, &err));
        shared_input_uv_buffer.emplace_back(cl::Buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size_uv, NULL, &err));

        gpu_in_y_tensor.emplace_back(gpu_context.create_tensor(param_input_y->get_output_element_type(0),
                                                               fake_image_data_y[i].get_shape(),
                                                               shared_input_y_buffer[i]));
        gpu_in_uv_tensor.emplace_back(gpu_context.create_tensor(param_input_uv->get_output_element_type(0),
                                                                fake_image_data_uv[i].get_shape(),
                                                                shared_input_uv_buffer[i]));
    }
    cl::Buffer shared_output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);
    auto gpu_out_tensor = gpu_context.create_tensor(output->get_output_element_type(0), output->get_output_shape(0), shared_output_buffer);
    auto out_tensor = FuncTestUtils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

    auto inf_req_shared = exec_net_shared.create_infer_request();
    inf_req_shared.set_tensors(*param_input_y->output(0).get_tensor().get_names().begin(), gpu_in_y_tensor);
    inf_req_shared.set_tensors(*param_input_uv->output(0).get_tensor().get_names().begin(), gpu_in_uv_tensor);
    inf_req_shared.set_tensor(output, gpu_out_tensor);

    for (size_t i = 0; i < num_batch; ++i) {
        void* buffer_y = fake_image_data_y[i].data();
        void* buffer_uv = fake_image_data_uv[i].data();

        ocl_instance->_queue.enqueueWriteBuffer(shared_input_y_buffer[i], false, 0, in_size_y, buffer_y);
        ocl_instance->_queue.enqueueWriteBuffer(shared_input_uv_buffer[i], false, 0, in_size_uv, buffer_uv);
    }

    inf_req_shared.start_async();
    ocl_instance->_queue.enqueueReadBuffer(shared_output_buffer, false, 0, out_size, out_tensor.data(), nullptr, nullptr);
    ocl_instance->_queue.finish();
    ASSERT_NO_THROW(out_tensor.data());

    // ------------------------------------------------------
    // inference using the same InferRequest but with new data
    inf_req_shared.wait();

    std::vector<cl::Buffer> shared_input_y_buffer_new, shared_input_uv_buffer_new;
    std::vector<ov::Tensor> gpu_in_y_tensor_new, gpu_in_uv_tensor_new;
    for (size_t i = 0; i < num_batch; ++i) {
        shared_input_y_buffer_new.emplace_back(cl::Buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size_y, NULL, &err));
        shared_input_uv_buffer_new.emplace_back(cl::Buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size_uv, NULL, &err));

        gpu_in_y_tensor_new.emplace_back(gpu_context.create_tensor(param_input_y->get_output_element_type(0),
                                                                   fake_image_data_y[i + num_batch].get_shape(),
                                                                   shared_input_y_buffer_new[i]));
        gpu_in_uv_tensor_new.emplace_back(gpu_context.create_tensor(param_input_uv->get_output_element_type(0),
                                                                    fake_image_data_uv[i + num_batch].get_shape(),
                                                                    shared_input_uv_buffer_new[i]));
    }
    cl::Buffer shared_output_buffer_new(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);
    auto gpu_out_tensor_new = gpu_context.create_tensor(output->get_output_element_type(0), output->get_output_shape(0), shared_output_buffer_new);
    auto out_tensor_new = FuncTestUtils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

    inf_req_shared.set_tensors(*param_input_y->output(0).get_tensor().get_names().begin(), gpu_in_y_tensor_new);
    inf_req_shared.set_tensors(*param_input_uv->output(0).get_tensor().get_names().begin(), gpu_in_uv_tensor_new);
    inf_req_shared.set_tensor(output, gpu_out_tensor_new);

    for (size_t i = 0; i < num_batch; ++i) {
        void* buffer_y = fake_image_data_y[i + num_batch].data();
        void* buffer_uv = fake_image_data_uv[i + num_batch].data();

        ocl_instance->_queue.enqueueWriteBuffer(shared_input_y_buffer_new[i], false, 0, in_size_y, buffer_y);
        ocl_instance->_queue.enqueueWriteBuffer(shared_input_uv_buffer_new[i], false, 0, in_size_uv, buffer_uv);
    }
    inf_req_shared.start_async();
    ocl_instance->_queue.enqueueReadBuffer(shared_output_buffer_new, false, 0, out_size, out_tensor_new.data(), nullptr, nullptr);
    ocl_instance->_queue.finish();
    ASSERT_NO_THROW(out_tensor_new.data());

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(GPU_CONFIG_KEY(BUFFER));
    p_reg.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto param_input_y_reg = fn_ptr_regular->get_parameters().at(0);
    auto param_input_uv_reg = fn_ptr_regular->get_parameters().at(1);

    auto exec_net_regular = ie.compile_model(function_regular, CommonTestUtils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();

    for (size_t i = 0; i < num_batch; ++i) {
        inf_req_regular.set_tensor(param_input_y_reg, fake_image_data_y[i + num_batch]);
        inf_req_regular.set_tensor(param_input_uv_reg, fake_image_data_uv[i + num_batch]);
        inf_req_regular.infer();
        auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

        ASSERT_EQ(output_tensor_regular.get_size() * num_batch, out_tensor_new.get_size());
        float thr = 0.1;

        FuncTestUtils::compareRawBuffers<float>(static_cast<float*>(out_tensor_new.data()) + i * output_tensor_regular.get_size(),
                                                static_cast<float*>(output_tensor_regular.data()),
                                                output_tensor_regular.get_size(), output_tensor_regular.get_size(), thr);
    }
}

const std::vector<size_t> num_batches{ 1, 2, 4 };
INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVRemoteTensorBatched_Test, ::testing::ValuesIn(num_batches), OVRemoteTensorBatched_Test::getTestCaseName);
