// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/remote_tensor.hpp"

#include "remote_tensor_tests/helpers.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"
#include "common_test_utils/subgraph_builders/convert_transpose.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

class OVRemoteTensor_Test : public ov::test::TestsCommon {
protected:
    std::shared_ptr<ov::Model> fn_ptr;

    void SetUp() override {
        fn_ptr = ov::test::utils::make_split_multi_conv_concat();
    }
};

namespace {
using ::testing::HasSubstr;

std::vector<bool> ov_dynamic {true, false};
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
}  // namespace

using RemoteTensorSharingTestOptionsParams = std::tuple<RemoteTensorSharingType, bool /*auto-batching*/, bool /*dynamic*/>;

class OVRemoteTensorInputBlob_Test : public OVRemoteTensor_Test,
        public testing::WithParamInterface<RemoteTensorSharingTestOptionsParams> {
protected:
    std::shared_ptr<ov::Model> fn_ptr;
    std::string deviceName;
    ov::AnyMap config;

public:
    void SetUp() override {
        deviceName = ov::test::utils::DEVICE_GPU;
        RemoteTensorSharingType sharing_type;
        bool with_auto_batching;
        bool is_dynamic;
        std::tie(sharing_type, with_auto_batching, is_dynamic) = this->GetParam();
        if (with_auto_batching) {
            config =
                    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                     // immediate timeout to avoid increasing the test time
                     ov::auto_batch_timeout(0)
                    };
        }
        fn_ptr = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
        if (is_dynamic) {
            std::map<size_t, ov::PartialShape> target_shape = {{0, ov::PartialShape::dynamic(4)}};
            fn_ptr->reshape(target_shape);
        }
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RemoteTensorSharingTestOptionsParams>& obj) {
        RemoteTensorSharingType sharing_type;
        bool with_auto_batching;
        bool is_dynamic;
        std::tie(sharing_type, with_auto_batching, is_dynamic) = obj.param;

        std::ostringstream result;
        result << "OVRemoteTensorInputBlob_Test_";
        result << sharing_type;
        if (with_auto_batching)
            result << "_WITH_AUTO_BATCHING";
        if (is_dynamic)
            result << "_DYNAMIC";
        return result.str();
    }
};

TEST_P(OVRemoteTensorInputBlob_Test, smoke_cantCreateBlobWithInvalidSize) {
    RemoteTensorSharingType sharing_type;
    bool with_auto_batching;
    bool is_dynamic;
    std::tie(sharing_type, with_auto_batching, is_dynamic) = GetParam();
    if (with_auto_batching || is_dynamic)
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
    bool is_dynamic;
    std::tie(sharing_type, with_auto_batching, is_dynamic) = GetParam();

    // auto-batching relies on availability of the lock() for the tensor (and the *USM_DEVICE is not lockable)
    if (with_auto_batching
            && (RemoteTensorSharingType::USER_USM_DEVICE_TENSOR == sharing_type
                    || RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR == sharing_type))
        GTEST_SKIP();

    auto exec_net = ie.compile_model(function, deviceName, config);
    ov::Shape input_shape{1, 2, 32, 32};

    // regular inference
    auto inf_req_regular = exec_net.create_infer_request();
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);
    auto fakeImageData = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input_shape);

    inf_req_regular.set_tensor(input, fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(output);

    // inference using remote tensor
    auto inf_req_shared = exec_net.create_infer_request();
    auto cldnn_context = exec_net.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    auto imSize = ov::shape_size(input_shape);

    switch (sharing_type) {
        case RemoteTensorSharingType::USER_CL_TENSOR: {
            cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imSize, NULL, &err);
            {
                void* buffer = fakeImageData.data();
                ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
            }

            auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input_shape, shared_buffer);
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

            auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input_shape, shared_buffer);
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

            auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input_shape, shared_buffer);
            inf_req_shared.set_tensor(input, cldnn_tensor);
            inf_req_shared.infer();

            ocl_instance->free_mem(shared_buffer);

            break;
        }
        case RemoteTensorSharingType::PLUGIN_CL_TENSOR: {
            auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input_shape);
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

            auto cldnn_tensor = cldnn_context.create_usm_host_tensor(input->get_element_type(), input_shape);
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

            auto cldnn_tensor = cldnn_context.create_usm_device_tensor(input->get_element_type(), input_shape);
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
            auto cldnn_tensor = cldnn_context.create_host_tensor(input->get_element_type(), input_shape);
            {
                OV_ASSERT_NO_THROW(cldnn_tensor.data());
                void* shared_buffer = cldnn_tensor.data();
                if (ocl_instance->supports_usm()) {
                    ASSERT_EQ(ocl_instance->get_allocation_type(shared_buffer), CL_MEM_TYPE_HOST_INTEL);
                }
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
        OV_ASSERT_NO_THROW(output_tensor_regular.data());
        OV_ASSERT_NO_THROW(output_tensor_shared.data());
        ov::test::utils::compare(output_tensor_regular, output_tensor_shared);
    }
}

TEST_P(OVRemoteTensorInputBlob_Test, smoke_canInputOutputRemoteTensor) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto ie = ov::Core();

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr);
    p.input().tensor().set_element_type(ov::element::i8);
    p.input().preprocess().convert_element_type(ov::element::f32);

    auto model = p.build();
    RemoteTensorSharingType sharing_type;
    bool with_auto_batching;
    bool is_dynamic;
    std::tie(sharing_type, with_auto_batching, is_dynamic) = GetParam();

    // auto-batching relies on availability of the lock() for the tensor (and the *USM_DEVICE is not lockable)
    if (with_auto_batching)
        GTEST_SKIP();

    auto compiled_model = ie.compile_model(model, deviceName, config);

    ov::Shape input_shape{1, 2, 32, 32};
    ov::Shape output_shape{1, 2, 32, 32};
    // regular inference
    auto inf_req_regular = compiled_model.create_infer_request();
    auto input = model->get_parameters().at(0);
    auto output = model->get_results().at(0);

    auto input_data = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input_shape);

    inf_req_regular.set_tensor(input, input_data);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(output);

    // inference using remote tensor
    auto inf_req_shared = compiled_model.create_infer_request();
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = gpu_context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    auto allocated_out_shape = output_shape;
    if (is_dynamic) {
        // In dynamic case we allocate more than required to check that out tensor is reshaped correctly
        allocated_out_shape[1]++;
    }

    auto in_size = ov::shape_size(input_shape);
    auto out_size = ov::shape_size(output_shape) * output->get_output_element_type(0).bitwidth() / 8;
    auto allocated_out_size = ov::shape_size(allocated_out_shape) * output->get_output_element_type(0).bitwidth() / 8;
    auto output_tensor_shared = ov::test::utils::create_and_fill_tensor(output->get_output_element_type(0), output_shape);

    switch (sharing_type) {
        case RemoteTensorSharingType::USER_CL_TENSOR: {
            cl::Buffer shared_input_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size, NULL, &err);
            cl::Buffer shared_output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, allocated_out_size, NULL, &err);
            {
                void* buffer = input_data.data();
                ocl_instance->_queue.enqueueWriteBuffer(shared_input_buffer, true, 0, in_size, buffer);
            }

            auto input_remote_tensor = gpu_context.create_tensor(input->get_element_type(), input_shape, shared_input_buffer);
            auto output_remote_tensor = gpu_context.create_tensor(output->get_output_element_type(0), allocated_out_shape, shared_output_buffer);
            inf_req_shared.set_tensor(input, input_remote_tensor);
            inf_req_shared.set_tensor(output, output_remote_tensor);
            inf_req_shared.infer();

            {
                void* buffer = output_tensor_shared.data();
                auto out_tensor = inf_req_shared.get_output_tensor();
                ASSERT_EQ(out_tensor.get_shape(), output_shape);
                ocl_instance->_queue.enqueueReadBuffer(shared_output_buffer, true, 0, out_size, buffer);
            }

            break;
        }
        case RemoteTensorSharingType::USER_USM_DEVICE_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            void* shared_input_buffer = ocl_instance->allocate_usm_device_buffer(in_size);
            void* shared_output_buffer = ocl_instance->allocate_usm_device_buffer(allocated_out_size);
            {
                void* buffer = input_data.data();
                err = ocl_instance->memcpy(ocl_instance->_queue, shared_input_buffer, buffer, in_size, true, nullptr, nullptr);
                if (err != CL_SUCCESS)
                    FAIL() << "Failed to copy data from host buffer to USM device";
            }

            auto input_remote_tensor = gpu_context.create_tensor(input->get_element_type(), input_shape, shared_input_buffer);
            auto output_remote_tensor = gpu_context.create_tensor(output->get_output_element_type(0), allocated_out_shape, shared_output_buffer);
            inf_req_shared.set_tensor(input, input_remote_tensor);
            inf_req_shared.set_tensor(output, output_remote_tensor);
            inf_req_shared.infer();

            {
                void* buffer = output_tensor_shared.data();
                auto out_tensor = inf_req_shared.get_output_tensor();
                ASSERT_EQ(out_tensor.get_shape(), output_shape);
                err = ocl_instance->memcpy(ocl_instance->_queue, buffer, shared_output_buffer, out_size, true, nullptr, nullptr);
                if (err != CL_SUCCESS)
                    FAIL() << "Failed to copy data from USM device to host buffer";
            }


            ocl_instance->free_mem(shared_input_buffer);
            ocl_instance->free_mem(shared_output_buffer);

            break;
        }
        case RemoteTensorSharingType::USER_USM_HOST_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            void* shared_input_buffer = ocl_instance->allocate_usm_host_buffer(in_size);
            void* shared_output_buffer = ocl_instance->allocate_usm_host_buffer(allocated_out_size);
            {
                void* buffer = input_data.data();
                std::memcpy(shared_input_buffer, buffer, in_size);
            }

            auto input_remote_tensor = gpu_context.create_tensor(input->get_element_type(), input_shape, shared_input_buffer);
            auto output_remote_tensor = gpu_context.create_tensor(output->get_output_element_type(0), allocated_out_shape, shared_output_buffer);
            inf_req_shared.set_tensor(input, input_remote_tensor);
            inf_req_shared.set_tensor(output, output_remote_tensor);
            inf_req_shared.infer();

            {
                void* buffer = output_tensor_shared.data();
                auto out_tensor = inf_req_shared.get_output_tensor();
                ASSERT_EQ(out_tensor.get_shape(), output_shape);
                err = ocl_instance->memcpy(ocl_instance->_queue, buffer, shared_output_buffer, out_size, true, nullptr, nullptr);
                if (err != CL_SUCCESS)
                    FAIL() << "Failed to copy data from USM host to host buffer";
            }

            ocl_instance->free_mem(shared_input_buffer);
            ocl_instance->free_mem(shared_output_buffer);

            break;
        }
        case RemoteTensorSharingType::PLUGIN_CL_TENSOR: {
            auto input_remote_tensor = gpu_context.create_tensor(input->get_element_type(), input_shape);
            auto output_remote_tensor = gpu_context.create_tensor(output->get_output_element_type(0), allocated_out_shape);
            ASSERT_TRUE(input_remote_tensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
            auto cl_tensor = input_remote_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();
            {
                cl::Buffer shared_buffer = cl_tensor;
                void* buffer = input_data.data();
                ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, in_size, buffer);
            }
            inf_req_shared.set_tensor(input, input_remote_tensor);
            inf_req_shared.set_tensor(output, output_remote_tensor);
            inf_req_shared.infer();

            {
                auto out_cl_tensor = output_remote_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();

                void* buffer = output_tensor_shared.data();
                auto out_tensor = inf_req_shared.get_output_tensor();
                ASSERT_EQ(out_tensor.get_shape(), output_shape);
                ocl_instance->_queue.enqueueReadBuffer(out_cl_tensor, true, 0, out_size, buffer);
            }

            break;
        }
        case RemoteTensorSharingType::PLUGIN_USM_HOST_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            auto input_remote_tensor = gpu_context.create_usm_host_tensor(input->get_element_type(), input_shape);
            auto output_remote_tensor = gpu_context.create_usm_host_tensor(output->get_output_element_type(0), allocated_out_shape);
            ASSERT_TRUE(input_remote_tensor.is<ov::intel_gpu::ocl::USMTensor>());
            {
                auto cl_tensor = input_remote_tensor.as<ov::intel_gpu::ocl::USMTensor>();
                void* shared_buffer = cl_tensor.get();
                ASSERT_EQ(ocl_instance->get_allocation_type(shared_buffer), CL_MEM_TYPE_HOST_INTEL);
                void* buffer = input_data.data();
                std::memcpy(shared_buffer, buffer, in_size);
            }

            inf_req_shared.set_tensor(input, input_remote_tensor);
            inf_req_shared.set_tensor(output, output_remote_tensor);
            inf_req_shared.infer();

            {
                void* buffer = output_tensor_shared.data();
                auto out_tensor = inf_req_shared.get_output_tensor();
                auto cl_tensor = out_tensor.as<ov::intel_gpu::ocl::USMTensor>();
                void* shared_output_buffer = cl_tensor.get();
                ASSERT_EQ(ocl_instance->get_allocation_type(shared_output_buffer), CL_MEM_TYPE_HOST_INTEL);
                ASSERT_EQ(out_tensor.get_shape(), output_shape);
                std::memcpy(buffer, shared_output_buffer, out_size);
            }

            break;
        }
        case RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR: {
            if (!ocl_instance->supports_usm())
                GTEST_SKIP();

            auto input_remote_tensor = gpu_context.create_usm_device_tensor(input->get_element_type(), input_shape);
            auto output_remote_tensor = gpu_context.create_usm_device_tensor(output->get_output_element_type(0), allocated_out_shape);
            ASSERT_TRUE(input_remote_tensor.is<ov::intel_gpu::ocl::USMTensor>());
            {
                auto cl_tensor = input_remote_tensor.as<ov::intel_gpu::ocl::USMTensor>();
                void* shared_buffer = cl_tensor.get();
                ASSERT_EQ(ocl_instance->get_allocation_type(shared_buffer), CL_MEM_TYPE_DEVICE_INTEL);
                void* buffer = input_data.data();
                err = ocl_instance->memcpy(ocl_instance->_queue, shared_buffer, buffer, in_size, true, nullptr, nullptr);
                if (err != CL_SUCCESS)
                    FAIL() << "Failed to copy data from host buffer to USM device";
            }

            inf_req_shared.set_tensor(input, input_remote_tensor);
            inf_req_shared.set_tensor(output, output_remote_tensor);
            inf_req_shared.infer();

            {
                auto cl_tensor = output_remote_tensor.as<ov::intel_gpu::ocl::USMTensor>();
                void* shared_output_buffer = cl_tensor.get();

                void* buffer = output_tensor_shared.data();
                auto out_tensor = inf_req_shared.get_output_tensor();
                ASSERT_EQ(out_tensor.get_shape(), output_shape);
                err = ocl_instance->memcpy(ocl_instance->_queue, buffer, shared_output_buffer, out_size, true, nullptr, nullptr);
            }

            break;
        }
        case RemoteTensorSharingType::PLUGIN_HOST_TENSOR: {
            auto input_tensor = gpu_context.create_host_tensor(input->get_element_type(), input_shape);
            auto output_tensor = gpu_context.create_host_tensor(output->get_output_element_type(0), allocated_out_shape);
            {
                OV_ASSERT_NO_THROW(input_tensor.data());
                void* shared_buffer = input_tensor.data();
                if (ocl_instance->supports_usm()) {
                    ASSERT_EQ(ocl_instance->get_allocation_type(shared_buffer), CL_MEM_TYPE_HOST_INTEL);
                }
                void* buffer = input_data.data();
                std::memcpy(shared_buffer, buffer, in_size);
            }

            inf_req_shared.set_tensor(input, input_tensor);
            inf_req_shared.set_tensor(output, output_tensor);
            inf_req_shared.infer();

            {
                void* buffer = output_tensor_shared.data();
                auto out_tensor = inf_req_shared.get_output_tensor();
                ASSERT_EQ(out_tensor.get_shape(), output_shape);
                err = ocl_instance->memcpy(ocl_instance->_queue, buffer, output_tensor.data(), out_size, true, nullptr, nullptr);
            }
            break;
        }
    }

    // compare results
    {
        ASSERT_EQ(output->get_element_type(), ov::element::f32);
        ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
        OV_ASSERT_NO_THROW(output_tensor_regular.data());
        OV_ASSERT_NO_THROW(output_tensor_shared.data());
        ov::test::utils::compare(output_tensor_regular, output_tensor_shared);
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
        ::testing::ValuesIn(ov_with_auto_batching),
        ::testing::ValuesIn(ov_dynamic)),
        OVRemoteTensorInputBlob_Test::getTestCaseName);

TEST(OVRemoteTensorTests, smoke_MixedTensorTypes) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();
    auto model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
    std::map<size_t, ov::PartialShape> dynamic_shape = {{0, ov::PartialShape::dynamic(4)}};
    model->reshape(dynamic_shape);

    auto dynamic_compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);

    auto input = model->get_parameters().at(0);
    auto output = model->get_results().at(0);

    auto gpu_context = dynamic_compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = gpu_context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);

    ov::Shape output_shape_allocated{1, 3, 32, 32};
    auto user_output_tensor = gpu_context.create_tensor(output->get_element_type(), output_shape_allocated);
    ov::Tensor output_tensor_copy_0(output->get_element_type(), output_shape_allocated);
    ov::Tensor output_tensor_copy_1(output->get_element_type(), output_shape_allocated);

    {
        auto infer_request = dynamic_compiled_model.create_infer_request();
        {
            // Run infer request with user's input & output tensor
            // Output tensor size is larger than required
            ov::Shape input_shape{1, 2, 32, 32};
            auto input_tensor = gpu_context.create_tensor(input->get_element_type(), input_shape);
            ov::Shape output_shape_actual{1, 2, 32, 32};

            infer_request.set_tensor(input, input_tensor);
            infer_request.set_tensor(output, user_output_tensor);
            infer_request.infer();
            auto output_tensor = infer_request.get_tensor(output);

            ASSERT_TRUE(output_tensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
            ASSERT_TRUE(user_output_tensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
            auto t1 = output_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();
            auto t2 = user_output_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();

            ASSERT_EQ(t1.get(), t2.get());
            ASSERT_EQ(output_tensor.get_shape(), output_shape_actual);
        }

        {
            // Keep same output, but use larger input
            // In that case user tensor is not enough to store the result and set shape will be called on the user
            // tensor
            ov::Shape input_shape{1, 4, 32, 32};
            ov::Shape output_shape_actual{1, 4, 32, 32};
            auto input_tensor = gpu_context.create_tensor(input->get_element_type(), input_shape);

            infer_request.set_tensor(input, input_tensor);
            OV_ASSERT_NO_THROW(infer_request.infer());

            ASSERT_EQ(infer_request.get_output_tensor().get_shape(), output_shape_actual);
        }

        {
            // Now try to increase buffer size comparing to the 1st run
            // User output buffer is supposed to be the same
            ov::Shape input_shape{1, 3, 32, 32};
            ov::Shape output_shape_actual{1, 3, 32, 32};
            auto input_tensor_1 = gpu_context.create_tensor(input->get_element_type(), input_shape);
            auto data = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input_shape);
            ASSERT_TRUE(input_tensor_1.is<ov::intel_gpu::ocl::ClBufferTensor>());
            auto cl_tensor = input_tensor_1.as<ov::intel_gpu::ocl::ClBufferTensor>();
            cl::Buffer shared_buffer = cl_tensor;
            void* buffer = data.data();
            ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, ov::shape_size(input_shape) * data.get_element_type().size(), buffer);

            infer_request.set_tensor(input, input_tensor_1);
            infer_request.infer();
            auto output_tensor = infer_request.get_tensor(output);
            ASSERT_TRUE(output_tensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
            ASSERT_TRUE(user_output_tensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
            auto t1 = output_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();
            auto t2 = user_output_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();

            // inference result of this iteration is stored to output_tensor_copy_0 for further values check
            ocl_instance->_queue.enqueueReadBuffer(t2, true, 0, user_output_tensor.get_byte_size(), output_tensor_copy_0.data());
            ASSERT_EQ(t1.get(), t2.get());
            ASSERT_EQ(output_tensor.get_shape(), output_shape_actual);
        }
    }

    {
        auto infer_request = dynamic_compiled_model.create_infer_request();
        ov::Shape input_shape_0{1, 2, 32, 32};
        ov::Shape output_shape_actual_0{1, 2, 32, 32};
        auto input_tensor_0 = gpu_context.create_tensor(input->get_element_type(), input_shape_0);
        auto data = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input_shape_0);
        ASSERT_TRUE(input_tensor_0.is<ov::intel_gpu::ocl::ClBufferTensor>());
        auto cl_tensor = input_tensor_0.as<ov::intel_gpu::ocl::ClBufferTensor>();
        cl::Buffer shared_buffer = cl_tensor;
        void* buffer = data.data();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, ov::shape_size(input_shape_0) * data.get_element_type().size(), buffer);

        infer_request.set_tensor(input, input_tensor_0);
        infer_request.infer();

        auto output_tensor = infer_request.get_tensor(output);

        ASSERT_FALSE(output_tensor.is<ov::RemoteTensor>());
        ASSERT_EQ(output_tensor.get_shape(), output_shape_actual_0);
    }

    // Finally, check that last result stored in user output tensor is not corrupted when we run after one more iteration with another output buffer
    ASSERT_TRUE(user_output_tensor.is<ov::intel_gpu::ocl::ClBufferTensor>());
    auto t2 = user_output_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();
    ocl_instance->_queue.enqueueReadBuffer(t2, true, 0, user_output_tensor.get_byte_size(), output_tensor_copy_1.data());

    for (size_t i = 0; i < output_tensor_copy_0.get_size(); i++) {
        ASSERT_EQ(output_tensor_copy_0.data<float>()[i], output_tensor_copy_1.data<float>()[i]) << " i = " << i;
    }
}

class OVRemoteTensor_TestsWithContext : public OVRemoteTensor_Test, public testing::WithParamInterface<bool> {
protected:
    std::shared_ptr<ov::Model> fn_ptr;
    std::string deviceName;
    ov::AnyMap config;

public:
    void SetUp() override {
        fn_ptr = ov::test::utils::make_split_multi_conv_concat();
        deviceName = ov::test::utils::DEVICE_GPU;
        auto with_auto_batching = this->GetParam();
        if (with_auto_batching) {
            config =
                    {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
                            // immediate timeout to avoid increasing the test time
                     ov::auto_batch_timeout(0)
                    };
        }
        fn_ptr = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();
    }
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        auto with_auto_batch = obj.param;
        return std::string("RemoteTensor_Test") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
    }

    void run_smoke_canInferOnUserContext(bool is_caching_test) {
        auto ie = ov::Core();

        std::string cacheDirName;
        if (is_caching_test) {
            auto with_auto_batch = this->GetParam();
            cacheDirName = std::string("smoke_canInferOnUserContext") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
            ie.set_property(ov::cache_dir(cacheDirName));
        }

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
        auto fakeImageData = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
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
            OV_ASSERT_NO_THROW(output_tensor_regular.data());
            OV_ASSERT_NO_THROW(output_tensor_shared.data());
            ov::test::utils::compare(output_tensor_regular, output_tensor_shared);
        }

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
        }
    }

    void run_smoke_canInferOnUserContextWithMultipleDevices(bool is_caching_test) {
        auto ie = ov::Core();

        std::string cacheDirName;
        if (is_caching_test) {
            auto with_auto_batch = this->GetParam();
            cacheDirName = std::string("smoke_canInferOnUserContextWithMultipleDevices") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
            ie.set_property(ov::cache_dir(cacheDirName));
        }

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
        auto fakeImageData = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
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
            OV_ASSERT_NO_THROW(output_tensor_regular.data());
            OV_ASSERT_NO_THROW(output_tensor_shared.data());
            ov::test::utils::compare(output_tensor_regular, output_tensor_shared);
        }

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
        }
    }

    void run_smoke_canInferOnUserQueue_out_of_order(bool is_caching_test) {
        auto ie = ov::Core();

        std::string cacheDirName;
        if (is_caching_test) {
            auto with_auto_batch = this->GetParam();
            cacheDirName = std::string("smoke_canInferOnUserQueue_out_of_order") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
            ie.set_property(ov::cache_dir(cacheDirName));
        }

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
        auto fakeImageData = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
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
        auto out_tensor = ov::test::utils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

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
            OV_ASSERT_NO_THROW(output_tensor_regular.data());
            ov::test::utils::compare(output_tensor_regular, out_tensor);
        }

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
        }
    }

    void run_smoke_canInferOnUserQueue_in_order(bool is_caching_test) {
        auto ie = ov::Core();

        std::string cacheDirName;
        if (is_caching_test) {
            auto with_auto_batch = this->GetParam();
            cacheDirName = std::string("smoke_canInferOnUserQueue_in_order") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
            ie.set_property(ov::cache_dir(cacheDirName));
        }

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
        auto fakeImageData = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
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
        auto out_tensor = ov::test::utils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

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
            OV_ASSERT_NO_THROW(output_tensor_regular.data());
            ov::test::utils::compare(output_tensor_regular, out_tensor);
        }

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
        }
    }

    void run_smoke_canInferOnUserQueue_infer_call_many_times(bool is_caching_test) {
        auto ie = ov::Core();

        std::string cacheDirName;
        if (is_caching_test) {
            auto with_auto_batch = this->GetParam();
            cacheDirName = std::string("smoke_canInferOnUserQueue_infer_call_many_times") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
            ie.set_property(ov::cache_dir(cacheDirName));
        }

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
        auto fakeImageData = ov::test::utils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
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
        auto out_tensor = ov::test::utils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

        auto inf_req_shared = exec_net_shared.create_infer_request();
        inf_req_shared.set_tensor(input, gpu_in_tensor);
        inf_req_shared.set_tensor(output, gpu_out_tensor);

        // 1. Pre-processing. Enqueue non-blocking copy from host ptr to shared device input buffer
        {
            void* buffer = fakeImageData.data();
            ocl_instance->_queue.enqueueWriteBuffer(shared_input_buffer, false, 0, in_size, buffer);
        }

        // 2. Enqueue inference primitives. Synchronous infer() call waits for completion of the result, thus results of the first iterations are discarded
        for (size_t i = 0; i < 10; i++) {
            inf_req_shared.infer();
        }

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
            OV_ASSERT_NO_THROW(output_tensor_regular.data());
            ov::test::utils::compare(output_tensor_regular, out_tensor);
        }

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
        }
    }

    void run_smoke_canCreateManyTensorsOnSameMem(bool is_caching_test) {
        bool with_auto_batching = GetParam();
        if (with_auto_batching)
            GTEST_SKIP();

        auto ocl_instance = std::make_shared<OpenCL>();
        if (!ocl_instance->supports_usm())
            GTEST_SKIP();

        auto input = fn_ptr->get_parameters().at(0);
        ov::Shape input_shape = input->get_shape();
        auto imSize = ov::shape_size(input_shape);
        void* usm_ptr = ocl_instance->allocate_usm_host_buffer(imSize*sizeof(float));

        auto ie = ov::Core();
        auto remote_context = ov::intel_gpu::ocl::ClContext(ie, ocl_instance->_context.get());

        std::string cacheDirName;
        if (is_caching_test) {
            auto with_auto_batch = this->GetParam();
            cacheDirName = std::string("smoke_canCreateManyTensorsOnSameMem") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
            ie.set_property(ov::cache_dir(cacheDirName));

            auto tmp_model = ie.compile_model(fn_ptr, remote_context, config);
        }

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

        if (is_caching_test) {
            ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
            ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
            ov::test::utils::removeDir(cacheDirName);
        }
    }
};

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserContext) {
    this->run_smoke_canInferOnUserContext(false);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserContext_cached) {
    this->run_smoke_canInferOnUserContext(true);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserContextWithMultipleDevices) {
    this->run_smoke_canInferOnUserContextWithMultipleDevices(false);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserQueue_out_of_order) {
    this->run_smoke_canInferOnUserQueue_out_of_order(false);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserQueue_out_of_order_cached) {
    this->run_smoke_canInferOnUserQueue_out_of_order(true);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserQueue_in_order) {
    this->run_smoke_canInferOnUserQueue_in_order(false);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserQueue_in_order_cached) {
    this->run_smoke_canInferOnUserQueue_in_order(true);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserQueue_infer_call_many_times) {
    this->run_smoke_canInferOnUserQueue_infer_call_many_times(false);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canInferOnUserQueue_infer_call_many_times_cached) {
    this->run_smoke_canInferOnUserQueue_infer_call_many_times(true);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canCreateManyTensorsOnSameMem) {
    this->run_smoke_canCreateManyTensorsOnSameMem(false);
}

TEST_P(OVRemoteTensor_TestsWithContext, smoke_canCreateManyTensorsOnSameMem_cached) {
    this->run_smoke_canCreateManyTensorsOnSameMem(true);
}

INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVRemoteTensor_TestsWithContext, ::testing::ValuesIn(ov_with_auto_batching),
                         OVRemoteTensor_TestsWithContext::getTestCaseName);


TEST_F(OVRemoteTensor_Test, NV12toGray) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 8;
    const int width = 8;
    const int feature = 1;

    // ------------------------------------------------------
    // Prepare input data
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 50;
    ov::Tensor fake_image = ov::test::utils::create_and_fill_tensor(ov::element::i8, {1, height, width, feature}, in_data);
    ov::Tensor fake_image_regular = ov::test::utils::create_and_fill_tensor(ov::element::f32, {1, height, width, feature });

    auto image_ptr = static_cast<uint8_t*>(fake_image.data());
    auto image_ptr_regular = static_cast<float*>(fake_image_regular.data());
    // Apply NV12 (Surface) -> Gray conversion for regular blob
    for (size_t i = 0; i < fake_image.get_size(); i++) {
        auto val = static_cast<float>(image_ptr[i]) / 255;
        val *= 296.82f;
        val += -18.624f;
        val = val < 0 ? 0 : val > 255 ? 255 : val;
        image_ptr_regular[i] = val;
    }

    auto core = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ov::test::utils::make_conv_pool_relu({1, feature, height, width});

    using namespace ov::preprocess;

    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_layout("NHWC")
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);

    auto exec_net = core.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req_remote = exec_net.create_infer_request();

    auto cldnn_context = exec_net.get_context().as<ov::intel_gpu::ocl::ClContext>();
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

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)height, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y,
        true, origin, y_region, 0, 0, fake_image.data(), 0, NULL, NULL);
    ASSERT_EQ(err, 0);

    cl::Image2D img_y = cl::Image2D(nv12_image_plane_y);

    auto tensor_remote_y = cldnn_context.create_tensor(param_input_y->get_element_type(), fake_image.get_shape(), img_y);
    inf_req_remote.set_tensor(*param_input_y->output(0).get_tensor().get_names().begin(), tensor_remote_y);

    inf_req_remote.infer();
    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ov::test::utils::make_conv_pool_relu({1, feature, height, width});

    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::f32)
                          .set_layout("NHWC")
                          .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto param_input_y_regular = function_regular->get_parameters().at(0);

    auto exec_net_regular = core.compile_model(function_regular, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();
    inf_req_regular.set_tensor(param_input_y_regular, fake_image_regular);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // ------------------------------------------------------
    // compare results
    ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
    OV_ASSERT_NO_THROW(output_tensor_regular.data());
    OV_ASSERT_NO_THROW(output_tensor_shared.data());
    float thr = 0.1f;
    ov::test::utils::compare(output_tensor_shared, output_tensor_regular, thr);
}

TEST_F(OVRemoteTensor_Test, NV12toBGR_image_ConvertTranspose) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 8;
    const int width = 8;

    // ------------------------------------------------------
    // Prepare input data
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 50;
    ov::Tensor fake_image_data_y = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height, width, 1}, in_data);
    in_data.range = 256;
    ov::Tensor fake_image_data_uv = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, in_data);

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ov::test::utils::make_convert_transpose({1, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);
    auto param_input_uv = fn_ptr_remote->get_parameters().at(1);

    auto exec_net_b = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
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

    auto nv12 = cldnn_context.create_tensor_nv12(img_y, img_uv);

    auto tensor_remote_y = nv12.first;
    auto tensor_remote_uv = nv12.second;

    inf_req_remote.set_tensor(*param_input_y->output(0).get_tensor().get_names().begin(), tensor_remote_y);
    inf_req_remote.set_tensor(*param_input_uv->output(0).get_tensor().get_names().begin(), tensor_remote_uv);

    inf_req_remote.infer();

    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ov::test::utils::make_convert_transpose({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p_reg.input().preprocess().convert_color(ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto exec_net_regular = ie.compile_model(function_regular, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();
    inf_req_regular.set_tensor(param_input_y, fake_image_data_y);
    inf_req_regular.set_tensor(param_input_uv, fake_image_data_uv);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // ------------------------------------------------------
    // compare results
    ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
    OV_ASSERT_NO_THROW(output_tensor_regular.data());
    OV_ASSERT_NO_THROW(output_tensor_shared.data());
    float thr = 0.1f;
    ov::test::utils::compare(output_tensor_shared, output_tensor_regular, thr);
}

TEST_F(OVRemoteTensor_Test, NV12toBGR_image_single_plane) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 50;
    ov::Tensor fake_image_data_yuv = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height * 3 / 2, width, 1}, in_data);

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ov::test::utils::make_conv_pool_relu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_yuv = fn_ptr_remote->get_parameters().at(0);

    auto exec_net_b = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
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
    image_desc.image_height = height * 3 / 2;
    cl_mem nv12_image_plane_yuv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    ASSERT_EQ(err, 0);

    size_t origin[3] = { 0, 0, 0 };
    size_t yuv_region[3] = { (size_t)width, (size_t)height * 3 / 2, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_yuv,
        true, origin, yuv_region, 0, 0, fake_image_data_yuv.data(), 0, NULL, NULL);
    ASSERT_EQ(err, 0);

    cl::Image2D img_yuv = cl::Image2D(nv12_image_plane_yuv);
    auto tensor_remote_yuv = cldnn_context.create_tensor(param_input_yuv->get_element_type(), fake_image_data_yuv.get_shape(), img_yuv);

    inf_req_remote.set_tensor(*param_input_yuv->output(0).get_tensor().get_names().begin(), tensor_remote_yuv);
    inf_req_remote.infer();

    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ov::test::utils::make_conv_pool_relu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
                          .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p_reg.input().preprocess().convert_color(ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto exec_net_regular = ie.compile_model(function_regular, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();
    inf_req_regular.set_tensor(param_input_yuv, fake_image_data_yuv);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // ------------------------------------------------------
    // compare results
    ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
    OV_ASSERT_NO_THROW(output_tensor_regular.data());
    OV_ASSERT_NO_THROW(output_tensor_shared.data());
    float thr = 0.1f;
    ov::test::utils::compare(output_tensor_shared, output_tensor_regular, thr);
}

TEST_F(OVRemoteTensor_Test, NV12toBGR_image_two_planes) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 50;
    ov::Tensor fake_image_data_y = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height, width, 1}, in_data);
    in_data.range = 256;
    ov::Tensor fake_image_data_uv = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, in_data);

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ov::test::utils::make_conv_pool_relu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);
    auto param_input_uv = fn_ptr_remote->get_parameters().at(1);

    auto exec_net_b = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
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
    auto fn_ptr_regular = ov::test::utils::make_conv_pool_relu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p_reg.input().preprocess().convert_color(ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto exec_net_regular = ie.compile_model(function_regular, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();
    inf_req_regular.set_tensor(param_input_y, fake_image_data_y);
    inf_req_regular.set_tensor(param_input_uv, fake_image_data_uv);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // ------------------------------------------------------
    // compare results
    ASSERT_EQ(output_tensor_regular.get_size(), output_tensor_shared.get_size());
    OV_ASSERT_NO_THROW(output_tensor_regular.data());
    OV_ASSERT_NO_THROW(output_tensor_shared.data());
    float thr = 0.1f;
    ov::test::utils::compare(output_tensor_shared, output_tensor_regular, thr);
}

TEST_F(OVRemoteTensor_Test, NV12toBGR_buffer) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 50;
    ov::Tensor fake_image_data_y = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height, width, 1}, in_data);
    in_data.range = 256;
    ov::Tensor fake_image_data_uv = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, in_data);

    auto ie = ov::Core();

    auto fn_ptr_remote = ov::test::utils::make_conv_pool_relu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p.input().preprocess().convert_color(ColorFormat::BGR);
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
    auto out_tensor = ov::test::utils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

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
    auto exec_net_regular = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();
    inf_req_regular.set_tensor(param_input_y, fake_image_data_y);
    inf_req_regular.set_tensor(param_input_uv, fake_image_data_uv);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

    // ------------------------------------------------------
    // compare results
    ASSERT_EQ(output_tensor_regular.get_size(), out_tensor.get_size());
    OV_ASSERT_NO_THROW(output_tensor_regular.data());
    OV_ASSERT_NO_THROW(out_tensor.data());
    float thr = 0.1f;
    ov::test::utils::compare(out_tensor, output_tensor_regular, thr);
}

class OVRemoteTensorBatched_Test : public ov::test::TestsCommon, public testing::WithParamInterface<size_t> {
    void SetUp() override {
        num_batch = this->GetParam();
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::size_t> &obj) {
        return "num_batch_" + std::to_string(obj.param);
    }

protected:
    size_t num_batch;
    std::vector<std::shared_ptr<ov::Model>> fn_ptrs;
};

TEST_P(OVRemoteTensorBatched_Test, NV12toBGR_image_single_plane) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    std::vector<ov::Tensor> fake_image_data_yuv;
    for (size_t i = 0; i < num_batch; i++) {
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 50;
        in_data.resolution = 1;
        in_data.seed = static_cast<int>(i);
        fake_image_data_yuv.push_back(ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height * 3 / 2, width, 1}, in_data));
    }

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ov::test::utils::make_conv_pool_relu({num_batch, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_yuv = fn_ptr_remote->get_parameters().at(0);

    auto exec_net_b = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req_remote = exec_net_b.create_infer_request();

    auto cldnn_context = exec_net_b.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context.get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    std::vector<cl_mem> nv12_image_plane_yuv;
    std::vector<cl::Image2D> img_yuv;
    std::vector<ov::Tensor> tensor_remote_yuv;

    for (size_t i = 0; i < num_batch; ++i) {
        cl_image_format image_format;
        cl_image_desc image_desc = { 0 };
        image_format.image_channel_order = CL_R;
        image_format.image_channel_data_type = CL_UNORM_INT8;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        image_desc.image_width = width;
        image_desc.image_height = height * 3 / 2;
        nv12_image_plane_yuv.emplace_back(clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err));
        ASSERT_EQ(err, 0);

        size_t origin[3] = { 0, 0, 0 };
        size_t yuv_region[3] = { (size_t)width, (size_t)height * 3 / 2, 1 };

        err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_yuv[i],
            true, origin, yuv_region, 0, 0, fake_image_data_yuv[i].data(), 0, NULL, NULL);
        ASSERT_EQ(err, 0);

        img_yuv.emplace_back(nv12_image_plane_yuv[i]);

        tensor_remote_yuv.emplace_back(cldnn_context.create_tensor(param_input_yuv->get_element_type(), fake_image_data_yuv[i].get_shape(), img_yuv[i]));
    }

    for (size_t i = 0; i < 5; ++i) {    // to test repeating set_tensors/infer functionality
        inf_req_remote.set_tensors(*param_input_yuv->output(0).get_tensor().get_names().begin(), tensor_remote_yuv);
        inf_req_remote.infer();
    }

    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));
    OV_ASSERT_NO_THROW(output_tensor_shared.data());

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ov::test::utils::make_conv_pool_relu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ColorFormat::NV12_SINGLE_PLANE)
                          .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p_reg.input().preprocess().convert_color(ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto param_input_yuv_reg = fn_ptr_regular->get_parameters().at(0);

    auto exec_net_regular = ie.compile_model(function_regular, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();

    for (size_t i = 0; i < num_batch; ++i) {
        inf_req_regular.set_tensor(param_input_yuv_reg, fake_image_data_yuv[i]);
        inf_req_regular.infer();
        auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

        ASSERT_EQ(output_tensor_regular.get_size() * num_batch, output_tensor_shared.get_size());
        float thr = 0.1f;

        ov::test::utils::compare_raw_data(static_cast<float*>(output_tensor_shared.data()) + i * output_tensor_regular.get_size(),
                                                static_cast<float*>(output_tensor_regular.data()),
                                                output_tensor_regular.get_size(), thr);
    }
}

TEST_P(OVRemoteTensorBatched_Test, NV12toBGR_image_two_planes) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    std::vector<ov::Tensor> fake_image_data_y, fake_image_data_uv;
    for (size_t i = 0; i < num_batch; i++) {
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 50;
        in_data.resolution = 1;
        in_data.seed = static_cast<int>(i);
        fake_image_data_y.push_back(ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height, width, 1}, in_data));
        in_data.range = 256;
        fake_image_data_uv.push_back(ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, in_data));
    }

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ov::test::utils::make_conv_pool_relu({num_batch, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().preprocess().convert_color(ColorFormat::BGR);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);
    auto param_input_uv = fn_ptr_remote->get_parameters().at(1);

    auto exec_net_b = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
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
    OV_ASSERT_NO_THROW(output_tensor_shared.data());

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ov::test::utils::make_conv_pool_relu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p_reg.input().preprocess().convert_color(ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto param_input_y_reg = fn_ptr_regular->get_parameters().at(0);
    auto param_input_uv_reg = fn_ptr_regular->get_parameters().at(1);

    auto exec_net_regular = ie.compile_model(function_regular, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();

    for (size_t i = 0; i < num_batch; ++i) {
        inf_req_regular.set_tensor(param_input_y_reg, fake_image_data_y[i]);
        inf_req_regular.set_tensor(param_input_uv_reg, fake_image_data_uv[i]);
        inf_req_regular.infer();
        auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

        ASSERT_EQ(output_tensor_regular.get_size() * num_batch, output_tensor_shared.get_size());
        float thr = 0.1f;

        ov::test::utils::compare_raw_data(static_cast<float*>(output_tensor_shared.data()) + i * output_tensor_regular.get_size(),
                                                static_cast<float*>(output_tensor_regular.data()),
                                                output_tensor_regular.get_size(), thr);
    }
}

TEST_P(OVRemoteTensorBatched_Test, NV12toGray) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 8;
    const int width = 16;
    const int feature = 1;

    // ------------------------------------------------------
    // Prepare input data
    std::vector<ov::Tensor> fake_image;
    std::vector<ov::Tensor> fake_image_regular;
    for (size_t i = 0; i < num_batch; i++) {
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 50;
        in_data.resolution = 1;
        in_data.seed = static_cast<int>(i);
        auto tensor_image = ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height, width, feature}, in_data);
        auto tensor_regular = ov::test::utils::create_and_fill_tensor(ov::element::f32, {1, feature, height, width });
        auto image_ptr = static_cast<uint8_t*>(tensor_image.data());
        auto image_ptr_regular = static_cast<float*>(tensor_regular.data());
        // Apply NV12 (Surface) -> Gray conversion for regular blob
        for (size_t i = 0; i < tensor_image.get_size(); i++) {
            auto val = static_cast<float>(image_ptr[i]) / 255;
            val *= 296.82f;
            val += -18.624f;
            val = val < 0 ? 0 : val > 255 ? 255 : val;
            image_ptr_regular[i] = val;
        }
        fake_image.push_back(tensor_image);
        fake_image_regular.push_back(tensor_regular);
    }

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ov::test::utils::make_conv_pool_relu({num_batch, feature, height, width});

    using namespace ov::preprocess;

    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_layout("NHWC")
                      .set_memory_type(ov::intel_gpu::memory_type::surface);
    p.input().model().set_layout("NCHW");
    auto function = p.build();

    auto param_input_y = fn_ptr_remote->get_parameters().at(0);

    auto exec_net_b = ie.compile_model(function, ov::test::utils::DEVICE_GPU);
    auto inf_req_remote = exec_net_b.create_infer_request();

    auto cldnn_context = exec_net_b.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context.get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    std::vector<cl_mem> nv12_image_plane_y;
    std::vector<cl::Image2D> img_y;
    std::vector<ov::Tensor> tensor_remote_y;

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

        size_t origin[3] = { 0, 0, 0 };
        size_t y_region[3] = { (size_t)width, (size_t)height, 1 };

        err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y[i],
            true, origin, y_region, 0, 0, fake_image[i].data(), 0, NULL, NULL);
        ASSERT_EQ(err, 0);

        img_y.emplace_back(nv12_image_plane_y[i]);
        tensor_remote_y.emplace_back(cldnn_context.create_tensor(param_input_y->get_element_type(), fake_image[i].get_shape(), img_y[i]));
    }

    // to test repeating set_tensors/infer functionality
    for (size_t i = 0; i < 5; ++i) {
        inf_req_remote.set_tensors(*param_input_y->output(0).get_tensor().get_names().begin(), tensor_remote_y);
        inf_req_remote.infer();
    }

    auto output_tensor_shared = inf_req_remote.get_tensor(function->get_results().at(0));
    OV_ASSERT_NO_THROW(output_tensor_shared.data());

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ov::test::utils::make_conv_pool_relu({1, 1, height, width});

    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::f32)
                          .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p_reg.input().model().set_layout("NHWC");
    auto function_regular = p_reg.build();

    auto param_input_y_reg = fn_ptr_regular->get_parameters().at(0);

    auto exec_net_regular = ie.compile_model(function_regular, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();

    for (size_t i = 0; i < num_batch; ++i) {
        inf_req_regular.set_tensor(param_input_y_reg, fake_image_regular[i]);
        inf_req_regular.infer();
        auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

        ASSERT_EQ(output_tensor_regular.get_size() * num_batch, output_tensor_shared.get_size());
        float thr = 0.1f;

        ov::test::utils::compare_raw_data(static_cast<float*>(output_tensor_shared.data()) + i * output_tensor_regular.get_size(),
                                                static_cast<float*>(output_tensor_regular.data()),
                                                output_tensor_regular.get_size(), thr);
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
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = 50;
        in_data.resolution = 1;
        in_data.seed = static_cast<int>(i);
        fake_image_data_y.push_back(ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height, width, 1}, in_data));
        in_data.range = 256;
        fake_image_data_uv.push_back(ov::test::utils::create_and_fill_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, in_data));
    }

    auto ie = ov::Core();

    // ------------------------------------------------------
    // inference using remote tensor
    auto fn_ptr_remote = ov::test::utils::make_conv_pool_relu({num_batch, 3, height, width});

    using namespace ov::preprocess;
    auto p = PrePostProcessor(fn_ptr_remote);
    p.input().tensor().set_element_type(ov::element::u8)
                      .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                      .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p.input().preprocess().convert_color(ColorFormat::BGR);
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
    auto out_tensor = ov::test::utils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

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
    OV_ASSERT_NO_THROW(out_tensor.data());

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
    auto out_tensor_new = ov::test::utils::create_and_fill_tensor(output->get_output_element_type(0), output->get_output_shape(0));

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
    OV_ASSERT_NO_THROW(out_tensor_new.data());

    // ------------------------------------------------------
    // regular inference
    auto fn_ptr_regular = ov::test::utils::make_conv_pool_relu({1, 3, height, width});

    using namespace ov::preprocess;
    auto p_reg = PrePostProcessor(fn_ptr_regular);
    p_reg.input().tensor().set_element_type(ov::element::u8)
                          .set_color_format(ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                          .set_memory_type(ov::intel_gpu::memory_type::buffer);
    p_reg.input().preprocess().convert_color(ColorFormat::BGR);
    p_reg.input().model().set_layout("NCHW");
    auto function_regular = p_reg.build();

    auto param_input_y_reg = fn_ptr_regular->get_parameters().at(0);
    auto param_input_uv_reg = fn_ptr_regular->get_parameters().at(1);

    auto exec_net_regular = ie.compile_model(function_regular, ov::test::utils::DEVICE_GPU);
    auto inf_req_regular = exec_net_regular.create_infer_request();

    for (size_t i = 0; i < num_batch; ++i) {
        inf_req_regular.set_tensor(param_input_y_reg, fake_image_data_y[i + num_batch]);
        inf_req_regular.set_tensor(param_input_uv_reg, fake_image_data_uv[i + num_batch]);
        inf_req_regular.infer();
        auto output_tensor_regular = inf_req_regular.get_tensor(exec_net_regular.output());

        ASSERT_EQ(output_tensor_regular.get_size() * num_batch, out_tensor_new.get_size());
        float thr = 0.1f;

        ov::test::utils::compare_raw_data(static_cast<float*>(out_tensor_new.data()) + i * output_tensor_regular.get_size(),
                                          static_cast<float*>(output_tensor_regular.data()),
                                          output_tensor_regular.get_size(), thr);
    }
}

const std::vector<size_t> num_batches{ 1, 2, 4 };
INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVRemoteTensorBatched_Test, ::testing::ValuesIn(num_batches), OVRemoteTensorBatched_Test::getTestCaseName);

static void check_contexts_are_same(const ov::RemoteContext& c1, const ov::RemoteContext& c2) {
    ASSERT_EQ(c1.get_device_name(), c2.get_device_name());

    // If we support other context type this check must be replaced
    ASSERT_TRUE(c1.is<ov::intel_gpu::ocl::ClContext>());
    ASSERT_TRUE(c2.is<ov::intel_gpu::ocl::ClContext>());

    auto c1_casted = c1.as<ov::intel_gpu::ocl::ClContext>();
    auto c2_casted = c2.as<ov::intel_gpu::ocl::ClContext>();

    ASSERT_EQ(c1_casted.get(), c2_casted.get());
}

TEST(OVRemoteContextGPU, smoke_CustomContextDeviceNames) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();
    std::vector<std::string> gpuDevices;
    std::vector<std::string> availableDevices = core.get_available_devices();

    std::for_each(availableDevices.begin(), availableDevices.end(), [&](const std::string& device){
        if (device.find(ov::test::utils::DEVICE_GPU) != std::string::npos)
            gpuDevices.push_back(device);
    });

    for (size_t i = 0; i < gpuDevices.size(); i++) {
        auto device_name = "GPU." + std::to_string(i);
        auto ctx = core.get_default_context(device_name).as<ov::intel_gpu::ocl::ClContext>();
        cl::Context original_ctx_handle = ctx;
        std::vector<cl::Device> devices = original_ctx_handle.getInfo<CL_CONTEXT_DEVICES>();
        cl::Context new_ctx_handle(devices);
        ASSERT_NE(new_ctx_handle.get(), original_ctx_handle.get());
        auto remote_context = ov::intel_gpu::ocl::ClContext(core, new_ctx_handle.get(), 0);
        ASSERT_EQ(remote_context.get_device_name(), device_name);

        // Check that ctx_device_id doesn't impact device name reported by context
        cl::Context new_ctx_handle_md({devices.front(), devices.front()});
        ASSERT_NE(original_ctx_handle.get(), new_ctx_handle_md.get());
        auto remote_context0 = ov::intel_gpu::ocl::ClContext(core, new_ctx_handle_md.get(), 0);
        auto remote_context1 = ov::intel_gpu::ocl::ClContext(core, new_ctx_handle_md.get(), 1);
        ASSERT_EQ(remote_context0.get_device_name(), device_name);
        ASSERT_EQ(remote_context1.get_device_name(), device_name);
    }
}

TEST(OVRemoteContextGPU, smoke_CantCreateContextForNullHandle) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();
    OV_EXPECT_THROW(ov::intel_gpu::ocl::ClContext(core, nullptr, 0), ov::Exception, HasSubstr("Can't create shared OCL context as user handle is nullptr!"));
}

TEST(OVRemoteContextGPU, smoke_RemoteContextPerDevice) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();
    std::vector<std::string> gpuDevices;
    std::vector<std::string> availableDevices = core.get_available_devices();

    std::for_each(availableDevices.begin(), availableDevices.end(), [&](const std::string& device){
        if (device.find(ov::test::utils::DEVICE_GPU) != std::string::npos)
            gpuDevices.push_back(device);
    });

    if (gpuDevices.size() < 2)
        GTEST_SKIP();

    const auto gpuDeviceFirst = gpuDevices[0];
    const auto gpuDeviceSecond = gpuDevices[1];

    auto defaultContextFirst = core.get_default_context(gpuDeviceFirst);
    auto defaultContextSecond = core.get_default_context(gpuDeviceSecond);

    // Check devices names
    ASSERT_EQ(defaultContextFirst.get_device_name(), gpuDeviceFirst);
    ASSERT_EQ(defaultContextSecond.get_device_name(), gpuDeviceSecond);
}

TEST(OVRemoteContextGPU, smoke_RemoteContextCaching) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();
    std::vector<std::string> gpuDevices;
    std::vector<std::string> availableDevices = core.get_available_devices();

    std::for_each(availableDevices.begin(), availableDevices.end(), [&](const std::string& device){
        if (device.find(ov::test::utils::DEVICE_GPU) != std::string::npos)
            gpuDevices.push_back(device);
    });

    if (gpuDevices.size() < 2)
        GTEST_SKIP();

    const auto gpuDeviceFirst = gpuDevices[0];
    const auto gpuDeviceSecond = gpuDevices[1];
    auto model = ov::test::utils::make_convert_transpose();

    auto compiledModelFirst = core.compile_model(model, gpuDeviceFirst);
    auto compiledModelSecond = core.compile_model(model, gpuDeviceSecond);

    auto compiledModelFirstContext = compiledModelFirst.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto compiledModelSecondContext = compiledModelSecond.get_context().as<ov::intel_gpu::ocl::ClContext>();

    auto defaultContextFirst = core.get_default_context(gpuDeviceFirst).as<ov::intel_gpu::ocl::ClContext>();
    // Check devices names
    ASSERT_EQ(defaultContextFirst.get_device_name(), gpuDeviceFirst);
    check_contexts_are_same(compiledModelFirstContext, defaultContextFirst);

    auto defaultContextSecond = core.get_default_context(gpuDeviceSecond).as<ov::intel_gpu::ocl::ClContext>();
    // Check devices names
    ASSERT_EQ(defaultContextSecond.get_device_name(), gpuDeviceSecond);
    // Check underlying OpenCL context handles
    ASSERT_EQ(compiledModelSecondContext.get(), compiledModelSecondContext.get());

    // Expect different contexts for different devices
    ASSERT_NE(compiledModelFirstContext.get(), compiledModelSecondContext.get());
}

TEST(OVRemoteContextGPU, smoke_RemoteContextSingleDevice) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();

    auto default_ctx = core.get_default_context(ov::test::utils::DEVICE_GPU).as<ov::intel_gpu::ocl::ClContext>();

    // Same context returned for multple calls
    check_contexts_are_same(default_ctx, core.get_default_context(ov::test::utils::DEVICE_GPU));

    // Set some properties which could impact engine config and check context again
    core.set_property(ov::test::utils::DEVICE_GPU, ov::streams::num(2));
    core.set_property(ov::test::utils::DEVICE_GPU, ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::LOW));
    core.set_property(ov::test::utils::DEVICE_GPU, ov::enable_profiling(true));
    check_contexts_are_same(default_ctx,  core.get_default_context(ov::test::utils::DEVICE_GPU));

    // Ensure compiled model uses default context too
    auto model = ov::test::utils::make_convert_transpose();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    check_contexts_are_same(default_ctx, compiled_model.get_context());
    ASSERT_EQ(2, compiled_model.get_property(ov::streams::num));

    auto ocl_instance = std::make_shared<OpenCL>();
    cl::Context default_ctx_handle = default_ctx;
    auto default_devices = default_ctx_handle.getInfo<CL_CONTEXT_DEVICES>();
    ASSERT_EQ(default_devices.size(), 1);
    cl::Device default_device_handle(default_devices[0]);
    // OCL instance looks for intel GPUs, so skip this part if ov::test::utils::DEVICE_GPU points to GPU from other vendor
    if (default_device_handle.getInfo<CL_DEVICE_VENDOR_ID>() == 0x8086) {
        ov::intel_gpu::ocl::ClContext custom_ctx(core, ocl_instance->_queue.get());
        auto compiled_model_custom_ctx = core.compile_model(model, custom_ctx, ov::streams::num(1));
        auto model_ctx = compiled_model_custom_ctx.get_context().as<ov::intel_gpu::ocl::ClContext>();

        // Check that compiled model uses custom context
        check_contexts_are_same(custom_ctx, model_ctx);
        ASSERT_EQ(1, compiled_model_custom_ctx.get_property(ov::streams::num));

        // Check that handle differs in default context and compiled model created with custom ctx
        ASSERT_NE(default_ctx.get(), model_ctx.get());

        // Check that default ctx is untouched
        check_contexts_are_same(default_ctx, core.get_default_context(ov::test::utils::DEVICE_GPU));
    }
}

TEST(OVRemoteContextGPU, smoke_RemoteTensorSetShape) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();
    auto context = core.get_default_context(ov::test::utils::DEVICE_GPU);

    auto remote_tensor = context.create_tensor(ov::element::f32, ov::Shape{1, 2, 3, 4});

    OV_ASSERT_NO_THROW(remote_tensor.set_shape({2, 3, 4, 5}));
    OV_ASSERT_NO_THROW(remote_tensor.set_shape({1, 3, 4, 5}));
    OV_ASSERT_NO_THROW(remote_tensor.set_shape({3, 3, 4, 5}));
}

TEST(RemoteTensor, smoke_CopyToEmptyTensor) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();
    auto context = core.get_default_context(ov::test::utils::DEVICE_GPU);

    auto empty_remote_tensor = ov::RemoteTensor();
    auto remote_tensor = context.create_tensor(ov::element::f32, ov::Shape{1, 2, 3, 4});

    OV_EXPECT_THROW_HAS_SUBSTRING(empty_remote_tensor.copy_to(remote_tensor),
                                  ov::Exception,
                                  "Check '_impl != nullptr' failed");
}

TEST(RemoteTensor, smoke_EmptyRoiTensor) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif

    auto empty_remote_tensor = ov::RemoteTensor();

    OV_EXPECT_THROW_HAS_SUBSTRING(ov::RemoteTensor(empty_remote_tensor, ov::Coordinate{}, ov::Coordinate{}),
                                  ov::Exception,
                                  "Cannot create RoiRemoteTensor on top of empty tensor");
}

struct TestParams {
    ov::Shape src_shape;
    ov::Shape dst_shape;
    ov::Coordinate begin;
    ov::Coordinate end;
};

struct RemoteTensor : ::testing::TestWithParam<std::tuple<ov::element::Type, RemoteTensorSharingType, TestParams>> {};

namespace {
template <class T>
std::vector<T> fill_data(const ov::Tensor& tensor) {
    std::vector<T> actual;
    const T* data = tensor.data<T>();
    auto strides = tensor.get_strides();
    for (auto&& c : ov::CoordinateTransformBasic{tensor.get_shape()}) {
        size_t offset = 0;
        for (size_t i = 0; i < strides.size(); i++)
            offset += c[i] * strides[i];
        actual.emplace_back(*(data + offset / tensor.get_element_type().size()));
    }
    return actual;
}

template <class T>
void compare_data(const ov::Tensor& src, const ov::Tensor& dst) {
    auto source_vec = fill_data<T>(src);
    auto dest_vec = fill_data<T>(dst);

    ASSERT_EQ(source_vec.size(), dest_vec.size());

    for (size_t i = 0; i < source_vec.size(); i++) {
        ASSERT_EQ(source_vec[i], dest_vec[i]);
    }
}

template <ov::element::Type_t ET,
          typename T = typename ov::element_type_traits<ET>::value_type>
void init_tensor(const ov::Tensor& tensor) {
    const auto origPtr = tensor.data<T>();
    ASSERT_NE(nullptr, origPtr);
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        origPtr[i] = static_cast<T>(i);
    }
}

void init_tensor(const ov::Tensor& tensor) {
    switch (tensor.get_element_type()) {
    case ov::element::f32:
        init_tensor<ov::element::f32>(tensor);
        break;
    case ov::element::u8:
        init_tensor<ov::element::u8>(tensor);
        break;
    default:
        OPENVINO_THROW("Unsupported data type");
    }
}

void compare_tensors(const ov::Tensor& src, const ov::Tensor& dst) {
    ASSERT_EQ(src.get_byte_size(), dst.get_byte_size());
    ASSERT_EQ(src.get_size(), dst.get_size());
    ASSERT_EQ(src.get_element_type(), dst.get_element_type());
    switch (src.get_element_type()) {
    case ov::element::f32:
        compare_data<ov::element_type_traits<ov::element::f32>::value_type>(src, dst);
        break;
    case ov::element::u8:
        compare_data<ov::element_type_traits<ov::element::u8>::value_type>(src, dst);
        break;
    default:
        OPENVINO_THROW("Unsupported data type");
    }
}

ov::RemoteTensor create_tensor(ov::intel_gpu::ocl::ClContext context,
                               RemoteTensorSharingType sharing_type,
                               const ov::element::Type& type,
                               const ov::Shape& shape) {
    switch (sharing_type) {
        case RemoteTensorSharingType::PLUGIN_CL_TENSOR:
            return context.create_tensor(type, shape);
        case RemoteTensorSharingType::PLUGIN_HOST_TENSOR:
            return context.create_usm_host_tensor(type, shape);
        case  RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR:
            return context.create_usm_device_tensor(type, shape);
        default:
            OPENVINO_THROW("Unsupported tensor allocation type");
    }
}
}  // namespace

TEST(RemoteTensor, smoke_LockableHandling) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif

    auto core = ov::Core();
    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    auto gpu_context = remote_context.as<ov::intel_gpu::ocl::ClContext>();
    auto type = ov::element::f32;
    ov::Shape shape = {4};

    auto remote_tensor = gpu_context.create_tensor(type, shape);

    auto host_tensor_in = ov::Tensor(type, shape);
    init_tensor(host_tensor_in);
    remote_tensor.copy_from(host_tensor_in);

    auto param_node = std::make_shared<ov::op::v0::Parameter>(type, ov::PartialShape{-1});
    auto const_node = std::make_shared<ov::op::v0::Constant>(host_tensor_in);
    auto add_node = std::make_shared<ov::op::v1::Add>(param_node, const_node);
    auto shape_of_node = std::make_shared<ov::op::v3::ShapeOf>(param_node);
    auto res1 = std::make_shared<ov::op::v0::Result>(add_node);
    auto res2 = std::make_shared<ov::op::v0::Result>(shape_of_node);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res1, res2}, ov::ParameterVector{param_node});

    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f32));
    auto request = compiled_model.create_infer_request();
    request.set_input_tensor(remote_tensor);

    request.infer();
    auto res = request.get_output_tensor(0);
    auto host_res = ov::Tensor(type, shape);
    res.copy_to(host_res);

    for (size_t i = 0; i < ov::shape_size(host_tensor_in.get_shape()); i++) {
        ASSERT_EQ(host_res.data<float>()[i], host_tensor_in.data<float>()[i] * 2);
    }
}

TEST_P(RemoteTensor, smoke_CopyFrom) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    ov::element::Type type;
    TestParams p;
    RemoteTensorSharingType sharing_type;
    std::tie(type, sharing_type, p) = GetParam();

    auto core = ov::Core();
    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    auto gpu_context = remote_context.as<ov::intel_gpu::ocl::ClContext>();
    bool use_roi = p.begin != ov::Coordinate{} && p.end != ov::Coordinate{};

    auto host_tensor_ref = ov::Tensor(type, p.src_shape);
    init_tensor(host_tensor_ref);

    auto first_remote_tensor = create_tensor(gpu_context, sharing_type, type, p.src_shape);
    auto second_remote_tensor = create_tensor(gpu_context, sharing_type, type, p.dst_shape);

    // Copy from remote tensor to remote tensor
    second_remote_tensor.copy_from(first_remote_tensor);

    // Check updated shape after copy_from call
    ASSERT_EQ(second_remote_tensor.get_shape(), first_remote_tensor.get_shape());

    // Copy from host tensor to remote tensor
    if (use_roi) {
        auto roi_host_tensor_ref = ov::Tensor(host_tensor_ref, p.begin, p.end);
        auto roi_second_remote_tensor = ov::RemoteTensor(second_remote_tensor, p.begin, p.end);
        auto second_remote_tensor_shape = second_remote_tensor.get_shape();

        roi_second_remote_tensor.copy_from(roi_host_tensor_ref);

        // Ensure that the shape of the underlying RemoteTensor of RoiRemoteTensor remains unchanged
        ASSERT_EQ(second_remote_tensor.get_shape(), second_remote_tensor_shape);
        ASSERT_EQ(roi_second_remote_tensor.get_shape(), roi_host_tensor_ref.get_shape());

        auto result_host_tensor = ov::Tensor(type, roi_second_remote_tensor.get_shape());
        roi_second_remote_tensor.copy_to(result_host_tensor);

        compare_tensors(roi_host_tensor_ref, result_host_tensor);
    } else {
        second_remote_tensor.copy_from(host_tensor_ref);

        auto result_host_tensor = ov::Tensor(type, p.src_shape);
        second_remote_tensor.copy_to(result_host_tensor);

        compare_tensors(host_tensor_ref, result_host_tensor);
    }
}

TEST_P(RemoteTensor, smoke_CopyTo) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    ov::element::Type type;
    TestParams p;
    RemoteTensorSharingType sharing_type;
    std::tie(type, sharing_type, p) = GetParam();

    auto core = ov::Core();
    auto remote_context = core.get_default_context(ov::test::utils::DEVICE_GPU);
    auto gpu_context = remote_context.as<ov::intel_gpu::ocl::ClContext>();
    bool use_roi = p.begin != ov::Coordinate{} && p.end != ov::Coordinate{};

    auto host_tensor_ref = ov::Tensor(type, p.src_shape);
    init_tensor(host_tensor_ref);

    auto first_remote_tensor = create_tensor(gpu_context, sharing_type, type, p.src_shape);
    auto second_remote_tensor = create_tensor(gpu_context, sharing_type, type, p.dst_shape);

    // Copy to remote tensor from remote tensor
    first_remote_tensor.copy_to(second_remote_tensor);

    // Check updated shape after copy_to call
    ASSERT_EQ(second_remote_tensor.get_shape(), first_remote_tensor.get_shape());

    // Copy to remote tensor from host tensor
    if (use_roi) {
        auto roi_host_tensor_ref = ov::Tensor(host_tensor_ref, p.begin, p.end);
        auto roi_second_remote_tensor = ov::RemoteTensor(second_remote_tensor, p.begin, p.end);
        auto second_remote_tensor_shape = second_remote_tensor.get_shape();

        roi_host_tensor_ref.copy_to(roi_second_remote_tensor);

        // Ensure that the shape of the underlying RemoteTensor of RoiRemoteTensor remains unchanged
        ASSERT_EQ(second_remote_tensor.get_shape(), second_remote_tensor_shape);
        ASSERT_EQ(roi_second_remote_tensor.get_shape(), roi_host_tensor_ref.get_shape());

        auto result_host_tensor = ov::Tensor(type, roi_second_remote_tensor.get_shape());
        roi_second_remote_tensor.copy_to(result_host_tensor);

        compare_tensors(roi_host_tensor_ref, result_host_tensor);
    } else {
        host_tensor_ref.copy_to(second_remote_tensor);

        auto host_tensor = ov::Tensor(type, p.src_shape);
        second_remote_tensor.copy_to(host_tensor);

        compare_tensors(host_tensor_ref, host_tensor);
    }
}

INSTANTIATE_TEST_SUITE_P(copy_tests,
                         RemoteTensor,
                         ::testing::Combine(::testing::Values(ov::element::u8,
                                                              ov::element::f32),
                                            ::testing::Values(RemoteTensorSharingType::PLUGIN_CL_TENSOR,
                                                              RemoteTensorSharingType::PLUGIN_USM_DEVICE_TENSOR,
                                                              RemoteTensorSharingType::PLUGIN_HOST_TENSOR),
                                            ::testing::Values(TestParams {
                                                                  ov::Shape{4, 3, 2, 5}, ov::Shape{4, 3, 2, 5},
                                                                  ov::Coordinate{}, ov::Coordinate{}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{4, 3, 2, 5}, ov::Shape{1, 3, 2, 5},
                                                                  ov::Coordinate{}, ov::Coordinate{}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{4, 3, 2, 5}, ov::Shape{4, 3, 2, 5},
                                                                  ov::Coordinate{0, 0, 0, 0}, ov::Coordinate{2, 3, 2, 5}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{4, 3, 2, 5}, ov::Shape{4, 3, 2, 5},
                                                                  ov::Coordinate{0, 0, 0, 4}, ov::Coordinate{2, 3, 2, 5}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{4, 3, 2, 5}, ov::Shape{4, 3, 2, 5},
                                                                  ov::Coordinate{2, 1, 1, 4}, ov::Coordinate{4, 3, 2, 5}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{4, 3, 2, 5}, ov::Shape{4, 3, 2, 5},
                                                                  ov::Coordinate{2, 0, 1, 0}, ov::Coordinate{4, 3, 2, 5}
                                                              },
                                                              TestParams {
                                                                  ov::Shape{4, 3, 2, 5}, ov::Shape{4, 3, 2, 5},
                                                                  ov::Coordinate{0, 1, 1, 0}, ov::Coordinate{4, 2, 2, 3}
                                                              })));

TEST(RemoteTensor, smoke_CanSetRoiRemoteTensor) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto core = ov::Core();
    auto model = ov::test::behavior::getDefaultNGraphFunctionForTheDevice();

    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);

    auto input = model->get_parameters().at(0);
    auto output = model->get_results().at(0);

    auto input_shape = input->get_shape();
    auto output_shape = output->get_shape();

    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    cl_context ctx = gpu_context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);

    auto host_tensor = ov::Tensor(input->get_element_type(), input_shape);
    init_tensor(host_tensor);

    auto output_tensor_copy_0 = ov::Tensor(output->get_element_type(), output_shape);
    auto output_tensor_copy_1 = ov::Tensor(output->get_element_type(), output_shape);

    auto infer_request = compiled_model.create_infer_request();
    {
        auto user_input_tensor = gpu_context.create_tensor(input->get_element_type(), input_shape);
        auto user_output_tensor = gpu_context.create_tensor(output->get_element_type(), output_shape);

        user_input_tensor.copy_from(host_tensor);

        infer_request.set_tensor(input, user_input_tensor);
        infer_request.set_tensor(output, user_output_tensor);

        OV_ASSERT_NO_THROW(infer_request.infer());

        auto output_tensor = infer_request.get_tensor(output);
        ASSERT_EQ(output_tensor.get_shape(), output_shape);

        output_tensor.copy_to(output_tensor_copy_0);
    }
    {
        auto larger_input_shape = input_shape;
        for (size_t i = 0; i < input_shape.size(); i++)
            larger_input_shape[i] += 2;

        auto larger_output_shape = input_shape;
        for (size_t i = 0; i < input_shape.size(); i++)
            larger_output_shape[i] += 2;

        auto roi_begin = ov::Coordinate(input_shape.size(), 2);
        auto roi_end = ov::Coordinate(larger_input_shape);

        auto user_input_tensor = gpu_context.create_tensor(input->get_element_type(), larger_input_shape);
        auto user_input_tensor_roi = ov::RemoteTensor(user_input_tensor, roi_begin, roi_end);
        auto user_output_tensor = gpu_context.create_tensor(output->get_element_type(), larger_output_shape);
        auto user_output_tensor_roi = ov::RemoteTensor(user_output_tensor, roi_begin, roi_end);

        user_input_tensor_roi.copy_from(host_tensor);

        infer_request.set_tensor(input, user_input_tensor_roi);
        infer_request.set_tensor(output, user_output_tensor_roi);

        OV_ASSERT_NO_THROW(infer_request.infer());

        auto output_tensor = infer_request.get_tensor(output);
        ASSERT_EQ(output_tensor.get_shape(), output_shape);

        output_tensor.copy_to(output_tensor_copy_1);
    }

    compare_tensors(output_tensor_copy_0, output_tensor_copy_1);
}


using RemoteTensorDataTypesOptionsParams = std::tuple<ov::element::Type_t>;
class OVRemoteTensorDataType_Test : public OVRemoteTensor_Test,
        public testing::WithParamInterface<RemoteTensorDataTypesOptionsParams> {
protected:
    std::shared_ptr<ov::Model> fn_ptr;
    std::string deviceName;
    ov::AnyMap config;
    ov::element::Type_t element_type;

public:
    void SetUp() override {
        deviceName = ov::test::utils::DEVICE_GPU;
        std::tie(element_type) = this->GetParam();
        config = {ov::hint::inference_precision(ov::element::f16),
                ov::hint::model_priority(ov::hint::Priority::HIGH),
                ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE),
                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)};

        auto input1 = std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape{1, 2, 10, 10});
        auto constant = ov::op::v0::Constant::create(element_type, ov::Shape{1, 2, 10, 10}, {1});
        auto add = std::make_shared<ov::op::v1::Add>(input1, constant);
        fn_ptr = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input1});
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RemoteTensorDataTypesOptionsParams>& obj) {
        ov::element::Type_t elem_type;
        std::tie(elem_type) = obj.param;

        std::ostringstream result;
        result << "OVRemoteTensorTest_" << elem_type;
        return result.str();
    }
};

TEST_P(OVRemoteTensorDataType_Test, smoke_RemoteTensorDataType) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto ppp = ov::preprocess::PrePostProcessor(fn_ptr);
    ppp.output(0).tensor().set_element_type(element_type);
    auto ov_model = ppp.build();

    auto core = ov::Core();
    ov::CompiledModel compiled_model = core.compile_model(ov_model, deviceName, config);

    // regular inference
    auto inf_req = compiled_model.create_infer_request();
    auto input_element_type = inf_req.get_input_tensor(0).get_element_type();
    auto input_shape = inf_req.get_input_tensor(0).get_shape();
    auto output_element_type = inf_req.get_output_tensor(0).get_element_type();
    auto output_shape = inf_req.get_output_tensor(0).get_shape();

    ASSERT_EQ(input_element_type, element_type);
    ASSERT_EQ(output_element_type, element_type);

    auto remote_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    auto input_tensor = ov::test::utils::create_and_fill_tensor(input_element_type, input_shape);
    auto output_tensor = ov::test::utils::create_and_fill_tensor(output_element_type, output_shape);

    auto input_cl_tensor = remote_context.create_tensor(input_element_type, input_shape);
    auto output_cl_tensor =  remote_context.create_tensor(output_element_type, output_shape);

    input_cl_tensor.copy_from(input_tensor);

    inf_req.set_input_tensor(0, input_tensor);
    inf_req.set_output_tensor(0, output_tensor);
    inf_req.infer();

    inf_req.set_input_tensor(0, input_cl_tensor);
    inf_req.set_output_tensor(0, output_cl_tensor);
    inf_req.infer();

    auto tmp_tensor = ov::Tensor(output_element_type, output_shape);
    output_cl_tensor.copy_to(tmp_tensor);

    if (element_type == ov::element::i16) {
        compare_data<ov::element_type_traits<ov::element::i16>::value_type>(output_tensor, tmp_tensor);
    } else if (element_type == ov::element::u16) {
        compare_data<ov::element_type_traits<ov::element::u16>::value_type>(output_tensor, tmp_tensor);
    } else if (element_type == ov::element::u32) {
        compare_data<ov::element_type_traits<ov::element::u32>::value_type>(output_tensor, tmp_tensor);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensorDataType, OVRemoteTensorDataType_Test,
                         ::testing::Combine(::testing::Values(ov::element::Type_t::i16,
                                                              ov::element::Type_t::u16,
                                                              ov::element::Type_t::u32)),
                         OVRemoteTensorDataType_Test::getTestCaseName);
