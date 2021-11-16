// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/runtime/gpu/ocl/ocl.hpp"
#include "openvino/runtime/core.hpp"

#include <gpu/gpu_config.hpp>
#include <remote_blob_tests/remote_blob_helpers.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/plugin_cache.hpp>
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

TEST_F(OVRemoteTensor_Test, smoke_canInputUserTensor) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto ie = ov::runtime::Core();

    using namespace ov::preprocess;
    auto function = PrePostProcessor(fn_ptr)
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(ov::element::i8))
                           .preprocess(PreProcessSteps().convert_element_type(ov::element::f32)))
            .build();

    auto exec_net = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);

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
    auto cldnn_context = exec_net.get_context().as<ov::runtime::gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context;
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    auto imSize = ov::shape_size(input->get_shape());

    cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imSize, NULL, &err);
    {
        void* buffer = fakeImageData.data();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
    }

    auto cldnn_tensor = cldnn_context.create_tensor(input->get_element_type(), input->get_shape(), shared_buffer);
    inf_req_shared.set_tensor(input, cldnn_tensor);

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

TEST_F(OVRemoteTensor_Test, smoke_canInferOnUserContext) {
    auto ie = ov::runtime::Core();

    using namespace ov::preprocess;
    auto function = PrePostProcessor(fn_ptr)
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(ov::element::i8))
                           .preprocess(PreProcessSteps().convert_element_type(ov::element::f32)))
            .build();

    auto exec_net_regular = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
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

    auto remote_context = ov::runtime::gpu::ocl::ClContext(ie, ocl_instance->_context.get());
    auto exec_net_shared = ie.compile_model(function, remote_context);
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

TEST_F(OVRemoteTensor_Test, smoke_canInferOnUserContextWithMultipleDevices) {
    auto ie = ov::runtime::Core();

    using namespace ov::preprocess;
    auto function = PrePostProcessor(fn_ptr)
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(ov::element::i8))
                           .preprocess(PreProcessSteps().convert_element_type(ov::element::f32)))
            .build();

    auto exec_net_regular = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
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

    auto remote_context = ov::runtime::gpu::ocl::ClContext(ie, ocl_instance->_context.get(), 1);

    ASSERT_EQ(remote_context.get_device_name(), "GPU.0");
    auto exec_net_shared = ie.compile_model(function, remote_context);
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

TEST_F(OVRemoteTensor_Test, smoke_canInferOnUserQueue_out_of_order) {
    auto ie = ov::runtime::Core();

    using namespace ov::preprocess;
    auto function = PrePostProcessor(fn_ptr)
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(ov::element::i8))
                           .preprocess(PreProcessSteps().convert_element_type(ov::element::f32)))
            .build();

    auto exec_net_regular = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
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

    auto remote_context = ov::runtime::gpu::ocl::ClContext(ie, ocl_instance->_queue.get());
    auto exec_net_shared = ie.compile_model(function, remote_context);
    auto gpu_context = exec_net_shared.get_context().as<ov::runtime::gpu::ocl::ClContext>();

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

TEST_F(OVRemoteTensor_Test, smoke_canInferOnUserQueue_in_order) {
    auto ie = ov::runtime::Core();

    using namespace ov::preprocess;
    auto function = PrePostProcessor(fn_ptr)
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(ov::element::i8))
                           .preprocess(PreProcessSteps().convert_element_type(ov::element::f32)))
            .build();

    auto exec_net_regular = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
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

    auto remote_context = ov::runtime::gpu::ocl::ClContext(ie, ocl_instance->_queue.get());
    auto exec_net_shared = ie.compile_model(function, remote_context);
    auto gpu_context = exec_net_shared.get_context().as<ov::runtime::gpu::ocl::ClContext>();

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

TEST_P(OVRemoteTensorBatched_Test, DISABLED_canInputNV12) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    std::vector<ov::runtime::Tensor> fake_image_data_y;
    std::vector<ov::runtime::Tensor> fake_image_data_uv;

    for (int i = 0; i < num_batch; i++) {
        fake_image_data_y.push_back(FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 1, height, width}, 50, 0, 1, i));
        fake_image_data_uv.push_back(FuncTestUtils::create_and_fill_tensor(ov::element::u8, {1, 2, height / 2, width / 2}, 256, 0, 1, i));
    }

    auto ie = ov::runtime::Core();

    // ------------------------------------------------------
    // inference using remote tensor with batch
    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({num_batch, 3, height, width});

    using namespace ov::preprocess;
    auto function = PrePostProcessor(fn_ptr_remote)
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(ov::element::i8).set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES))
                           .preprocess(PreProcessSteps().convert_element_type(ov::element::f32)))
            .build();

    auto exec_net_b = ie.compile_model(fn_ptr_remote, CommonTestUtils::DEVICE_GPU);
    auto inf_req_remote = exec_net_b.create_infer_request();
    auto cldnn_context = exec_net_b.get_context().as<ov::runtime::gpu::ocl::ClContext>();
    cl_context ctx = cldnn_context.get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    std::vector<cl_mem> nv12_image_plane_y, nv12_image_plane_uv;
    std::vector<cl::Image2D> img_y, img_uv;
    std::vector<std::pair<ov::runtime::RemoteTensor, ov::runtime::RemoteTensor>> tensor_remote;

    for (int i = 0; i < num_batch; i++) {
        cl_image_format image_format;
        cl_image_desc image_desc = { 0 };
        image_format.image_channel_order = CL_R;
        image_format.image_channel_data_type = CL_UNORM_INT8;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        image_desc.image_width = width;
        image_desc.image_height = height;

        nv12_image_plane_y.push_back(clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err));
        ASSERT_EQ(err, 0);

        image_format.image_channel_order = CL_RG;
        image_desc.image_width = width / 2;
        image_desc.image_height = height / 2;

        nv12_image_plane_uv.push_back(clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err));
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

        img_y.push_back(cl::Image2D(nv12_image_plane_y[i]));
        img_uv.push_back(cl::Image2D(nv12_image_plane_uv[i]));

        tensor_remote.push_back(cldnn_context.create_tensor_nv12(img_y[i], img_uv[i]));
    }

    if (num_batch == 1) {
        inf_req_remote.set_tensor(fn_ptr_remote->get_parameters().front()->get_friendly_name() + "/y", tensor_remote[0].first);
        inf_req_remote.set_tensor(fn_ptr_remote->get_parameters().front()->get_friendly_name() + "/uv", tensor_remote[0].second);
    } else {
        GTEST_SKIP() << "Not implemented test";
    }

    inf_req_remote.infer();

    auto outputTensor_shared = inf_req_remote.get_tensor(
        ngraph::op::util::create_ie_output_name(fn_ptr_remote->get_results().front()->input_value(0)));

    // ------------------------------------------------------
    // Setup to inference using local tensor with batch=1
    auto fn_ptr_local = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, height, width});

    // net_local.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    // net_local.getInputsInfo().begin()->second->setPrecision(Precision::U8);
    // net_local.getInputsInfo().begin()->second->getPreProcess().setColorFormat(ColorFormat::NV12);

    auto exec_net_b1 = ie.compile_model(fn_ptr_local, CommonTestUtils::DEVICE_GPU);

    auto inf_req_local = exec_net_b1.create_infer_request();

    // Run regular input for each image and compare against batched tensor
    for (int i = 0; i < num_batch; i++) {
        auto y_tensor = ov::runtime::Tensor{ov::element::u8, {1, 1, height, width}};
        auto uv_tensor = ov::runtime::Tensor{ov::element::u8, {1, 2, height / 2, width / 2}};
        inf_req_local.set_tensor(fn_ptr_local->get_parameters().front()->get_friendly_name() + "/y", y_tensor);
        inf_req_local.set_tensor(fn_ptr_local->get_parameters().front()->get_friendly_name() + "/uv", uv_tensor);
        inf_req_local.infer();
        auto output_tensor_local = inf_req_local.get_tensor(
            ngraph::op::util::create_ie_output_name(fn_ptr_local->get_results().front()->input_value(0)));

        // This network generates [1, size] tensor whether batch=1 or 2. So need to split
        auto split_shared_tensor = ov::runtime::Tensor{output_tensor_local.get_element_type(),
                                        output_tensor_local.get_shape(),
                                        outputTensor_shared.data<float_t>() + output_tensor_local.get_size() * i};
        ASSERT_EQ(output_tensor_local.get_size(), split_shared_tensor.get_size());
        float thr = 0.1;

        FuncTestUtils::compare_tensor(output_tensor_local, split_shared_tensor, thr, "", false);
    }
}

const std::vector<size_t> num_batches{1, 2, 4};

INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVRemoteTensorBatched_Test, ::testing::ValuesIn(num_batches), OVRemoteTensorBatched_Test::getTestCaseName);
