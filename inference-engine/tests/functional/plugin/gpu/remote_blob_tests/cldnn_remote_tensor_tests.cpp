// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/runtime/gpu/ocl.hpp"
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

TEST_F(OVRemoteTensor_Test, DISABLED_smoke_canInputUserTensor) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    auto ie = ov::runtime::Core();

    using namespace ov::preprocess;
    auto function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(ov::element::i8))
                           .preprocess(PreProcessSteps().convert_element_type(ov::element::f32)))
            .build(fn_ptr);

    auto exec_net = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);

    // regular inference
    auto inf_req_regular = exec_net.create_infer_request();
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());

    inf_req_regular.set_tensor(input->get_friendly_name(), fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(ngraph::op::util::create_ie_output_name(output->input_value(0)));

    // inference using remote tensor
    auto inf_req_shared = exec_net.create_infer_request();
    auto cldnn_context = exec_net.get_context().as<ov::runtime::gpu::ClContext>();
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
    inf_req_shared.set_tensor(input->get_friendly_name(), cldnn_tensor);

    inf_req_shared.infer();
    auto output_tensor_shared = inf_req_shared.get_tensor(ngraph::op::util::create_ie_output_name(output->input_value(0)));

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

TEST_F(OVRemoteTensor_Test, DISABLED_smoke_canInferOnUserContext) {
    auto ie = ov::runtime::Core();

    using namespace ov::preprocess;
    auto function = PrePostProcessor()
            .input(InputInfo()
                           .tensor(InputTensorInfo().set_element_type(ov::element::i8))
                           .preprocess(PreProcessSteps().convert_element_type(ov::element::f32)))
            .build(fn_ptr);

    auto exec_net_regular = ie.compile_model(function, CommonTestUtils::DEVICE_GPU);
    auto input = function->get_parameters().at(0);
    auto output = function->get_results().at(0);

    // regular inference
    auto inf_req_regular = exec_net_regular.create_infer_request();
    auto fakeImageData = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
    inf_req_regular.set_tensor(input->get_friendly_name(), fakeImageData);

    inf_req_regular.infer();
    auto output_tensor_regular = inf_req_regular.get_tensor(ngraph::op::util::create_ie_output_name(output->input_value(0)));

    // inference using remote tensor
    auto ocl_instance = std::make_shared<OpenCL>();

    auto remote_context = ov::runtime::gpu::ClContext(ie, ocl_instance->_context.get());
    auto exec_net_shared = ie.compile_model(function, remote_context);
    auto inf_req_shared = exec_net_shared.create_infer_request();
    inf_req_shared.set_tensor(input->get_friendly_name(), fakeImageData);

    inf_req_shared.infer();
    auto output_tensor_shared = inf_req_shared.get_tensor(ngraph::op::util::create_ie_output_name(output->input_value(0)));

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

    // TODO: Add preprocessing!
    // CNNNetwork net_remote(fn_ptr_remote);
    // net_remote.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    // net_remote.getInputsInfo().begin()->second->setPrecision(Precision::U8);
    // net_remote.getInputsInfo().begin()->second->getPreProcess().setColorFormat(ColorFormat::NV12);

    /* XXX: is it correct to set KEY_CLDNN_NV12_TWO_INPUTS in case of remote tensor? */
    auto exec_net_b = ie.compile_model(fn_ptr_remote, CommonTestUtils::DEVICE_GPU,
                { { ov::ie::GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS, ov::ie::PluginConfigParams::YES} });
    auto inf_req_remote = exec_net_b.create_infer_request();
    auto cldnn_context = exec_net_b.get_context().as<ov::runtime::gpu::ClContext>();
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

using TwoNetsParams = std::tuple<size_t,   // number of streams
                                 size_t>;  // number of requests

class OVRemoteTensorTwoNets_Test : public CommonTestUtils::TestsCommon,
    public testing::WithParamInterface<TwoNetsParams> {
    void SetUp() override {
        std::tie(num_streams, num_requests) = this->GetParam();
        fn_ptrs = {ngraph::builder::subgraph::makeSplitMultiConvConcat(),
                   ngraph::builder::subgraph::makeMultiSingleConv()};
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TwoNetsParams>& obj) {
        size_t streams, requests;
        std::tie(streams, requests) = obj.param;
        return "_num_streams_" + std::to_string(streams) + "_num_req_" +
            std::to_string(requests);
    }

protected:
    size_t num_streams;
    size_t num_requests;
    std::vector<std::shared_ptr<ngraph::Function>> fn_ptrs;
};

TEST_P(OVRemoteTensorTwoNets_Test, DISABLED_canInferTwoExecNets) {
    auto ie = ov::runtime::Core();

    std::vector<std::string> outputs;
    std::vector<ov::runtime::InferRequest> irs;
    std::vector<std::vector<uint8_t>> ref;
    std::vector<int> outElementsCount;

    for (size_t i = 0; i < fn_ptrs.size(); ++i) {
        auto fn = fn_ptrs[i];

        auto exec_net = ie.compile_model(fn_ptrs[i], CommonTestUtils::DEVICE_GPU,
                                         {{ov::ie::PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, std::to_string(num_streams)}});

        auto input = fn_ptrs[i]->get_parameters().at(0);
        auto output = fn_ptrs[i]->get_results().at(0);

        for (int j = 0; j < num_streams * num_requests; j++) {
            outputs.push_back(ngraph::op::util::create_ie_output_name(output->input_value(0)));

            auto inf_req = exec_net.create_infer_request();
            irs.push_back(inf_req);

            auto tensor = FuncTestUtils::create_and_fill_tensor(input->get_element_type(), input->get_shape());
            inf_req.set_tensor(input->get_friendly_name(), tensor);

            outElementsCount.push_back(
                    std::accumulate(begin(fn_ptrs[i]->get_output_shape(0)), end(fn_ptrs[i]->get_output_shape(0)), 1,
                                    std::multiplies<size_t>()));
            const auto in_tensor = inf_req.get_tensor(input->get_friendly_name());
            const auto tensorSize = in_tensor.get_byte_size();
            const auto inBlobBuf = static_cast<uint8_t*>(in_tensor.data());
            std::vector<uint8_t> inData(inBlobBuf, inBlobBuf + tensorSize);
            auto reOutData = ngraph::helpers::interpreterFunction(fn_ptrs[i], {inData}).front().second;
            ref.push_back(reOutData);
        }
    }

    const int niter = 10;
    for (int i = 0; i < niter; i++) {
        for (auto ir : irs) {
            ir.start_async();
        }

        for (auto ir : irs) {
            ir.wait();
        }
    }

    auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
    for (size_t i = 0; i < irs.size(); ++i) {
        const auto &refBuffer = ref[i].data();
        ASSERT_EQ(outElementsCount[i], irs[i].get_tensor(outputs[i]).get_size());
        FuncTestUtils::compareRawBuffers(irs[i].get_tensor(outputs[i]).data<float>(),
                                         reinterpret_cast<const float *>(refBuffer), outElementsCount[i],
                                         outElementsCount[i],
                                         thr);
    }
}

const std::vector<size_t> num_streams{ 1, 2 };
const std::vector<size_t> num_requests{ 1, 4 };

INSTANTIATE_TEST_SUITE_P(smoke_RemoteTensor, OVRemoteTensorTwoNets_Test,
    ::testing::Combine(::testing::ValuesIn(num_streams),
        ::testing::ValuesIn(num_requests)),
    OVRemoteTensorTwoNets_Test::getTestCaseName);
