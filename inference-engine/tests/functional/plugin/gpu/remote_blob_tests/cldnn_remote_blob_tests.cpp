// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include <ie_compound_blob.h>

#include <gpu/gpu_config.hpp>
#include <remote_blob_tests/remote_blob_helpers.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/plugin_cache.hpp>

#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::gpu;

class RemoteBlob_Test : public CommonTestUtils::TestsCommon {
protected:
    std::shared_ptr<ngraph::Function> fn_ptr;

    virtual void SetUp() {
        fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    }
};

TEST_F(RemoteBlob_Test, smoke_canInputUserBlob) {
#if defined(_WIN32) || defined(ANDROID)
    GTEST_SKIP();
#endif
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
    // TODO: Issue: investigate issue with IECore
    auto ie = InferenceEngine::Core();
    auto exec_net = ie.LoadNetwork(net, CommonTestUtils::DEVICE_GPU);

    // regular inference
    auto inf_req_regular = exec_net.CreateInferRequest();
    InferenceEngine::Blob::Ptr fakeImageData = FuncTestUtils::createAndFillBlob(
            net.getInputsInfo().begin()->second->getTensorDesc());
    inf_req_regular.SetBlob(net.getInputsInfo().begin()->first, fakeImageData);

    inf_req_regular.Infer();
    auto outputBlob_regular = inf_req_regular.GetBlob(net.getOutputsInfo().begin()->first);

    // inference using remote blob
    auto inf_req_shared = exec_net.CreateInferRequest();
    auto cldnn_context = exec_net.GetContext();
    cl_context ctx = std::dynamic_pointer_cast<ClContext>(cldnn_context)->get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    auto dims = net.getInputsInfo().begin()->second->getTensorDesc().getDims();
    size_t imSize = dims[1] * dims[2] * dims[3];

    cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imSize, NULL, &err);
    {
        void *buffer = fakeImageData->buffer();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
    }

    Blob::Ptr shared_blob = make_shared_blob(net.getInputsInfo().begin()->second->getTensorDesc(), cldnn_context,
                                             shared_buffer);
    inf_req_shared.SetBlob(net.getInputsInfo().begin()->first, shared_blob);

    inf_req_shared.Infer();
    auto outputBlob_shared = inf_req_shared.GetBlob(net.getOutputsInfo().begin()->first);

    // compare results
    {
        ASSERT_EQ(net.getOutputsInfo().begin()->second->getPrecision(), InferenceEngine::Precision::FP32);
        ASSERT_EQ(outputBlob_regular->size(), outputBlob_shared->size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        FuncTestUtils::compareBlobs(outputBlob_regular, outputBlob_shared, thr);
    }
}

TEST_F(RemoteBlob_Test, smoke_canInferOnUserContext) {
#if defined _WIN32
    GTEST_SKIP();
#endif
    auto fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());

    auto ie = PluginCache::get().ie();
    auto exec_net_regular = ie->LoadNetwork(net, CommonTestUtils::DEVICE_GPU);

    // regular inference
    auto inf_req_regular = exec_net_regular.CreateInferRequest();
    auto fakeImageData = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
    inf_req_regular.SetBlob(net.getInputsInfo().begin()->first, fakeImageData);

    inf_req_regular.Infer();
    auto outputBlob_regular = inf_req_regular.GetBlob(net.getOutputsInfo().begin()->first);

    // inference using remote blob
    auto ocl_instance = std::make_shared<OpenCL>();
    auto remote_context = make_shared_context(*ie, CommonTestUtils::DEVICE_GPU, ocl_instance->_context.get());
    auto exec_net_shared = ie->LoadNetwork(net, remote_context);
    auto inf_req_shared = exec_net_shared.CreateInferRequest();
    inf_req_shared.SetBlob(net.getInputsInfo().begin()->first, fakeImageData);

    inf_req_shared.Infer();
    auto outputBlob_shared = inf_req_shared.GetBlob(net.getOutputsInfo().begin()->first);

    // compare results
    {
        ASSERT_EQ(net.getOutputsInfo().begin()->second->getPrecision(), InferenceEngine::Precision::FP32);
        ASSERT_EQ(outputBlob_regular->size(), outputBlob_shared->size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        FuncTestUtils::compareBlobs(outputBlob_regular, outputBlob_shared, thr);
    }
}

class BatchedBlob_Test : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<size_t> {
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

TEST_P(BatchedBlob_Test, canInputNV12) {
#if defined(_WIN32) || defined(ANDROID)
    GTEST_SKIP();
#endif
    const int height = 16;
    const int width = 16;

    // ------------------------------------------------------
    // Prepare input data
    const InferenceEngine::TensorDesc y_plane_desc(InferenceEngine::Precision::U8, {1, 1, height, width},
        InferenceEngine::Layout::NHWC);
    const InferenceEngine::TensorDesc uv_plane_desc(InferenceEngine::Precision::U8, {1, 2, height / 2, width / 2},
        InferenceEngine::Layout::NHWC);
    std::vector<InferenceEngine::Blob::Ptr> fake_image_data_y;
    std::vector<InferenceEngine::Blob::Ptr> fake_image_data_uv;

    for (int i = 0; i < num_batch; i++) {
        fake_image_data_y.push_back(FuncTestUtils::createAndFillBlob(y_plane_desc, 50, 0, 1, i));
        fake_image_data_uv.push_back(FuncTestUtils::createAndFillBlob(uv_plane_desc, 256, 0, 1, i));
    }

    auto ie = InferenceEngine::Core();

    // ------------------------------------------------------
    // inference using remote blob with batch
    auto fn_ptr_remote = ngraph::builder::subgraph::makeConvPoolRelu({num_batch, 3, height, width});

    CNNNetwork net_remote(fn_ptr_remote);
    net_remote.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net_remote.getInputsInfo().begin()->second->setPrecision(Precision::U8);
    net_remote.getInputsInfo().begin()->second->getPreProcess().setColorFormat(ColorFormat::NV12);

    /* XXX: is it correct to set KEY_CLDNN_NV12_TWO_INPUTS in case of remote blob? */
    auto exec_net_b = ie.LoadNetwork(net_remote, CommonTestUtils::DEVICE_GPU,
                { { GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS, PluginConfigParams::YES} });
    auto inf_req_remote = exec_net_b.CreateInferRequest();
    auto cldnn_context = exec_net_b.GetContext();
    cl_context ctx = std::dynamic_pointer_cast<ClContext>(cldnn_context)->get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    std::vector<cl_mem> nv12_image_plane_y, nv12_image_plane_uv;
    std::vector<cl::Image2D> img_y, img_uv;
    std::vector<Blob::Ptr> blob_remote;

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
            true, origin, y_region, 0, 0, fake_image_data_y[i]->buffer(), 0, NULL, NULL);
        ASSERT_EQ(err, 0);

        err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv[i],
            true, origin, uv_region, 0, 0, fake_image_data_uv[i]->buffer(), 0, NULL, NULL);
        ASSERT_EQ(err, 0);

        img_y.push_back(cl::Image2D(nv12_image_plane_y[i]));
        img_uv.push_back(cl::Image2D(nv12_image_plane_uv[i]));

        blob_remote.push_back(make_shared_blob_nv12(cldnn_context, img_y[i], img_uv[i]));
    }

    if (num_batch == 1) {
        inf_req_remote.SetBlob(net_remote.getInputsInfo().begin()->first, blob_remote[0]);
    } else {
        auto batched_blob = make_shared_blob<BatchedBlob>(blob_remote);
        inf_req_remote.SetBlob(net_remote.getInputsInfo().begin()->first, batched_blob);
    }

    inf_req_remote.Infer();

    auto outputBlob_shared = inf_req_remote.GetBlob(net_remote.getOutputsInfo().begin()->first);

    // ------------------------------------------------------
    // Setup to inference using local blob with batch=1
    auto fn_ptr_local = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, height, width});

    CNNNetwork net_local(fn_ptr_local);

    net_local.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net_local.getInputsInfo().begin()->second->setPrecision(Precision::U8);
    net_local.getInputsInfo().begin()->second->getPreProcess().setColorFormat(ColorFormat::NV12);

    auto exec_net_b1 = ie.LoadNetwork(net_local, CommonTestUtils::DEVICE_GPU);

    auto inf_req_local = exec_net_b1.CreateInferRequest();

    // Run regular input for each image and compare against batched blob
    for (int i = 0; i < num_batch; i++) {
        auto y_blob = make_shared_blob<uint8_t>(y_plane_desc, fake_image_data_y[i]->buffer().as<uint8_t *>());
        auto uv_blob = make_shared_blob<uint8_t>(uv_plane_desc, fake_image_data_uv[i]->buffer().as<uint8_t *>());
        auto blob = make_shared_blob<NV12Blob>(y_blob, uv_blob);
        inf_req_local.SetBlob(net_local.getInputsInfo().begin()->first, blob);
        inf_req_local.Infer();
        auto output_blob_local = inf_req_local.GetBlob(net_local.getOutputsInfo().begin()->first);

        // This network generates [1, size] tensor whether batch=1 or 2. So need to split
        auto split_shared_blob = make_shared_blob<float_t>(output_blob_local->getTensorDesc(),
                                    outputBlob_shared->buffer().as<float_t *>() + output_blob_local->size() * i);
        ASSERT_EQ(output_blob_local->size(), split_shared_blob->size());
        float thr = 0.1;

        FuncTestUtils::compareBlobs(output_blob_local, split_shared_blob, thr, "", false);
    }
}

const std::vector<size_t> num_batches{1, 2, 4};

INSTANTIATE_TEST_CASE_P(smoke_RemoteBlob, BatchedBlob_Test, ::testing::ValuesIn(num_batches), BatchedBlob_Test::getTestCaseName);

class TwoNets_Test : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<size_t> {
    void SetUp() override {
        num_streams = this->GetParam();
        fn_ptrs = {ngraph::builder::subgraph::makeSplitMultiConvConcat(),
                   ngraph::builder::subgraph::makeMultiSingleConv()};
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::size_t> &obj) {
        return "num_streams_" + std::to_string(obj.param);
    }

protected:
    size_t num_streams;
    std::vector<std::shared_ptr<ngraph::Function>> fn_ptrs;
};

TEST_P(TwoNets_Test, canInferTwoExecNets) {
    std::vector<InferenceEngine::CNNNetwork> nets;
    for (auto &fn_ptr : fn_ptrs) {
        nets.push_back(CNNNetwork(fn_ptr));
    }

    auto ie = InferenceEngine::Core();

    std::vector<std::string> outputs;
    std::vector<InferRequest> irs;
    std::vector<std::vector<uint8_t>> ref;
    std::vector<int> outElementsCount;

    for (size_t i = 0; i < nets.size(); ++i) {
        auto net = nets[i];

        net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
        net.getInputsInfo().begin()->second->setPrecision(Precision::FP32);

        auto exec_net = ie.LoadNetwork(net, CommonTestUtils::DEVICE_GPU,
                                       {{PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS, std::to_string(num_streams)}});

        for (int j = 0; j < num_streams; j++) {
            outputs.push_back(net.getOutputsInfo().begin()->first);

            auto inf_req = exec_net.CreateInferRequest();
            irs.push_back(inf_req);

            auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
            inf_req.SetBlob(net.getInputsInfo().begin()->first, blob);

            outElementsCount.push_back(
                    std::accumulate(begin(fn_ptrs[i]->get_output_shape(0)), end(fn_ptrs[i]->get_output_shape(0)), 1,
                                    std::multiplies<size_t>()));
            const auto inBlob = inf_req.GetBlob(net.getInputsInfo().begin()->first);
            const auto blobSize = inBlob->byteSize();
            const auto inBlobBuf = inBlob->cbuffer().as<uint8_t *>();
            std::vector<uint8_t> inData(inBlobBuf, inBlobBuf + blobSize);
            auto reOutData = ngraph::helpers::interpreterFunction(fn_ptrs[i], {inData}).front().second;
            ref.push_back(reOutData);
        }
    }

    const int niter = 10;
    for (int i = 0; i < niter; i++) {
        for (auto ir : irs) {
            ir.StartAsync();
        }

        for (auto ir : irs) {
            ir.Wait(InferRequest::RESULT_READY);
        }
    }

    for (auto &net : nets) {
        ASSERT_EQ(net.getOutputsInfo().begin()->second->getPrecision(), InferenceEngine::Precision::FP32);
    }
    auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
    for (size_t i = 0; i < irs.size(); ++i) {
        const auto &refBuffer = ref[i].data();
        ASSERT_EQ(outElementsCount[i], irs[i].GetBlob(outputs[i])->size());
        FuncTestUtils::compareRawBuffers(irs[i].GetBlob(outputs[i])->buffer().as<float *>(),
                                         reinterpret_cast<const float *>(refBuffer), outElementsCount[i],
                                         outElementsCount[i],
                                         thr);
    }
}

const std::vector<size_t> num_streams{1, 2};

INSTANTIATE_TEST_CASE_P(smoke_RemoteBlob, TwoNets_Test, ::testing::ValuesIn(num_streams), TwoNets_Test::getTestCaseName);
