// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <thread>

#include <ie_compound_blob.h>

#include <gpu/gpu_config.hpp>
#include <remote_blob_tests/remote_blob_helpers.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/plugin_cache.hpp>

#include "base/ov_behavior_test_utils.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::gpu;

class RemoteBlob_Test : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<bool> {
protected:
    std::shared_ptr<ngraph::Function> fn_ptr;
    std::string deviceName;
    std::map<std::string, std::string> config;

public:
    void SetUp() override {
        deviceName = CommonTestUtils::DEVICE_GPU;
        auto with_auto_batching = this->GetParam();
        if (with_auto_batching) { // BATCH:GPU
            config =
                    {{CONFIG_KEY(PERFORMANCE_HINT) , CONFIG_VALUE(THROUGHPUT)},
                            // immediate timeout to avoid increasing the test time
                            {CONFIG_KEY(AUTO_BATCH_TIMEOUT) , "0"},
                            };
            }
        fn_ptr = ov::test::behavior::getDefaultNGraphFunctionForTheDevice(with_auto_batching ? CommonTestUtils::DEVICE_BATCH : deviceName);
    }
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        auto with_auto_batch = obj.param;
        return std::string("RemoteBlob_Test") + (with_auto_batch ? "_WITH_AUTO_BATCHING": "");
    }
};

TEST_P(RemoteBlob_Test, smoke_canInputUserBlob) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    // TODO: Issue: investigate issue with IECore
    auto ie = InferenceEngine::Core();
    auto exec_net = ie.LoadNetwork(net, deviceName, config);

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

    auto desc = net.getInputsInfo().begin()->second->getTensorDesc();
    size_t imSize = std::accumulate(desc.getDims().begin(), desc.getDims().end(),
                                    desc.getPrecision().size(),
                                    std::multiplies<size_t>());

    cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imSize, NULL, &err);
    {
        void *buffer = fakeImageData->buffer();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
    }

    Blob::Ptr shared_blob = make_shared_blob(net.getInputsInfo().begin()->second->getTensorDesc(), cldnn_context,
                                             shared_buffer);
    shared_blob->allocate();
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

TEST_P(RemoteBlob_Test, smoke_canUseRemoteBlobSimultaneously) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    const int batch = 2;
    const int channels = 3;
    const int height = 512;
    const int width = 512;
    const size_t img_size = batch * channels * height * width;
    cl_int err;

    const InferenceEngine::TensorDesc tensor_desc{InferenceEngine::Precision::U8,
                                                  {batch, channels, height, width},
                                                  InferenceEngine::Layout::NHWC};

    InferenceEngine::Blob::Ptr ref_blob = FuncTestUtils::createAndFillBlob(tensor_desc);

    auto ie = PluginCache::get().ie();
    auto ocl_instance = std::make_shared<OpenCL>();
    ocl_instance->_queue = cl::CommandQueue(ocl_instance->_context, ocl_instance->_device);

    // Allocate OpenCL buffer for data
    cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, img_size, NULL, &err);

    // Create shared context
    auto remote_context = make_shared_context(*ie, deviceName, ocl_instance->_queue.get());

    // Wrap buffer above with IE blob
    Blob::Ptr shared_blob = make_shared_blob(tensor_desc, remote_context, shared_buffer);
    // Allocate is needed to actually trigger memory handle sharing. For other buffers it's called inside SetBlob impl
    // TODO: Why do we need to call it explicitly? Consider doing it internally
    shared_blob->allocate();

    // Copy data from ordinary blob to OpenCL buffer
    {
        void* buffer = ref_blob->buffer();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, img_size, buffer);
    }

    // Lock remote buffer in multiple threads and compare data with ordinary one
    const int threads_num = 8;
    std::vector<std::thread> threads;
    for (int i = 0; i < threads_num; i++) {
        threads.emplace_back(std::thread{[&] {
            auto ref_blob_buf = ref_blob->cbuffer();
            auto ref_blob_ptr = ref_blob_buf.as<const char*>();
            auto remote_blob_buf = shared_blob->cbuffer();
            auto remote_blob_ptr = remote_blob_buf.as<const char*>();
            ASSERT_EQ(ref_blob->size(), shared_blob->size());
            for (size_t j = 0; j < ref_blob->size(); j++) {
                ASSERT_EQ(ref_blob_ptr[j], remote_blob_ptr[j]);
            }
        }});
    }

    for (auto& t : threads)
        t.join();
}

TEST_P(RemoteBlob_Test, smoke_canInputPluginRemoteBlob) {
#if defined(ANDROID)
    GTEST_SKIP();
#endif
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    // TODO: Issue: investigate issue with IECore
    auto ie = InferenceEngine::Core();
    auto exec_net = ie.LoadNetwork(net, deviceName, config);

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

    auto desc = net.getInputsInfo().begin()->second->getTensorDesc();
    size_t imSize = std::accumulate(desc.getDims().begin(), desc.getDims().end(),
                                    desc.getPrecision().size(),
                                    std::multiplies<size_t>());

    Blob::Ptr shared_blob = make_shared_blob(net.getInputsInfo().begin()->second->getTensorDesc(), cldnn_context);
    shared_blob->allocate();
    {
        cl::Buffer shared_buffer = *shared_blob->as<gpu::ClBufferBlob>();
        void *buffer = fakeImageData->buffer();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
    }

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


TEST_P(RemoteBlob_Test, smoke_canInferOnUserContext) {
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());

    auto ie = PluginCache::get().ie();
    auto exec_net_regular = ie->LoadNetwork(net, deviceName);

    // regular inference
    auto inf_req_regular = exec_net_regular.CreateInferRequest();
    auto fakeImageData = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
    inf_req_regular.SetBlob(net.getInputsInfo().begin()->first, fakeImageData);

    inf_req_regular.Infer();
    auto outputBlob_regular = inf_req_regular.GetBlob(net.getOutputsInfo().begin()->first);

    // inference using remote blob
    auto ocl_instance = std::make_shared<OpenCL>();
    auto remote_context = make_shared_context(*ie, deviceName, ocl_instance->_context.get());
    // since there is no way to enable the Auto-Batching thru the device name when loading with the RemoteContext
    // (as the device name is deduced from the context, which is the "GPU")
    // the only-way to test the auto-batching is explicit config with perf hint set to THROUGHPUT
    auto exec_net_shared = ie->LoadNetwork(net, remote_context, config);
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

TEST_P(RemoteBlob_Test, smoke_canInferOnUserQueue_out_of_order) {
#if defined _WIN32
    GTEST_SKIP();
#endif

    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());

    auto ie = PluginCache::get().ie();
    auto exec_net_regular = ie->LoadNetwork(net, deviceName);

    // regular inference
    auto inf_req_regular = exec_net_regular.CreateInferRequest();
    auto fakeImageData = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
    inf_req_regular.SetBlob(net.getInputsInfo().begin()->first, fakeImageData);

    inf_req_regular.Infer();
    auto outputBlob_regular = inf_req_regular.GetBlob(net.getOutputsInfo().begin()->first);

    // inference using remote blob
    auto ocl_instance = std::make_shared<OpenCL>();
    cl_int err;

    auto in_desc = net.getInputsInfo().begin()->second->getTensorDesc();
    auto out_desc = net.getOutputsInfo().begin()->second->getTensorDesc();
    auto in_dims = in_desc.getDims();
    auto out_dims = out_desc.getDims();
    size_t in_size = std::accumulate(in_desc.getDims().begin(), in_desc.getDims().end(),
                                     in_desc.getPrecision().size(),
                                     std::multiplies<size_t>());
    size_t out_size = std::accumulate(out_desc.getDims().begin(), out_desc.getDims().end(),
                                      out_desc.getPrecision().size(),
                                      std::multiplies<size_t>());

    // In this scenario we create shared OCL queue and run simple pre-process action and post-process action (buffer copies in both cases)
    // without calling thread blocks
    auto remote_context = make_shared_context(*ie, deviceName, ocl_instance->_queue.get());
    auto exec_net_shared = ie->LoadNetwork(net, remote_context); // no auto-batching support, so no config is passed
    auto inf_req_shared = exec_net_shared.CreateInferRequest();

    // Allocate shared buffers for input and output data which will be set to infer request
    cl::Buffer shared_input_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size, NULL, &err);
    cl::Buffer shared_output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);
    // Allocate output buffer where inference result will be put as a post-processing step
    cl::Buffer output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);

    // Wrap buffers above with IE blobs
    Blob::Ptr shared_input_blob = make_shared_blob(in_desc, remote_context, shared_input_buffer);
    Blob::Ptr shared_output_blob = make_shared_blob(out_desc, remote_context, shared_output_buffer);
    Blob::Ptr output_blob = make_shared_blob(out_desc, remote_context, output_buffer);
    // Allocate is needed to actually trigger memory handle sharing. For other buffers it's called inside SetBlob impl
    // TODO: Why do we need to call it explicitly? Consider doing it internally
    output_blob->allocate();

    // Pass shared blobs to infer request
    inf_req_shared.SetBlob(net.getInputsInfo().begin()->first, shared_input_blob);
    inf_req_shared.SetBlob(net.getOutputsInfo().begin()->first, shared_output_blob);

    // 1. Pre-processing. Enqueue non-blocking copy from host ptr to shared device input buffer and barrier to ensure that copy is finished before
    // inference primitives starts execution
    {
        void *buffer = fakeImageData->buffer();
        ocl_instance->_queue.enqueueWriteBuffer(shared_input_buffer, false, 0, in_size, buffer);
        ocl_instance->_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
    }

    // 2. Enqueue inference primitives. With shared queue this call ensures that all kernels are scheduled to the corresponding queue
    // before giving the control back
    inf_req_shared.StartAsync();

    // 3. Post-processing. Enqueue copy from shared blob with inference result to another output blob
    // Enqueue barrier with empty wait list is needed to ensure that previous kernels are finished before copying the data. It's needed here since we
    // create OOO queue.
    // Note: inf_req_shared.Wait() can be dropped in some cases, but if plugin-side post-processing is required,
    // then the result may be incorrect without Wait().
    {
        ocl_instance->_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
        ocl_instance->_queue.enqueueCopyBuffer(shared_output_buffer, output_buffer, 0, 0, output_blob->byteSize());
    }

    // 4. Wait for infer request and post-processing completion
    ocl_instance->_queue.finish();

    // compare results
    {
        ASSERT_EQ(net.getOutputsInfo().begin()->second->getPrecision(), InferenceEngine::Precision::FP32);
        ASSERT_EQ(outputBlob_regular->size(), output_blob->size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        FuncTestUtils::compareBlobs(outputBlob_regular, output_blob, thr);
    }
}

TEST_P(RemoteBlob_Test, smoke_canInferOnUserQueue_in_order) {
#if defined _WIN32
    GTEST_SKIP();
#endif

    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());

    auto ie = PluginCache::get().ie();
    auto exec_net_regular = ie->LoadNetwork(net, deviceName);

    // regular inference
    auto inf_req_regular = exec_net_regular.CreateInferRequest();
    auto fakeImageData = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
    inf_req_regular.SetBlob(net.getInputsInfo().begin()->first, fakeImageData);

    inf_req_regular.Infer();
    auto outputBlob_regular = inf_req_regular.GetBlob(net.getOutputsInfo().begin()->first);

    // inference using remote blob
    auto ocl_instance = std::make_shared<OpenCL>();
    ocl_instance->_queue = cl::CommandQueue(ocl_instance->_context, ocl_instance->_device);
    cl_int err;

    auto in_desc = net.getInputsInfo().begin()->second->getTensorDesc();
    auto out_desc = net.getOutputsInfo().begin()->second->getTensorDesc();
    auto in_dims = in_desc.getDims();
    auto out_dims = out_desc.getDims();
    size_t in_size = std::accumulate(in_desc.getDims().begin(), in_desc.getDims().end(),
                                     in_desc.getPrecision().size(),
                                     std::multiplies<size_t>());
    size_t out_size = std::accumulate(out_desc.getDims().begin(), out_desc.getDims().end(),
                                      out_desc.getPrecision().size(),
                                      std::multiplies<size_t>());

    // In this scenario we create shared OCL queue and run simple pre-process action and post-process action (buffer copies in both cases)
    // without calling thread blocks
    auto remote_context = make_shared_context(*ie, deviceName, ocl_instance->_queue.get());
    auto exec_net_shared = ie->LoadNetwork(net, remote_context); // no auto-batching support, so no config is passed
    auto inf_req_shared = exec_net_shared.CreateInferRequest();

    // Allocate shared buffers for input and output data which will be set to infer request
    cl::Buffer shared_input_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, in_size, NULL, &err);
    cl::Buffer shared_output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);
    // Allocate output buffer where inference result will be put as a post-processing step
    cl::Buffer output_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, out_size, NULL, &err);

    // Wrap buffers above with IE blobs
    Blob::Ptr shared_input_blob = make_shared_blob(in_desc, remote_context, shared_input_buffer);
    Blob::Ptr shared_output_blob = make_shared_blob(out_desc, remote_context, shared_output_buffer);
    Blob::Ptr output_blob = make_shared_blob(out_desc, remote_context, output_buffer);
    // Allocate is needed to actually trigger memory handle sharing. For other buffers it's called inside SetBlob impl
    // TODO: Why do we need to call it explicitly? Consider doing it internally
    output_blob->allocate();

    // Pass shared blobs to infer request
    inf_req_shared.SetBlob(net.getInputsInfo().begin()->first, shared_input_blob);
    inf_req_shared.SetBlob(net.getOutputsInfo().begin()->first, shared_output_blob);

    // 1. Pre-processing. Enqueue non-blocking copy from host ptr to shared device input buffer
    {
        void *buffer = fakeImageData->buffer();
        ocl_instance->_queue.enqueueWriteBuffer(shared_input_buffer, false, 0, in_size, buffer);
    }

    // 2. Enqueue inference primitives. With shared queue this call ensures that all kernels are scheduled to the corresponding queue
    // before giving the control back
    inf_req_shared.StartAsync();

    // 3. Post-processing. Enqueue copy from shared blob with inference result to another output blob
    // Note: inf_req_shared.Wait() can be dropped in some cases, but if plugin-side post-processing is required,
    // then the result may be incorrect without Wait().
    {
        ocl_instance->_queue.enqueueCopyBuffer(shared_output_buffer, output_buffer, 0, 0, output_blob->byteSize());
    }

    // 4. Wait for infer request and post-processing completion
    ocl_instance->_queue.finish();

    // compare results
    {
        ASSERT_EQ(net.getOutputsInfo().begin()->second->getPrecision(), InferenceEngine::Precision::FP32);
        ASSERT_EQ(outputBlob_regular->size(), output_blob->size());
        auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
        FuncTestUtils::compareBlobs(outputBlob_regular, output_blob, thr);
    }
}

std::vector<bool> with_auto_batching {true, false};
INSTANTIATE_TEST_SUITE_P(smoke_RemoteBlob, RemoteBlob_Test, ::testing::ValuesIn(with_auto_batching),
        RemoteBlob_Test::getTestCaseName);

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
#if defined(ANDROID)
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

INSTANTIATE_TEST_SUITE_P(smoke_RemoteBlob, BatchedBlob_Test, ::testing::ValuesIn(num_batches), BatchedBlob_Test::getTestCaseName);

using TwoNetsParams = std::tuple<size_t,   // number of streams
                                 size_t>;  // number of requests

class TwoNets_Test : public CommonTestUtils::TestsCommon,
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

        for (int j = 0; j < num_streams * num_requests; j++) {
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

const std::vector<size_t> num_streams{ 1, 2 };
const std::vector<size_t> num_requests{ 1, 4 };

INSTANTIATE_TEST_SUITE_P(smoke_RemoteBlob, TwoNets_Test,
    ::testing::Combine(::testing::ValuesIn(num_streams),
        ::testing::ValuesIn(num_requests)),
    TwoNets_Test::getTestCaseName);
