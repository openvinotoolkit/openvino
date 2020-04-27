// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include <inference_engine.hpp>
#include <ie_compound_blob.h>

#include <cldnn/cldnn_config.hpp>

#ifdef _WIN32
# include <gpu/gpu_context_api_dx.hpp>
#elif defined ENABLE_LIBVA
# include <gpu/gpu_context_api_va.hpp>
#endif
#include <gpu/gpu_context_api_ocl.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/plugin_cache.hpp>

#include "ngraph_functions/subgraph_builders.hpp"
#include "functional_test_utils/blob_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::gpu;

struct OpenCL {
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;

    explicit OpenCL(std::shared_ptr<std::vector<cl_context_properties>> media_api_context_properties = nullptr) {
        // get Intel iGPU OCL device, create context and queue
        {
            const unsigned int refVendorID = 0x8086;
            cl_uint n = 0;
            cl_int err = clGetPlatformIDs(0, NULL, &n);

            // Get platform list
            std::vector<cl_platform_id> platform_ids(n);
            err = clGetPlatformIDs(n, platform_ids.data(), NULL);

            for (auto& id : platform_ids) {
                cl::Platform platform = cl::Platform(id);
                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    if (refVendorID == d.getInfo<CL_DEVICE_VENDOR_ID>()) {
                        _device = d;
                        _context = cl::Context(_device);
                        break;
                    }
                }
            }
            cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            _queue = cl::CommandQueue(_context, _device, props);
        }
    }

    explicit OpenCL(cl_context context) {
        // user-supplied context handle
        _context = cl::Context(context, true);
        _device = cl::Device(_context.getInfo<CL_CONTEXT_DEVICES>()[0].get(), true);

        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        _queue = cl::CommandQueue(_context, _device, props);
    }
};

class RemoteBlob_Test : public CommonTestUtils::TestsCommon {
protected:
    std::shared_ptr<ngraph::Function> fn_ptr;
    virtual void SetUp() {
        fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    }
};

TEST_F(RemoteBlob_Test, canInputUserBlob) {
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
    InferenceEngine::Blob::Ptr fakeImageData = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
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
        void* buffer = fakeImageData->buffer();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
    }

    Blob::Ptr shared_blob = make_shared_blob(net.getInputsInfo().begin()->second->getTensorDesc(), cldnn_context, shared_buffer);
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

TEST_F(RemoteBlob_Test, canInferOnUserContext) {
#if defined _WIN32
    GTEST_SKIP();
#endif
    auto fn_ptr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
    CNNNetwork net(fn_ptr);

    net.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net.getInputsInfo().begin()->second->setPrecision(Precision::U8);

    auto blob = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());

    auto ie = InferenceEngine::Core();
    auto exec_net_regular = ie.LoadNetwork(net, CommonTestUtils::DEVICE_GPU);

    // regular inference
    auto inf_req_regular = exec_net_regular.CreateInferRequest();
    auto fakeImageData = FuncTestUtils::createAndFillBlob(net.getInputsInfo().begin()->second->getTensorDesc());
    inf_req_regular.SetBlob(net.getInputsInfo().begin()->first, fakeImageData);

    inf_req_regular.Infer();
    auto outputBlob_regular = inf_req_regular.GetBlob(net.getOutputsInfo().begin()->first);

    // inference using remote blob
    auto ocl_instance = std::make_shared<OpenCL>();
    auto remote_context = make_shared_context(ie, CommonTestUtils::DEVICE_GPU, ocl_instance->_context.get());
    auto exec_net_shared = ie.LoadNetwork(net, remote_context);
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

class TwoNets_Test : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<size_t> {
    void SetUp() override {
        num_streams = this->GetParam();
        fn_ptrs = {ngraph::builder::subgraph::makeSplitMultiConvConcat(),
                   ngraph::builder::subgraph::makeMultiSingleConv()};
    };
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::size_t> & obj) {
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
    std::vector<std::shared_ptr<float*>> ref;
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

            outElementsCount.push_back(std::accumulate(begin(fn_ptrs[i]->get_output_shape(0)), end(fn_ptrs[i]->get_output_shape(0)), 1,
                                                       std::multiplies<size_t>()));

            std::shared_ptr<float*> reOutData = ngraph::helpers::inferFnWithInterp<ngraph::element::Type_t::f32>(
                    fn_ptrs[i], {inf_req.GetBlob(net.getInputsInfo().begin()->first)->buffer()}).front();
            ref.push_back(reOutData);
        }
    }

    const int niter = 10;
    for (int i = 0; i < niter; i++) {
        for (auto ir : irs) {
            ir.StartAsync();
        }

        for (auto ir : irs) {
            ir.Wait(IInferRequest::RESULT_READY);
        }
    }

    for (auto& net : nets) {
        ASSERT_EQ(net.getOutputsInfo().begin()->second->getPrecision(), InferenceEngine::Precision::FP32);
    }
    auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32);
    for (size_t i = 0; i < irs.size(); ++i) {
        ASSERT_EQ(outElementsCount[i], irs[i].GetBlob(outputs[i])->size());
        FuncTestUtils::compareRawBuffers(irs[i].GetBlob(outputs[i])->buffer().as<float*>(), *ref[i], outElementsCount[i],
                                         outElementsCount[i],
                                         thr);
    }
}

const std::vector<size_t> num_strems{1, 2};

INSTANTIATE_TEST_CASE_P(RemoteBlob, TwoNets_Test, ::testing::ValuesIn(num_strems), TwoNets_Test::getTestCaseName);
