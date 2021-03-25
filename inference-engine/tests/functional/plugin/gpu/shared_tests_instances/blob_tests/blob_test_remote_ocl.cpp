// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <blob_tests/blob_test_base.hpp>
#include <ie_blob.h>
#include <ngraph_functions/subgraph_builders.hpp>
#include <remote_blob_tests/remote_blob_helpers.hpp>
#include <ie_remote_context.hpp>

using namespace InferenceEngine;

namespace BlobTestsDefinitions {
namespace RemoteOCL {
std::vector<ngraph::element::Type> inputTypes = {
    ngraph::element::f16,
    ngraph::element::f32,
};

std::vector<std::pair<nghraphSubgraphFuncType, std::vector<size_t>>> funtions = {
    {ngraph::builder::subgraph::makeSingleConv, {1, 3, 24, 24}},
    {ngraph::builder::subgraph::makeSplitConvConcat, {1, 4, 20, 20}},
    {ngraph::builder::subgraph::make2InputSubtract, {1, 3, 24, 24}}
};

std::vector<size_t> inferCounts = {
    1,
    4
};

std::vector<size_t> batchSizes = {
    1,
    7,
    8
};

std::vector<bool> asyncExec = {
    true,
    false
};

std::map<std::string, std::string> dyn_batch_config = {
    {InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}
};

static std::map<Blob*, cl::Buffer> bufferMap;

void Teardown() {
    bufferMap.clear();
}

InferenceEngine::Blob::Ptr MakeTestBlobOCL(InferenceEngine::Blob::Ptr inputBlob, InferenceEngine::ExecutableNetwork& execNet) {
    // inference using remote blob
    auto cldnn_context = execNet.GetContext();
    cl_context ctx = std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(cldnn_context)->get();
    auto ocl_instance = std::make_shared<OpenCL>(ctx);
    cl_int err;

    size_t imSize = inputBlob->byteSize();

    auto shared_buffer = cl::Buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imSize, NULL, &err);
    {
        void *buffer = inputBlob->buffer();
        ocl_instance->_queue.enqueueWriteBuffer(shared_buffer, true, 0, imSize, buffer);
    }
    InferenceEngine::Blob::Ptr shared_blob = InferenceEngine::gpu::make_shared_blob(inputBlob->getTensorDesc(), cldnn_context, shared_buffer);

    bufferMap[&(*shared_blob)] = shared_buffer;
    return shared_blob;
}

std::tuple<networkPreprocessFuncType,
           generateInputFuncType,
           generateReferenceFuncType,
           makeTestBlobFuncType,
           teardownFuncType> fnSet = {nullptr, nullptr, nullptr, MakeTestBlobOCL, Teardown};
} // namespace RemoteOCL

INSTANTIATE_TEST_CASE_P(smoke_Remote_OCL, BlobTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(RemoteOCL::inputTypes),
        ::testing::ValuesIn(RemoteOCL::funtions),
        ::testing::Values(RemoteOCL::fnSet),
        ::testing::ValuesIn(RemoteOCL::inferCounts),
        ::testing::ValuesIn(RemoteOCL::batchSizes),
        ::testing::ValuesIn(RemoteOCL::asyncExec),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values(std::map<std::string, std::string>())),
    BlobTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Remote_OCL_dynamic_batch, BlobTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(RemoteOCL::inputTypes),
        ::testing::ValuesIn(RemoteOCL::funtions),
        ::testing::Values(RemoteOCL::fnSet),
        ::testing::ValuesIn(RemoteOCL::inferCounts),
        ::testing::ValuesIn(RemoteOCL::batchSizes),
        ::testing::ValuesIn(RemoteOCL::asyncExec),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values(RemoteOCL::dyn_batch_config)),
    BlobTestBase::getTestCaseName);
} // namespace BlobTestsDefinitions
