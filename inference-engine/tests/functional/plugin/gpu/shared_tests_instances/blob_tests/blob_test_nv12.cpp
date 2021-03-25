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
namespace nv12 {
std::vector<ngraph::element::Type> inputTypes = {
    ngraph::element::f32,
    ngraph::element::f16
};

std::vector<std::pair<nghraphSubgraphFuncType, std::vector<size_t>>> funtions = {
    {ngraph::builder::subgraph::makeSingleConv, {1, 3, 24, 24}}
};

std::vector<size_t> inferCounts = {
    1,
    4
};

std::vector<size_t> batchSizes = {
    1
};

std::vector<bool> asyncExec = {
    true,
    false
};

std::map<std::string, std::string> dyn_batch_config = {
    {InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED, InferenceEngine::PluginConfigParams::YES}
};

void PreprocessNetwork(InferenceEngine::CNNNetwork& network) {
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;

    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);
    // set input resize algorithm to enable input autoresize
    input_info->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    // set input color format to ColorFormat::NV12 to enable automatic input color format
    // pre-processing
    input_info->getPreProcess().setColorFormat(ColorFormat::NV12);
}

static std::vector<InferenceEngine::Blob::Ptr> blobs;

InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) {
    size_t seed = 0;

    auto td = info.getTensorDesc();
    auto width = td.getDims()[2];
    auto height = td.getDims()[3];

    blobs.push_back(make_blob_with_precision(td));
    InferenceEngine::Blob::Ptr blob = blobs.back();
    blob->allocate();
    auto buffer = blob->buffer();
    auto ptr = buffer.as<uint8_t*>();

    for (int i= 0; i < blob->byteSize(); i++) {
        ptr[i] = seed % 64 + 96;
        seed++;
    }

    // Create tensor descriptors for Y and UV blobs
    const InferenceEngine::TensorDesc y_plane_desc(InferenceEngine::Precision::U8, {1, 1, height, width},
        InferenceEngine::Layout::NHWC);
    const InferenceEngine::TensorDesc uv_plane_desc(InferenceEngine::Precision::U8, {1, 2, height / 2, width / 2},
        InferenceEngine::Layout::NHWC);
    const size_t offset = width * height;

    for (int i = offset; i < blob->byteSize(); i++) {
        ptr[i] = 128;
    }
    // --------------------------- Create a blob to hold the NV12 input data -------------------------------
    // Create blob for Y plane from raw data
    Blob::Ptr y_blob = make_shared_blob<uint8_t>(y_plane_desc, ptr);
    // Create blob for UV plane from raw data
    Blob::Ptr uv_blob = make_shared_blob<uint8_t>(uv_plane_desc, ptr + offset);
    // Create NV12Blob from Y and UV blobs
    auto res = make_shared_blob<NV12Blob>(y_blob, uv_blob);

    return res;
}

std::vector<std::vector<uint8_t>> generateReference(InferenceEngine::CNNNetwork& cnnNetwork, std::vector<Blob::Ptr>& inputs) {
    InferenceEngine::Core core;
    auto cpu_network = core.LoadNetwork(cnnNetwork, "CPU");
    auto inferRequest = cpu_network.CreateInferRequest();
    const auto& cpu_inputsInfo = cpu_network.GetInputsInfo();
    int i = 0;
    for (const auto& info : cpu_inputsInfo) {
        auto blob = inputs[i];
        inferRequest.SetBlob(info.second->name(), blob);
        i++;
    }

    inferRequest.Infer();

    auto cpu_outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    for (const auto &output : cpu_network.GetOutputsInfo()) {
        const auto &name = output.first;
        cpu_outputs.push_back(inferRequest.GetBlob(name));
    }

    std::vector<std::vector<uint8_t>> res;
    for (auto& output : cpu_outputs) {
        auto buffer = output->buffer().as<uint8_t*>();
        res.push_back(std::vector<uint8_t>(buffer, buffer + output->byteSize()));
    }

    return res;
}

void Teardown() {
    blobs.clear();
}

std::tuple<networkPreprocessFuncType,
    generateInputFuncType,
    generateReferenceFuncType,
    makeTestBlobFuncType,
    teardownFuncType> fnSet = {PreprocessNetwork, GenerateInput, generateReference, nullptr, Teardown};
} // namespace nv12

INSTANTIATE_TEST_CASE_P(nv12, BlobTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(nv12::inputTypes),
        ::testing::ValuesIn(nv12::funtions),
        ::testing::Values(nv12::fnSet),
        ::testing::ValuesIn(nv12::inferCounts),
        ::testing::ValuesIn(nv12::batchSizes),
        ::testing::ValuesIn(nv12::asyncExec),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values(std::map<std::string, std::string>())),
    BlobTestBase::getTestCaseName);

INSTANTIATE_TEST_CASE_P(nv12_dynamic_batch, BlobTestBase,
    ::testing::Combine(
        ::testing::ValuesIn(nv12::inputTypes),
        ::testing::ValuesIn(nv12::funtions),
        ::testing::Values(nv12::fnSet),
        ::testing::ValuesIn(nv12::inferCounts),
        ::testing::ValuesIn(nv12::batchSizes),
        ::testing::ValuesIn(nv12::asyncExec),
        ::testing::Values(CommonTestUtils::DEVICE_GPU),
        ::testing::Values(nv12::dyn_batch_config)),
    BlobTestBase::getTestCaseName);
} // namespace BlobTestsDefinitions
