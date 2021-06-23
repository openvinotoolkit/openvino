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

class IEBatchedBlob_Test : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<size_t> {
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

TEST_P(IEBatchedBlob_Test, canInputNV12) {
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
    // Setup network to inference using multiple batch_size != 1
    // reduce network sizes to force ie preprocessing
    const int convolution_height = height / 1.5f;
    const int convolution_width = width / 1.5f;
    auto fn_ptr_mbatch = ngraph::builder::subgraph::makeConvPoolRelu({num_batch, 3, convolution_height, convolution_width});

    CNNNetwork net_mbatch(fn_ptr_mbatch);
    // prepare input descriptions
    net_mbatch.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net_mbatch.getInputsInfo().begin()->second->setPrecision(Precision::U8);
    net_mbatch.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    net_mbatch.getInputsInfo().begin()->second->getPreProcess().setColorFormat(ColorFormat::NV12);

    // ------------------------------------------------------
    // prepare output descriptions
    DataPtr output_info = net_mbatch.getOutputsInfo().begin()->second;
    std::string output_name = net_mbatch.getOutputsInfo().begin()->first;
    output_info->setPrecision(Precision::FP32);

    auto exec_net_mbatch = ie.LoadNetwork(net_mbatch, CommonTestUtils::DEVICE_GPU);
    auto inf_req_mbatch = exec_net_mbatch.CreateInferRequest();

    // ------------------------------------------------------
    // prepare input batched blob
    std::vector<Blob::Ptr> full_blob;
    for (int i = 0; i < num_batch; i++) {
        full_blob.push_back(make_shared_blob<NV12Blob>(fake_image_data_y[i], fake_image_data_uv[i]));
    }

    // ------------------------------------------------------
    // execute BatchedBlob inference
    if (num_batch == 1) {
        inf_req_mbatch.SetBlob(net_mbatch.getInputsInfo().begin()->first, full_blob[0]);
    } else {
        auto batched_blob = make_shared_blob<BatchedBlob>(full_blob);
        inf_req_mbatch.SetBlob(net_mbatch.getInputsInfo().begin()->first, batched_blob);
    }

    inf_req_mbatch.Infer();

    auto outputBlob_shared = inf_req_mbatch.GetBlob(net_mbatch.getOutputsInfo().begin()->first);

    // ------------------------------------------------------
    // Setup another network to inference using batch_size = 1
    auto fn_ptr_local_sbatch = ngraph::builder::subgraph::makeConvPoolRelu({1, 3, convolution_height, convolution_width});

    CNNNetwork net_sbatch(fn_ptr_local_sbatch);

    net_sbatch.getInputsInfo().begin()->second->setLayout(Layout::NCHW);
    net_sbatch.getInputsInfo().begin()->second->setPrecision(Precision::U8);
    net_sbatch.getInputsInfo().begin()->second->getPreProcess().setColorFormat(ColorFormat::NV12);
    net_sbatch.getInputsInfo().begin()->second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
    net_sbatch.getOutputsInfo().begin()->second->setPrecision(Precision::FP32);

    auto exec_net_sbatch = ie.LoadNetwork(net_sbatch, CommonTestUtils::DEVICE_GPU);

    auto inf_req_sbatch = exec_net_sbatch.CreateInferRequest();

    // ------------------------------------------------------
    // Run regular input for each image and compare against batched blob
    for (int i = 0; i < num_batch; i++) {
        auto y_blob = make_shared_blob<uint8_t>(y_plane_desc, fake_image_data_y[i]->buffer().as<uint8_t *>());
        auto uv_blob = make_shared_blob<uint8_t>(uv_plane_desc, fake_image_data_uv[i]->buffer().as<uint8_t *>());
        auto blob = make_shared_blob<NV12Blob>(y_blob, uv_blob);
        inf_req_sbatch.SetBlob(net_sbatch.getInputsInfo().begin()->first, blob);
        inf_req_sbatch.Infer();
        auto output_blob_local = inf_req_sbatch.GetBlob(net_sbatch.getOutputsInfo().begin()->first);

        // ------------------------------------------------------
        // This network generates [1, size] tensor whether batch=1 or 2. So need to split
        auto split_shared_blob = make_shared_blob<float_t>(output_blob_local->getTensorDesc(),
                                    outputBlob_shared->buffer().as<float_t *>() + output_blob_local->size() * i);
        ASSERT_EQ(output_blob_local->size(), split_shared_blob->size());
        float thr = 0.1;

        FuncTestUtils::compareBlobs(output_blob_local, split_shared_blob, thr, "", false);
    }
}

const std::vector<size_t> num_batches{1, 2, 4, 5, 6};

INSTANTIATE_TEST_CASE_P(smoke_PreprocessBatchedBlob, IEBatchedBlob_Test, ::testing::ValuesIn(num_batches), IEBatchedBlob_Test::getTestCaseName);
