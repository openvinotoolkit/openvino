// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_layers_tests.hpp"
#include "vpu_case_params.hpp"
#include "common/include/vpu/utils/error.hpp"

#include "single_layer_common.hpp"

#include "gtest/gtest.h"

#include <string>
#include <ngraph_functions/subgraph_builders.hpp>
#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/blob_utils.hpp>
#include <vpu_case_common.hpp>

namespace {

class MyriadLayersTestsTensorIterator : public CommonTestUtils::TestsCommon {
public:
    void SetUp() override {
        fn_ptr = ngraph::builder::subgraph::makeTIwithLSTMcell();
    }
protected:
    std::shared_ptr<ngraph::Function> fn_ptr;
};

// TODO: Issue: 29485
TEST_F(MyriadLayersTestsTensorIterator, CompareNativeVersionWithUnrolledLoop) {
    DISABLE_IF(!CheckMyriadX () && !CheckMA2085());
    CNNNetwork network(fn_ptr);
    network.getInputsInfo().begin()->second->setPrecision(Precision::FP16);


    auto ie = PluginCache::get().ie();

    ExecutableNetwork exeNetworkWithConfig = ie->LoadNetwork(network, CommonTestUtils::DEVICE_MYRIAD,
                                                             {{VPU_CONFIG_KEY(FORCE_PURE_TENSOR_ITERATOR), CONFIG_VALUE(NO)},
                                                              {VPU_CONFIG_KEY(ENABLE_TENSOR_ITERATOR_UNROLLING), CONFIG_VALUE(YES)}});
    InferRequest inferRequestWithConfig = exeNetworkWithConfig.CreateInferRequest();
    auto blobWithConfig = FuncTestUtils::createAndFillBlob(network.getInputsInfo().begin()->second->getTensorDesc());
    inferRequestWithConfig.SetBlob(network.getInputsInfo().begin()->first, blobWithConfig);
    inferRequestWithConfig.Infer();
    auto* outRawDataWithConfig = inferRequestWithConfig.GetBlob(network.getOutputsInfo().begin()->first)->cbuffer().as<float*>();

    ExecutableNetwork exeNetworkWithoutConfig = ie->LoadNetwork(network, CommonTestUtils::DEVICE_MYRIAD,
                                                                {{VPU_CONFIG_KEY(FORCE_PURE_TENSOR_ITERATOR), CONFIG_VALUE(YES)},
                                                                 {VPU_CONFIG_KEY(ENABLE_TENSOR_ITERATOR_UNROLLING), CONFIG_VALUE(NO)}});
    InferRequest inferRequestWithoutConfig = exeNetworkWithoutConfig.CreateInferRequest();
    auto blobWithoutConfig = FuncTestUtils::createAndFillBlob(network.getInputsInfo().begin()->second->getTensorDesc());
    inferRequestWithoutConfig.SetBlob(network.getInputsInfo().begin()->first, blobWithoutConfig);
    inferRequestWithoutConfig.Infer();
    auto* outRawDataWithoutConfig = inferRequestWithoutConfig.GetBlob(network.getOutputsInfo().begin()->first)->cbuffer().as<float*>();

    auto thr = FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP16);
    size_t outElementsCount = std::accumulate(begin(fn_ptr->get_output_shape(0)), end(fn_ptr->get_output_shape(0)), 1,
                                              std::multiplies<size_t>());

    FuncTestUtils::compareRawBuffers(outRawDataWithoutConfig, outRawDataWithConfig, outElementsCount,
                                     outElementsCount,
                                     thr);

}
}
