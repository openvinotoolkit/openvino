// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph_functions/subgraph_builders.hpp>
#include <functional_test_utils/blob_utils.hpp>
#include "myriad_hw_tests_base.hpp"

using HwNetworkParams = std::tuple<Precision, Precision>;

class MyriadX_HW_Networks_Tests_nightly :
        public MyriadX_HW_Tests_nightly,
        public testing::WithParamInterface<HwNetworkParams> {
public:
    Precision inputPrecision;
    Precision outputPrecision;

    Blob::Ptr _input;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(MyriadX_HW_Tests_nightly::SetUp());

        inputPrecision = std::get<0>(GetParam());
        outputPrecision = std::get<1>(GetParam());
    }

    Blob::Ptr getFp32Blob(const Blob::Ptr& in) {
        if (in->getTensorDesc().getPrecision() == Precision::FP32)
            return in;

        auto out = make_shared_blob<float>({Precision::FP32, in->getTensorDesc().getDims(), in->getTensorDesc().getLayout()});
        out->allocate();

        if (in->getTensorDesc().getPrecision() == Precision::FP16) {
            PrecisionUtils::f16tof32Arrays(out->buffer().as<float *>(), in->cbuffer().as<ie_fp16 *>(), in->size());
        } else {
            ADD_FAILURE() << "Unsupported precision " << in->getTensorDesc().getPrecision();
        }

        return out;
    }

    Blob::Ptr getFp16Blob(const Blob::Ptr& in) {
        if (in->getTensorDesc().getPrecision() == Precision::FP16)
            return in;

        auto out = make_shared_blob<ie_fp16>({Precision::FP16, in->getTensorDesc()/*??*/.getDims(), in->getTensorDesc().getLayout()});
        out->allocate();

        if (in->getTensorDesc().getPrecision() == Precision::FP32) {
            PrecisionUtils::f32tof16Arrays(out->buffer().as<ie_fp16 *>(), in->cbuffer().as<float *>(), in->size());
        } else {
            ADD_FAILURE() << "Unsupported precision " << in->getTensorDesc().getPrecision();
        }

        return out;
    }

    void RunAsyncTest(int numIters = 20) {
        if (!CheckMyriadX()) {
            GTEST_SKIP() << "Non-MyriadX device";
        }

        auto fnPtr = ngraph::builder::subgraph::makeSplitMultiConvConcat();
        ASSERT_NO_THROW(_cnnNetwork = CNNNetwork(fnPtr));

        _cnnNetwork.getInputsInfo().begin()->second->setPrecision(inputPrecision);
        _cnnNetwork.getOutputsInfo().begin()->second->setPrecision(outputPrecision);

        _input = FuncTestUtils::createAndFillBlob(_cnnNetwork.getInputsInfo().begin()->second->getTensorDesc());

        auto runTest = [&]() {
            const int NUM_REQUESTS = 4;

            std::map<std::string, std::string> config = {
                { InferenceEngine::MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES) },
                { CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES) },
                { InferenceEngine::MYRIAD_PERF_REPORT_MODE, InferenceEngine::MYRIAD_PER_STAGE }
            };

            ASSERT_NO_THROW(_exeNetwork = _vpuPluginPtr->LoadNetwork(_cnnNetwork, config));
            
            InferRequest inferRequests[NUM_REQUESTS];
            Blob::Ptr outputs[NUM_REQUESTS];

            for (int i = 0; i < NUM_REQUESTS; ++i) {
                ASSERT_NO_THROW(inferRequests[i] = _exeNetwork.CreateInferRequest());
                ASSERT_NO_THROW(inferRequests[i].SetBlob(_cnnNetwork.getInputsInfo().begin()->first.c_str(), _input));
                ASSERT_NO_THROW(outputs[i] = inferRequests[i].GetBlob(_cnnNetwork.getOutputsInfo().begin()->first.c_str()));
            }

            std::vector<Blob::Ptr> allOutputs[NUM_REQUESTS];
            for (int i = 0; i < NUM_REQUESTS; ++i) {
                allOutputs[i].resize(numIters);
            }

            for (int iterInd = 0; iterInd < numIters; ++iterInd) {
                for (int inferInd = 0; inferInd < NUM_REQUESTS; ++inferInd) {
                    ASSERT_NO_THROW(inferRequests[inferInd].StartAsync());
                }

                for (int inferInd = 0; inferInd < NUM_REQUESTS; ++inferInd) {
                    ASSERT_EQ(StatusCode::OK, inferRequests[inferInd].Wait(InferRequest::RESULT_READY));
                }

                for (int inferInd = 0; inferInd < NUM_REQUESTS; ++inferInd) {
                    auto tensorDesc = outputs[inferInd]->getTensorDesc();
                    tensorDesc.setPrecision(Precision::FP16);

                    allOutputs[inferInd][iterInd] = make_blob_with_precision(Precision::FP16, tensorDesc);
                    allOutputs[inferInd][iterInd]->allocate();

                    auto outputFP16 = getFp16Blob(outputs[inferInd]);

                    ie_memcpy(allOutputs[inferInd][iterInd]->buffer(), allOutputs[inferInd][iterInd]->byteSize(),
                              outputFP16->cbuffer(), outputFP16->byteSize());
                }
            }

            for (int iterInd1 = 0; iterInd1 < numIters; ++iterInd1) {
                for (int iterInd2 = iterInd1; iterInd2 < numIters; ++iterInd2) {
                    for (int inferInd1 = 0; inferInd1 < NUM_REQUESTS; ++inferInd1) {
                        for (int inferInd2 = inferInd1; inferInd2 < NUM_REQUESTS; ++inferInd2) {
                            ASSERT_NO_FATAL_FAILURE(CompareCommonAbsolute(allOutputs[inferInd1][iterInd1], allOutputs[inferInd2][iterInd2], 0.0f))
                                    << "inferInd1=" << inferInd1 << " "
                                    << "iterInd1=" << iterInd1 << " "
                                    << "inferInd2=" << inferInd2 << " "
                                    << "iterInd2=" << iterInd2;
                        }
                    }
                }
            }
        };

        runTest();
    }
};

TEST_P(MyriadX_HW_Networks_Tests_nightly, SimpleNetAsync) {
    RunAsyncTest(100);
}

inline std::string getTestCaseName(const testing::TestParamInfo<HwNetworkParams>& param) {
    return std::string((std::get<0>(param.param)).name()) + "_" +
           std::string((std::get<1>(param.param)).name());
}

INSTANTIATE_TEST_SUITE_P(Input_Output_ExecMode, MyriadX_HW_Networks_Tests_nightly,
    testing::Values(
          std::make_tuple(Precision::FP16, Precision::FP16)
    ),
    getTestCaseName
);
