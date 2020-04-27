// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <algorithm>

using std::tuple;
using std::get;

using namespace InferenceEngine;

PRETTY_PARAM(layoutPreference, vpu::LayoutPreference);
PRETTY_PARAM(hasThreeInputs, bool);
PRETTY_PARAM(transposeA, bool);
PRETTY_PARAM(transposeB, bool);

struct gemm_parameters {
    float alpha;
    float beta;

    long unsigned int M;
    long unsigned int N;
    long unsigned int K;

    long unsigned int MB1_A; long unsigned int MB2_A;
    long unsigned int MB1_B; long unsigned int MB2_B;
    long unsigned int MB1_C; long unsigned int MB2_C;
    long unsigned int MB1_D; long unsigned int MB2_D;

    friend std::ostream& operator<<(std::ostream& os, gemm_parameters const& tst)
    {
        return os << "alpha=" << tst.alpha << ", " << "beta=" << tst.beta << ", "
                  << "M=" << tst.M << ", " << "N=" << tst.N << ", " << "K=" << tst.K << ", "
                  << "MB1_A=" << tst.MB1_A << ", " << "MB2_A=" << tst.MB2_A << ", "
                  << "MB1_B=" << tst.MB1_B << ", " << "MB2_B=" << tst.MB2_B << ", "
                  << "MB1_C=" << tst.MB1_C << ", " << "MB2_C=" << tst.MB2_C << ", "
                  << "MB1_D=" << tst.MB1_D << ", " << "MB2_D=" << tst.MB2_D;
    };
};

static void gemm_ref(int M, int N, int K,
                     int MB1_A, int MB2_A,
                     int MB1_B, int MB2_B,
                     int MB1_C, int MB2_C,
                     int MB1, int MB2,
                     Blob::Ptr srcBlob1,
                     Blob::Ptr srcBlob2,
                     Blob::Ptr srcBlob3,
                     Blob::Ptr dstBlob,
                     float alpha,
                     float beta,
                     bool transposeA,
                     bool transposeB
                    )
{

    ie_fp16 *a = static_cast<ie_fp16*>(srcBlob1->buffer());
    ie_fp16 *b = static_cast<ie_fp16*>(srcBlob2->buffer());
    ie_fp16 *c = nullptr;
    ie_fp16 *d = static_cast<ie_fp16*>(dstBlob->buffer());

    const int stride_a = (transposeA ? M : K);
    const int stride_b = (transposeB ? K : N);
    const int stride_d = N;

    const int strideMB2_src1 = (MB2 != MB2_A) ? 0 : 1;
    const int strideMB2_src2 = (MB2 != MB2_B) ? 0 : 1;
    const int strideMB2_dst  = 1;

    const int strideMB1_src1 = (MB1 != MB1_A) ? 0 : MB2_A * M * K;
    const int strideMB1_src2 = (MB1 != MB1_B) ? 0 : MB2_B * K * N;
    const int strideMB1_dst  = MB2 * M * N;

    int strideMB2_src3 = 0;
    int strideMB1_src3 = 0;

    if (srcBlob3 != nullptr) {
        c = static_cast<ie_fp16 *>(srcBlob3->buffer());
        strideMB2_src3 = (MB2 != MB2_C) ? 0 : 1;
        strideMB1_src3 = (MB1 != MB1_C) ? 0 : MB2_C * M * N;
    }

    for (int mb1 = 0; mb1 < MB1; mb1++) {
        for (int mb2 = 0; mb2 < MB2; mb2++) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float dst = 0.0;
                    if (srcBlob3 != nullptr) {
                        dst = beta * PrecisionUtils::f16tof32(*(c + MB2_C * (j + i * N) + mb2 * strideMB2_src3 + mb1 * strideMB1_src3));
                    }
                    for (int k = 0; k < K; k++) {
                        float src1 = PrecisionUtils::f16tof32(transposeA ? *(a + MB2_A * (i + k * stride_a) + mb2 * strideMB2_src1 + mb1 * strideMB1_src1) : *(a + (k + i * stride_a) * MB2_A + mb2 * strideMB2_src1 + mb1 * strideMB1_src1));
                        float src2 = PrecisionUtils::f16tof32(transposeB ? *(b + MB2_B * (k + j * stride_b) + mb2 * strideMB2_src2 + mb1 * strideMB1_src2) : *(b + (j + k * stride_b) * MB2_B + mb2 * strideMB2_src2 + mb1 * strideMB1_src2));

                        dst += alpha * src1 * src2;
                    }

                    *(d + (j + i * N) * MB2 + mb2 * strideMB2_dst + mb1 * strideMB1_dst) = PrecisionUtils::f32tof16(dst);
                }
            }
        }
    }
}

typedef myriadLayerTestBaseWithParam<tuple<gemm_parameters, layoutPreference, hasThreeInputs, transposeA, transposeB>> myriadLayerGEMM;

TEST_P(myriadLayerGEMM, GEMM) {
    gemm_parameters gemm_parameter = get<0>(GetParam());
    auto layoutPreference = get<1>(GetParam());
    auto hasThreeInputs = get<2>(GetParam());
    auto transposeA = get<3>(GetParam());
    auto transposeB = get<4>(GetParam());

    const float alpha = gemm_parameter.alpha;
    const float beta = gemm_parameter.beta;

    const long unsigned int MB1_A = gemm_parameter.MB1_A; const long unsigned int MB2_A = gemm_parameter.MB2_A;
    const long unsigned int MB1_B = gemm_parameter.MB1_B; const long unsigned int MB2_B = gemm_parameter.MB2_B;
    const long unsigned int MB1_C = gemm_parameter.MB1_C; const long unsigned int MB2_C = gemm_parameter.MB2_C;
    const long unsigned int MB1_D = gemm_parameter.MB1_D; const long unsigned int MB2_D = gemm_parameter.MB2_D;

    IN_OUT_desc dims_input;
    IN_OUT_desc dims_output;

    dims_input.resize(2);
    if (hasThreeInputs) {
        dims_input.resize(3);
    }

    /* inputs */
    dims_input[0].resize(4);
    dims_input[0][0] = MB1_A;
    dims_input[0][1] = MB2_A;
    dims_input[0][2] = transposeA ? gemm_parameter.K : gemm_parameter.M;
    dims_input[0][3] = transposeA ? gemm_parameter.M : gemm_parameter.K;
    dims_input[1].resize(4);
    dims_input[1][0] = MB1_B;
    dims_input[1][1] = MB2_B;
    dims_input[1][2] = transposeB ? gemm_parameter.N : gemm_parameter.K;
    dims_input[1][3] = transposeB ? gemm_parameter.K : gemm_parameter.N;

    if (hasThreeInputs) {
        dims_input[2].resize(4);
        dims_input[2][0] = MB1_C;
        dims_input[2][1] = MB2_C;
        dims_input[2][2] = gemm_parameter.M;
        dims_input[2][3] = gemm_parameter.N;
    }


    dims_output.resize(1);
    dims_output[0].resize(4);
    dims_output[0][0] = MB1_D;
    dims_output[0][1] = MB2_D;
    dims_output[0][2] = gemm_parameter.M;
    dims_output[0][3] = gemm_parameter.N;

    SetInputTensors(dims_input);
    SetOutputTensors(dims_output);

    std::map<std::string, std::string> params {{"alpha", std::to_string(alpha)},
                                               {"beta", std::to_string(beta)},
                                               {"transpose_a", std::to_string(transposeA)},
                                               {"transpose_b", std::to_string(transposeB)},
                                              };

    if (MB1_D > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);


    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("GEMM").params(params), NetworkInitParams().layoutPreference(layoutPreference)));

    /* input tensor generating */
    auto pInputBlob = _inputMap.begin();
    Blob::Ptr inputBlob0 = pInputBlob->second;
    pInputBlob++;
    Blob::Ptr inputBlob1 = pInputBlob->second;
    Blob::Ptr inputBlob2 = nullptr;

    if (hasThreeInputs) {
        pInputBlob++;
        inputBlob2 = pInputBlob->second;
    }

    /* reference version */
    auto refOutBlob = make_shared_blob<ie_fp16>({Precision::FP16, {MB1_D, MB2_D, gemm_parameter.M, gemm_parameter.N}, Layout::NHWC});
    refOutBlob->allocate();
    gemm_ref(gemm_parameter.M, gemm_parameter.N, gemm_parameter.K,
             MB1_A, MB2_A,
             MB1_B, MB2_B,
             MB1_C, MB2_C,
             MB1_D, MB2_D,

             inputBlob0,
             inputBlob1,
             inputBlob2,
             refOutBlob,

             gemm_parameter.alpha,
             gemm_parameter.beta,
             transposeA,
             transposeB
            );

    ASSERT_TRUE(Infer());

    auto pOutputBlob = _outputMap.begin();
    auto outputBlob = pOutputBlob->second;
    float maxerr = 0.0016f * gemm_parameter.K;
    CompareCommonAbsolute(outputBlob, refOutBlob, maxerr);
}
