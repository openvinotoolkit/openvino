// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

static void reorg_calculate(const Blob::Ptr src, Blob::Ptr dst, int stride)
{
	ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
          uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);

    const auto inputDims = src->getTensorDesc().getDims();
    const int C = inputDims[1];
    const int H = inputDims[2];
    const int W = inputDims[3];

	const auto inputCHW = [&] {
		auto inputCHW = std::vector<ie_fp16>(C*H*W);
		if (Layout::NCHW == src->getTensorDesc().getLayout()) {
			std::copy(src_data, src_data + C*H*W, begin(inputCHW));
		} else {
			for (int c = 0; c < C; c++) {
				for (int h = 0; h < H; h++) {
					for (int w = 0; w < W; w++) {
						inputCHW[c*H*W + h*W + w] = src_data[h*W*C + w*C + c];
					}
				}
			}
		}
		return inputCHW;
	}();

	const int C2 = C/(stride*stride);
	const int H2 = H*stride;
	const int W2 = W*stride;

    for (int c = 0; c < C; ++c) {
		for (int h = 0; h < H; ++h) {
			for (int w = 0; w < W; ++w) {
				const int offset = c/C2;
				const int c2 = c - C2*offset;
				const int h2 = h*stride + offset/stride;
				const int w2 = w*stride + offset - stride*(offset/stride);

				dst_data[c*H*W + h*W + w] = inputCHW[c2*H2*W2 + h2*W2 + w2];
			}
		}
	}

	dst->getTensorDesc().setLayout(Layout::NCHW);
}

PRETTY_PARAM(Stride, int);
PRETTY_PARAM(layoutPreference, vpu::LayoutPreference);
PRETTY_PARAM(CustomConfig, std::string)

typedef myriadLayerTestBaseWithParam<std::tuple<SizeVector, Stride, layoutPreference, IRVersion, CustomConfig>>
	myriadLayersTestsReorg_smoke;

TEST_P(myriadLayersTestsReorg_smoke, TestsReorg) {
    const SizeVector dimsInput = std::get<0>(GetParam());
    const int stride = std::get<1>(GetParam());
    const auto layoutPreference = std::get<2>(GetParam());
    _irVersion = std::get<3>(GetParam());
	const std::string customConfig = std::get<4>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
		GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
	}
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    const auto dimsOutput = SizeVector{dimsInput[0],
									   dimsInput[1] * (stride * stride),
									   dimsInput[2] / stride,
									   dimsInput[3] / stride};

    SetInputTensors({dimsInput});
    SetOutputTensors({dimsOutput});

    std::map<std::string, std::string> params;
    params["stride"] = std::to_string(stride);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("ReorgYolo").params(params),
												   NetworkInitParams()
													   .layoutPreference(layoutPreference)
													   .lockLayout(true)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(reorg_calculate(_inputMap.begin()->second, _refBlob, stride));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, 0);
}

static std::vector<CustomConfig> s_CustomConfig = {
	{""},
#ifdef VPU_HAS_CUSTOM_KERNELS
    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};

static std::vector<SizeVector> s_ReorgInputs = {
		{1, 64, 26, 26},
		{1, 192, 6 * 26, 6 * 26},
		{1, 4, 6, 6}
};