// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "myriad_layers_tests.hpp"
#include <cmath>

using namespace InferenceEngine;

#define ERROR_BOUND 0.0005f

PRETTY_PARAM(Coords, int)
PRETTY_PARAM(Classes, int)
PRETTY_PARAM(Num, int)
PRETTY_PARAM(MaskSize, int)
PRETTY_PARAM(DoSoftmax, int)
PRETTY_PARAM(CustomConfig, std::string)

typedef myriadLayerTestBaseWithParam<std::tuple<Coords, Classes, Num, MaskSize, DoSoftmax,
    vpu::LayoutPreference, IRVersion, CustomConfig>> myriadLayersTestsRegionYolo_smoke;

TEST_P(myriadLayersTestsRegionYolo_smoke, RegionYolo) {
    const int coords = std::get<0>(GetParam());
    const int classes = std::get<1>(GetParam());
    const int num = std::get<2>(GetParam());
    const int maskSize = std::get<3>(GetParam());
    const int doSoftmax = std::get<4>(GetParam());
    const auto layoutPreference = std::get<5>(GetParam());
    _irVersion = std::get<6>(GetParam());
    const std::string customConfig = std::get<7>(GetParam());

    if (!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP() << "Custom layers for MYRIAD2 not supported";
    }

    _config[InferenceEngine::MYRIAD_CUSTOM_LAYERS] = customConfig;

    const auto mask = [&] {
        std::string mask;
        for (int i = 0; i < maskSize; i++) {
            mask += std::to_string(i) + ',';
        }
        if (!mask.empty()) mask.pop_back();
        return mask;
    }();

    std::map<std::string, std::string> params;
    params["coords"] = std::to_string(coords);
    params["classes"] = std::to_string(classes);
    params["num"] = std::to_string(num);
    params["mask"] = mask;
    params["do_softmax"] = std::to_string(doSoftmax);
    params["axis"] = "0";
    params["end_axis"] = "0";

    const auto dims = [&] {
        const auto regions = doSoftmax ? num : maskSize;
        const uint32_t channels = (coords + classes + 1) * regions;
        IE_ASSERT(channels > 0);
        return tensor_test_params{1, channels, 13, 13};
    }();

    SetInputTensor(dims);
    SetOutputTensor(dims);

    ASSERT_NO_FATAL_FAILURE(makeSingleLayerNetwork(LayerInitParams("RegionYolo").params(params),
                                                   NetworkInitParams()
                                                       .layoutPreference(layoutPreference)
                                                       .lockLayout(true)));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(ref_RegionYolo(_inputMap.begin()->second, _refBlob,
    	coords, classes, num, maskSize, doSoftmax));

    CompareCommonAbsolute(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

std::vector<CustomConfig> s_CustomConfig = {
	{""},
#ifdef VPU_HAS_CUSTOM_KERNELS
    getIELibraryPath() + "/vpu_custom_kernels/customLayerBindings.xml"
#endif
};
