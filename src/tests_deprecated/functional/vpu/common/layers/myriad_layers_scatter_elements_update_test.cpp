// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_scatter_elements_update_test.hpp"

using namespace testing;

static const std::vector<DataType> dataTypeList = { "FP16", "I32" };

static const std::vector<DataShape> dataShapeList_ndTensors = {
    // tiny `data` tensor
    { 10 },
    { 3, 3 },
    { 2, 3, 2 },
    { 2, 2, 2, 2 },

    // small `data` tensor
    { 100 },
    { 10, 10 },
    { 5, 5, 5 },
    { 3, 3, 3, 3 },

    // medium-size `data` tensor
    { 1000 },
    { 32, 33 },
    { 10, 10, 10 },
    { 5, 5, 5, 8 },
    { 3, 5, 4, 5, 3 },
    { 3, 3, 3, 3, 3, 4 },
    { 2, 3, 3, 3, 3, 3, 2 },
    { 3, 3, 3, 2, 2, 2, 2, 2 },

    // large `data` tensor
    { 100000 },
    { 351, 299 },
    { 48, 55, 39 },
    { 23, 14, 19, 17 },
    { 10, 9, 11, 8, 13 },
    { 9, 5, 11, 7, 5, 6 },
    { 7, 6, 5, 7, 6, 3, 4 },
    { 5, 3, 5, 7, 3, 4, 6, 3 },
};

static const std::vector<DataShape> dataShapeList_useCases = {
    // from Mask R-CNN: N = 1000, C = 256, HxW = 7x7
    { 1000, 256, 7, 7 },

    // large 1D copy: N=1 (hidden), C=64, D=40, H = W = 112
    { 64, 40, 112, 112 },

    // many planes for 3D copy: N=16, C=512, H=W=56
    { 16, 512, 56, 56 },
};

INSTANTIATE_TEST_SUITE_P(nd_tensors, myriadLayersScatterElementsUpdateTest_smoke,
                        Combine(
                            ValuesIn(dataShapeList_ndTensors),
                            ValuesIn(dataTypeList)));

INSTANTIATE_TEST_SUITE_P(use_cases, myriadLayersScatterElementsUpdateTest_smoke,
                        Combine(
                            ValuesIn(dataShapeList_useCases),
                            ValuesIn(dataTypeList)));
