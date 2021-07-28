// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_scatter_update_test.hpp"

using namespace testing;

//----------------------------------------------------------------------
//
// Multi-dimensional input/output and other tensors
//
//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    nd_tensors,
    myriadLayersScatterUpdateTest_smoke,
    Values(
        //  1-dimensional `indices`
        ScatterUpdateTestParams { { 1000 }, { 100000 } },
        ScatterUpdateTestParams { { 105 },  { 351, 299 } },
        ScatterUpdateTestParams { { 17 },   { 48, 55, 39 } },
        ScatterUpdateTestParams { { 10 },   { 23, 14, 19, 17 } },
        ScatterUpdateTestParams { { 7 },    { 10, 9, 11, 8, 13 } },
        ScatterUpdateTestParams { { 6 },    { 9, 5, 11, 7, 5, 6 } },
        ScatterUpdateTestParams { { 5 },    { 7, 6, 5, 7, 6, 3, 4 } },
        ScatterUpdateTestParams { { 3 },    { 5, 3, 5, 7, 3, 4, 6, 3 } },
        //  2-dimensional `indices`
        ScatterUpdateTestParams { { 35, 29 }, { 100000 } },
        ScatterUpdateTestParams { { 13, 9 },  { 351, 299 } },
        ScatterUpdateTestParams { { 5, 3 },   { 48, 55, 39 } },
        ScatterUpdateTestParams { { 3, 3 },   { 23, 14, 19, 17 } },
        ScatterUpdateTestParams { { 3, 2 },   { 10, 9, 11, 8, 13 } },
        ScatterUpdateTestParams { { 3, 2 },   { 9, 5, 11, 7, 5, 6 } },
        ScatterUpdateTestParams { { 2, 2 },   { 7, 6, 5, 7, 6, 3, 4 } },
        //  3-dimensional `indices`
        ScatterUpdateTestParams { { 13, 11, 7 }, { 100000 } },
        ScatterUpdateTestParams { { 5, 7, 3 },   { 351, 299 } },
        ScatterUpdateTestParams { { 5, 2, 2 },   { 48, 55, 39 } },
        ScatterUpdateTestParams { { 3, 2, 2 },   { 23, 14, 19, 17 } },
        ScatterUpdateTestParams { { 2, 2, 2 },   { 10, 9, 11, 8, 13 } },
        ScatterUpdateTestParams { { 2, 2, 2 },   { 9, 5, 11, 7, 5, 6 } }
    )
);

//----------------------------------------------------------------------
//
// Real-life (or similar) test cases
//
//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    use_cases,
    myriadLayersScatterUpdateTest_smoke,
    Values(
        // use case from Mask R-CNN: N = 1000, C = 256, HxW = 7x7
        ScatterUpdateTestParams { { 32 },      { 1000, 256, 7, 7} },
        ScatterUpdateTestParams { { 5, 6 },    { 1000, 256, 7, 7} },
        ScatterUpdateTestParams { { 5, 3, 2 }, { 1000, 256, 7, 7} },
        // large 1D copy: N=1 (hidden), C=64, D=40, H = W = 112
        ScatterUpdateTestParams { { 32 },      { 64, 40, 112, 112 } },
        ScatterUpdateTestParams { { 5, 6 },    { 64, 40, 112, 112 } },
        ScatterUpdateTestParams { { 5, 3, 2 }, { 64, 40, 112, 112 } },
        // many planes for 3D copy: N=16, C=512, H=W=56
        ScatterUpdateTestParams { { 12, },      { 16, 512, 56, 56 } },
        ScatterUpdateTestParams { { 3, 4, },    { 16, 512, 56, 56 } },
        ScatterUpdateTestParams { { 3, 2, 2, }, { 16, 512, 56, 56 } }
    )
);
