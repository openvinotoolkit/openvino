// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_oneHot_test.hpp"

INSTANTIATE_TEST_SUITE_P(accuracy, myriadLayerTestOneHot_smoke,
                        ::testing::Values<oneHot_test_params>(
                                MAKE_STRUCT(OneHotParams, {64}, 2, {0}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {64}, 2, {-1}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {32, 64}, 2, {0}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {32, 64}, 2, {1}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {32, 64}, 2, {-1}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {16, 32, 64}, 2, {0}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {16, 32, 64}, 2, {1}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {16, 32, 64}, 2, {-1}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {8, 16, 32,64}, 2, {0}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {8, 16, 32,64}, 2, {1}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {8, 16, 32,64}, 2, {-1}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {4, 8, 16, 32, 64}, 2, {0}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {4, 8, 16, 32, 64}, 2, {1}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {4, 8, 16, 32, 64}, 2, {-1}, {}, {})
                        ));

INSTANTIATE_TEST_SUITE_P(accuracy_add, myriadLayerTestOneHot_smoke,
                        ::testing::Values<oneHot_test_params>(
                                MAKE_STRUCT(OneHotParams, {16, 32, 64}, 2, {2}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {8, 16, 32,64}, 2, {2}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {8, 16, 32,64}, 2, {3}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {4, 8, 16, 32, 64}, 2, {2}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {4, 8, 16, 32, 64}, 2, {3}, {}, {}),
                                MAKE_STRUCT(OneHotParams, {4, 8, 16, 32, 64}, 2, {4}, {}, {})
                        ));
