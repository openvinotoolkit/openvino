// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_pool_nd_test.hpp"

using namespace testing;

//======================================================================
//
// 3D, tricky input size, kernel shape, strides, pads
//
//======================================================================

INSTANTIATE_TEST_SUITE_P(tricky_ncdhw_avg_userpad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 19, 65, 47}),
        Values(KernelShape {1, 3, 5}),
        Values(Strides {1, 2, 3}),
        Values(PadsBegin {0, 1, 1}),
        Values(PadsEnd {0, 1, 3}),
        Values(AutoPad("")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(false),
               ExcludePad(true))
    )
);

INSTANTIATE_TEST_SUITE_P(tricky_ncdhw_max_userpad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 19, 65, 47}),
        Values(KernelShape {1, 3, 5}),
        Values(Strides {1, 2, 3}),
        Values(PadsBegin {0, 1, 1}),
        Values(PadsEnd {0, 1, 3}),
        Values(AutoPad("")),
        Values(PoolingMethod("max")),
        Values(RoundingType("")),
        Values(ExcludePad(false))
    )
);

INSTANTIATE_TEST_SUITE_P(tricky_ncdhw_avg_autopad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 19, 65, 47}),
        Values(KernelShape {1, 3, 5}),
        Values(Strides {1, 2, 3}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(false),
               ExcludePad(true))
    )
);

INSTANTIATE_TEST_SUITE_P(tricky_ncdhw_max_autopad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 19, 65, 47}),
        Values(KernelShape {1, 3, 5}),
        Values(Strides {1, 2, 3}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(PoolingMethod("max")),
        Values(RoundingType("")),
        Values(ExcludePad(false))
    )
);

//======================================================================
//
// 3D, simple input size, kernel shape, strides, pads
//
//======================================================================

INSTANTIATE_TEST_SUITE_P(simple_ncdhw_avg_userpad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 20, 64, 48}),
        Values(KernelShape {3, 3, 3}),
        Values(Strides {2, 2, 2}),
        Values(PadsBegin {1, 1, 1}),
        Values(PadsEnd {1, 1, 1}),
        Values(AutoPad("")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(false),
               ExcludePad(true))
    )
);

INSTANTIATE_TEST_SUITE_P(simple_ncdhw_max_userpad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 20, 64, 48}),
        Values(KernelShape {3, 3, 3}),
        Values(Strides {2, 2, 2}),
        Values(PadsBegin {1, 1, 1}),
        Values(PadsEnd {1, 1, 1}),
        Values(AutoPad("")),
        Values(PoolingMethod("max")),
        Values(RoundingType("")),
        Values(ExcludePad(false))
    )
);

//----------------------------------------------------------------------
//
// HACK: Exclude "same_upper" with excludePad=false case,
//       as 2D pool for Myriad seems to always exclude pad
//
// Issue-25902 does software 2D avg pooling for Myriad support exclude-pad
// Issue-15146 HW AvgPool doesn't support excludePad parameter
//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(simple_ncdhw_avg_autopad_1,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 20, 64, 48}),
        Values(KernelShape {3, 3, 3}),
        Values(Strides {2, 2, 2}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(false),
               ExcludePad(true))
    )
);

INSTANTIATE_TEST_SUITE_P(simple_ncdhw_avg_autopad_2,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 20, 64, 48}),
        Values(KernelShape {3, 3, 3}),
        Values(Strides {2, 2, 2}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("same_upper")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(true))
    )
);

//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(simple_ncdhw_max_autopad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 20, 64, 48}),
        Values(KernelShape {3, 3, 3}),
        Values(Strides {2, 2, 2}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(PoolingMethod("max")),
        Values(RoundingType("")),
        Values(ExcludePad(false))
    )
);

//======================================================================
//
// 2D, tricky input size, kernel shape, strides, pads
//
//======================================================================

INSTANTIATE_TEST_SUITE_P(tricky_nchw_avg_userpad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 65, 47}),
        Values(KernelShape {1, 5}),
        Values(Strides {1, 3}),
        Values(PadsBegin {0, 1}),
        Values(PadsEnd {0, 3}),
        Values(AutoPad("")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(false),
               ExcludePad(true))
    )
);

INSTANTIATE_TEST_SUITE_P(tricky_nchw_max_userpad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 65, 47}),
        Values(KernelShape {1, 5}),
        Values(Strides {1, 3}),
        Values(PadsBegin {0, 1}),
        Values(PadsEnd {0, 3}),
        Values(AutoPad("")),
        Values(PoolingMethod("max")),
        Values(RoundingType("")),
        Values(ExcludePad(false))
    )
);

INSTANTIATE_TEST_SUITE_P(tricky_nchw_avg_autopad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 65, 47}),
        Values(KernelShape {1, 5}),
        Values(Strides {1, 3}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(false),
               ExcludePad(true))
    )
);

INSTANTIATE_TEST_SUITE_P(tricky_nchw_max_autopad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 65, 47}),
        Values(KernelShape {1, 5}),
        Values(Strides {1, 3}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(PoolingMethod("max")),
        Values(RoundingType("")),
        Values(ExcludePad(false))
    )
);

//======================================================================
//
// 2D, simple input size, kernel shape, strides, pads
//
//======================================================================

INSTANTIATE_TEST_SUITE_P(simple_nchw_avg_userpad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 64, 48}),
        Values(KernelShape {3, 3}),
        Values(Strides {2, 2}),
        Values(PadsBegin {1, 1}),
        Values(PadsEnd {1, 1}),
        Values(AutoPad("")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(false),
               ExcludePad(true))
    )
);

INSTANTIATE_TEST_SUITE_P(simple_nchw_max_userpad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 64, 48}),
        Values(KernelShape {3, 3}),
        Values(Strides {2, 2}),
        Values(PadsBegin {1, 1}),
        Values(PadsEnd {1, 1}),
        Values(AutoPad("")),
        Values(PoolingMethod("max")),
        Values(RoundingType("")),
        Values(ExcludePad(false))
    )
);

//----------------------------------------------------------------------
//
// HACK: Exclude "same_upper" with excludePad=false case,
//       as 2D pool for Myriad seems to always exclude pad
//
// Issue-25902 does software 2D avg pooling for Myriad support exclude-pad
// Issue-15146 HW AvgPool doesn't support excludePad parameter
//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(simple_nchw_avg_autopad_1,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 64, 48}),
        Values(KernelShape {3, 3}),
        Values(Strides {2, 2}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(false),
               ExcludePad(true))
    )
);

INSTANTIATE_TEST_SUITE_P(simple_nchw_avg_autopad_2,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 64, 48}),
        Values(KernelShape {3, 3}),
        Values(Strides {2, 2}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("same_upper")),
        Values(PoolingMethod("avg")),
        Values(RoundingType("")),
        Values(ExcludePad(true))
    )
);

//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(simple_nchw_max_autopad,
                        myriadLayersPoolNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 64, 48}),
        Values(KernelShape {3, 3}),
        Values(Strides {2, 2}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(PoolingMethod("max")),
        Values(RoundingType("")),
        Values(ExcludePad(false))
    )
);

//======================================================================
//
//  Test cases from the I3D network
//
//======================================================================

INSTANTIATE_TEST_SUITE_P(i3d_id10,
                        myriadLayersPoolNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 64, 40, 112, 112}),
                                Values(KernelShape {1, 3, 3}),
                                Values(Strides {1, 2, 2}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(PoolingMethod("max")),
                                Values(RoundingType("")),
                                Values(ExcludePad(true))));

INSTANTIATE_TEST_SUITE_P(i3d_id47,
                        myriadLayersPoolNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 192, 40, 28, 28}),
                                Values(KernelShape {3, 3, 3}),
                                Values(Strides {1, 1, 1}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(PoolingMethod("max")),
                                Values(RoundingType("")),
                                Values(ExcludePad(true))));

INSTANTIATE_TEST_SUITE_P(i3d_id247,
                        myriadLayersPoolNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 832, 20, 14, 14}),
                                Values(KernelShape {2, 2, 2}),
                                Values(Strides {2, 2, 2}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(PoolingMethod("max")),
                                Values(RoundingType("")),
                                Values(ExcludePad(true))));

INSTANTIATE_TEST_SUITE_P(i3d_id312,
                        myriadLayersPoolNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 1024, 10, 7, 7}),
                                Values(KernelShape {2, 7, 7}),
                                Values(Strides {1, 1, 1}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("valid")),
                                Values(PoolingMethod("avg")),
                                Values(RoundingType("")),
                                Values(ExcludePad(true))));

//======================================================================
