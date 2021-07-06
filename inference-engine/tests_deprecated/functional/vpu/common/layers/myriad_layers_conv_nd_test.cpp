// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_conv_nd_test.hpp"

using namespace testing;

//----------------------------------------------------------------------
//
// 3D, tricky input size, kernel shape, pads, strides, and dilations
//
//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(tricky_ncdhw_userpad, myriadLayersConvNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 19, 65, 47}),
        Values(KernelShape {1, 3, 5}),
        Values(PadsBegin {0, 1, 1}),
        Values(PadsEnd {0, 1, 3}),
        Values(AutoPad("")),
        Values(Strides {1, 2, 3}),
        Values(Dilations {3, 2, 1}),
        Values(OutputChannels(16)),
        Values(Groups(1))
    )
);

INSTANTIATE_TEST_SUITE_P(tricky_ncdhw_autopad, myriadLayersConvNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 19, 65, 47}),
        Values(KernelShape {1, 3, 5}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(Strides {1, 2, 3}),
        Values(Dilations {3, 2, 1}),
        Values(OutputChannels(16)),
        Values(Groups(1))
    )
);

//----------------------------------------------------------------------
//
// 3D, simple input size, kernel shape, pads, strides, and dilations
//
//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(simple_ncdhw_userpad, myriadLayersConvNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 20, 64, 48}),
        Values(KernelShape {3, 3, 3}),
        Values(PadsBegin {1, 1, 1}),
        Values(PadsEnd {1, 1, 1}),
        Values(AutoPad("")),
        Values(Strides {2, 2, 2}),
        Values(Dilations {1, 1, 1}),
        Values(OutputChannels(16)),
        Values(Groups(1))
    )
);

INSTANTIATE_TEST_SUITE_P(simple_ncdhw_autopad, myriadLayersConvNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 20, 64, 48}),
        Values(KernelShape {3, 3, 3}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(Strides {2, 2, 2}),
        Values(Dilations {1, 1, 1}),
        Values(OutputChannels(16)),
        Values(Groups(1))
    )
);

//----------------------------------------------------------------------
//
// 2D, tricky input size, kernel shape, pads, strides, and dilations
//
//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(tricky_nchw_userpad, myriadLayersConvNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 65, 47}),
        Values(KernelShape {1, 3}),
        Values(PadsBegin {0, 0}),
        Values(PadsEnd {0, 2}),
        Values(AutoPad("")),
        Values(Strides {1, 2}),
        Values(Dilations {2, 1}),
        Values(OutputChannels(16)),
        Values(Groups(1))
    )
);

INSTANTIATE_TEST_SUITE_P(tricky_nchw_autopad, myriadLayersConvNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 65, 47}),
        Values(KernelShape {1, 3}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(Strides {1, 2}),
        Values(Dilations {2, 1}),
        Values(OutputChannels(16)),
        Values(Groups(1))
    )
);

//----------------------------------------------------------------------
//
// 2D, simple input size, kernel shape, pads, strides, and dilations
//
//----------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(simple_nchw_userpad, myriadLayersConvNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 64, 48}),
        Values(KernelShape {3, 3}),
        Values(PadsBegin {1, 1}),
        Values(PadsEnd {1, 1}),
        Values(AutoPad("")),
        Values(Strides {2, 2}),
        Values(Dilations {1, 1}),
        Values(OutputChannels(16)),
        Values(Groups(1))
    )
);

INSTANTIATE_TEST_SUITE_P(simple_nchw_autopad, myriadLayersConvNDTest_smoke,
    Combine(
        Values(InputShape {1, 3, 64, 48}),
        Values(KernelShape {3, 3}),
        Values(PadsBegin {}),
        Values(PadsEnd {}),
        Values(AutoPad("valid"),
               AutoPad("same_lower"),
               AutoPad("same_upper")),
        Values(Strides {2, 2}),
        Values(Dilations {1, 1}),
        Values(OutputChannels(16)),
        Values(Groups(1))
    )
);

//----------------------------------------------------------------------
//
// Test cases from the I3D network
//
//----------------------------------------------------------------------

// NB: requires 1GB of RAM on device (e.g. ma2085 board)
// Stress test: large image with large depth, large kernel
INSTANTIATE_TEST_SUITE_P(i3d_id6, myriadLayersConvNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 3, 79, 224, 224}),
                                Values(KernelShape {7, 7, 7}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(Strides {2, 2, 2}),
                                Values(Dilations {1, 1, 1}),
                                Values(OutputChannels(64)),
                                Values(Groups(1))));

// Like `i3d_id6` test but with smaller image (so must fit in Myriad X)
INSTANTIATE_TEST_SUITE_P(i3d_id6_shrink, myriadLayersConvNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 3, 39, 112, 112}),
                                Values(KernelShape {7, 7, 7}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(Strides {2, 2, 2}),
                                Values(Dilations {1, 1, 1}),
                                Values(OutputChannels(64)),
                                Values(Groups(1))));

// Average-size image, trivial kernel 1x1x1
INSTANTIATE_TEST_SUITE_P(i3d_id12, myriadLayersConvNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 64, 40, 56, 56}),
                                Values(KernelShape {1, 1, 1}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(Strides {1, 1, 1}),
                                Values(Dilations {1, 1, 1}),
                                Values(OutputChannels(64)),
                                Values(Groups(1))));

// Average-size image, non-trivial kernel 3x3x3
INSTANTIATE_TEST_SUITE_P(i3d_id17, myriadLayersConvNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 64, 40, 56, 56}),
                                Values(KernelShape {3, 3, 3}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(Strides {1, 1, 1}),
                                Values(Dilations {1, 1, 1}),
                                Values(OutputChannels(192)),
                                Values(Groups(1))));

// Small image (7x7), trivial kernel
INSTANTIATE_TEST_SUITE_P(i3d_id249, myriadLayersConvNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 832, 10, 7, 7}),
                                Values(KernelShape {1, 1, 1}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(Strides {1, 1, 1}),
                                Values(Dilations {1, 1, 1}),
                                Values(OutputChannels(256)),
                                Values(Groups(1))));

// Small image (7x7), non-trivial kernel
INSTANTIATE_TEST_SUITE_P(i3d_id301, myriadLayersConvNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 48, 10, 7, 7}),
                                Values(KernelShape {3, 3, 3}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(Strides {1, 1, 1}),
                                Values(Dilations {1, 1, 1}),
                                Values(OutputChannels(128)),
                                Values(Groups(1))));

// Trivial image (1x1), trivial kernel
INSTANTIATE_TEST_SUITE_P(i3d_id314, myriadLayersConvNDTest_smoke,
                        Combine(
                                Values(InputShape {1, 1024, 9, 1, 1}),
                                Values(KernelShape {1, 1, 1}),
                                Values(PadsBegin {}),
                                Values(PadsEnd {}),
                                Values(AutoPad("same_upper")),
                                Values(Strides {1, 1, 1}),
                                Values(Dilations {1, 1, 1}),
                                Values(OutputChannels(400)),
                                Values(Groups(1))));
