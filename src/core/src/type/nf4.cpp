// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Contains logic derived from bitsandbytes
// https://github.com/TimDettmers/bitsandbytes/blob/c82f51c0f784d8a43ebcb9cdefbf94e3f3b9c6c3/csrc/kernels.cu#L223
// implementation.
// Copyright notice from original source file is as follows.

//*******************************************************************************
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//==============================================================================

#include "openvino/core/type/nf4.hpp"

using namespace ov;

float ConvertNF4::dequantize(uint8_t val) {
    static const std::array<float, 16> lookup = {-1.0f,
                                                 -0.6961928009986877f,
                                                 -0.5250730514526367f,
                                                 -0.39491748809814453f,
                                                 -0.28444138169288635f,
                                                 -0.18477343022823334f,
                                                 -0.09105003625154495f,
                                                 0.0f,
                                                 0.07958029955625534f,
                                                 0.16093020141124725f,
                                                 0.24611230194568634f,
                                                 0.33791524171829224f,
                                                 0.44070982933044434f,
                                                 0.5626170039176941f,
                                                 0.7229568362236023f,
                                                 1.0f};
    return lookup[val];
}

uint8_t ConvertNF4::quantize(float x) {
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)          // 1
            if (x > 0.6427869200706482f)      // 11
                if (x > 0.8614784181118011f)  // 111
                    return 0b1111;
                else
                    return 0b1110;
            else if (x > 0.5016634166240692f)  // 110
                return 0b1101;
            else
                return 0b1100;
        else if (x > 0.2035212516784668f)  // 10
            if (x > 0.2920137718319893f)   // 101
                return 0b1011;
            else
                return 0b1010;
        else if (x > 0.1202552504837513f)  // 100
            return 0b1001;
        else
            return 0b1000;
    else if (x > -0.33967943489551544f)      // 0
        if (x > -0.13791173323988914f)       // 01
            if (x > -0.045525018125772476f)  // 011
                return 0b0111;
            else
                return 0b0110;
        else if (x > -0.23460740596055984f)  // 010
            return 0b0101;
        else
            return 0b0100;
    else if (x > -0.6106329262256622f)  // 00
        if (x > -0.4599952697753906f)   // 001
            return 0b0011;
        else
            return 0b0010;
    else if (x > -0.8480964004993439f)  // 000
        return 0b0001;
    else
        return 0b0000;
}
