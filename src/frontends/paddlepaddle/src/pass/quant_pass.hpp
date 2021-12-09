// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pass {

// handle FakeQuantInternal: fake_quantize_range_abs_max, fake_quantize_moving_average_abs_max
//    fake_quantize_abs_max(?), fake_channel_wise_quantize_abs_max(?)
class FuseQuant : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::FuseQuant");
    FuseQuant();
};

// handle FakeDequantInternal: fake_dequantize_max_abs
class FuseDequant : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::FuseDequant");
    FuseDequant();
};

// handle ChannelFakeQuantInternal: fake_channel_wise_dequantize_max_abs
class FuseChannelDequant : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::FuseChannelDequant");
    FuseChannelDequant();
};

// handle FakeQuantDequantInternal: fake_quantize_dequantize_moving_average_abs_max
//    fake_quantize_dequantize_abs_max, fake_channel_wise_quantize_dequantize_abs_max
class FuseQuantDequant : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::FuseQuantDequant");
    FuseQuantDequant();
};

}  // namespace pass
}  // namespace frontend
}  // namespace ov
