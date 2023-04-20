// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {
class ConvertConv1DBase: public ngraph::pass::MatcherPass {
protected:
    OPENVINO_RTTI("ConvertConv1DBase", "0");
    template <class Conv>
    ngraph::matcher_pass_callback convert_conv1d_to_conv2d();
};

class ConvertConv1D: public ConvertConv1DBase {
public:
    OPENVINO_RTTI("ConvertConv1D", "0");
    ConvertConv1D();
};

class ConvertGroupConv1D: public ConvertConv1DBase {
public:
    OPENVINO_RTTI("ConvertGroupConv1D", "0");
    ConvertGroupConv1D();
};
}  // namespace intel_cpu
}  // namespace ov