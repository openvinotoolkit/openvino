// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ConvertGroupConvolution: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGroupConvolution", "0");
    ConvertGroupConvolution();
};
}  // namespace intel_cpu
}  // namespace ov
