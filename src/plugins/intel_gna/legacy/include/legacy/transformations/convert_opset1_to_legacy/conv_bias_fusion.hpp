// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

#include "legacy/ngraph_ops/convolution_ie.hpp"
#include "legacy/ngraph_ops/deconvolution_ie.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/rt_info.hpp"

namespace ngraph {
namespace pass {

class ConvFusion;
class ConvAddFusion;
class ConvMultiplyFusion;
class DeconvAddFusion;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvAddFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvAddFusion", "0");
    ConvAddFusion();
};

class ngraph::pass::ConvMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvMultiplyFusion", "0");
    ConvMultiplyFusion();
};

class ngraph::pass::DeconvAddFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("DeconvAddFusion", "0");
    DeconvAddFusion();
};

class ngraph::pass::ConvFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvFusion", "0");
    ConvFusion() {
        add_matcher<ngraph::pass::ConvAddFusion>();
        add_matcher<ngraph::pass::ConvMultiplyFusion>();
        add_matcher<ngraph::pass::DeconvAddFusion>();
    }
};
