// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>

#include "ngraph/pattern/matcher.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/add.hpp"

#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph_ops/deconvolution_ie.hpp"
#include "ngraph/op/group_conv.hpp"
#include "ngraph/rt_info.hpp"

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvFusion;
class TRANSFORMATIONS_API ConvAddFusion;
class TRANSFORMATIONS_API ConvMultiplyFusion;
class TRANSFORMATIONS_API DeconvAddFusion;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvAddFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvAddFusion();
};

class ov::pass::ConvMultiplyFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvMultiplyFusion();
};

class ov::pass::DeconvAddFusion: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    DeconvAddFusion();
};

class ov::pass::ConvFusion: public ov::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvFusion() {
        add_matcher<ov::pass::ConvAddFusion>();
        add_matcher<ov::pass::ConvMultiplyFusion>();
        add_matcher<ov::pass::DeconvAddFusion>();
    }
};
