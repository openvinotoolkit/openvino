// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class ConvertReduceMultiAxisBase: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertReduceMultiAxisBase", "0");
    template <class T>
    ngraph::matcher_pass_callback convert_reduce();
};

class ConvertReduceProd: public ConvertReduceMultiAxisBase {
public:
    OPENVINO_RTTI("ConvertReduceProd", "0");
    ConvertReduceProd();
};

class ConvertReduceMin: public ConvertReduceMultiAxisBase {
public:
    OPENVINO_RTTI("ConvertReduceMin", "0");
    ConvertReduceMin();
};

class ConvertReduceMax: public ConvertReduceMultiAxisBase {
public:
    OPENVINO_RTTI("ConvertReduceMax", "0");
    ConvertReduceMax();
};

class ConvertReduceSum: public ConvertReduceMultiAxisBase {
public:
    OPENVINO_RTTI("ConvertReduceSum", "0");
    ConvertReduceSum();
};

class ConvertReduceMultiAxis: public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertReduceMultiAxis", "0");
    ConvertReduceMultiAxis() {
        add_matcher<ConvertReduceProd>();
        add_matcher<ConvertReduceMin>();
        add_matcher<ConvertReduceMax>();
        add_matcher<ConvertReduceSum>();
    }
};

}   // namespace intel_cpu
}   // namespace ov
