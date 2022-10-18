// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>
#include <utility>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertNMS1ToNMS5;
class TRANSFORMATIONS_API ConvertNMS3ToNMS5;
class TRANSFORMATIONS_API ConvertNMS4ToNMS5;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertNMS1ToNMS5 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMS1ToNMS5", "0");
    ConvertNMS1ToNMS5();
};

class ngraph::pass::ConvertNMS3ToNMS5 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMS3ToNMS5", "0");
    ConvertNMS3ToNMS5();
};

class ngraph::pass::ConvertNMS4ToNMS5 : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMS4ToNMS5", "0");
    ConvertNMS4ToNMS5();
};
