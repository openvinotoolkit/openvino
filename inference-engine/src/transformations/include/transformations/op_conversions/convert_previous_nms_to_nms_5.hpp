// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertNMS1ToNMS5;
class TRANSFORMATIONS_API ConvertNMS3ToNMS5;
class TRANSFORMATIONS_API ConvertNMS4ToNMS5;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNMS1ToNMS5: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNMS1ToNMS5();
};

class ov::pass::ConvertNMS3ToNMS5: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNMS3ToNMS5();
};

class ov::pass::ConvertNMS4ToNMS5: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertNMS4ToNMS5();
};

