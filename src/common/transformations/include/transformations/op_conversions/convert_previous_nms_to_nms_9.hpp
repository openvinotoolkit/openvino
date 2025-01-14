// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertNMS1ToNMS9;
class TRANSFORMATIONS_API ConvertNMS3ToNMS9;
class TRANSFORMATIONS_API ConvertNMS4ToNMS9;
class TRANSFORMATIONS_API ConvertNMS5ToNMS9;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNMS1ToNMS9 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertNMS1ToNMS9");
    ConvertNMS1ToNMS9();
};

class ov::pass::ConvertNMS3ToNMS9 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertNMS3ToNMS9");
    ConvertNMS3ToNMS9();
};

class ov::pass::ConvertNMS4ToNMS9 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertNMS4ToNMS9");
    ConvertNMS4ToNMS9();
};

class ov::pass::ConvertNMS5ToNMS9 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertNMS5ToNMS9");
    ConvertNMS5ToNMS9();
};
