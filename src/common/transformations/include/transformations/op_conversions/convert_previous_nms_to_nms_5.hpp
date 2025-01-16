// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertNMS1ToNMS5;
class TRANSFORMATIONS_API ConvertNMS3ToNMS5;
class TRANSFORMATIONS_API ConvertNMS4ToNMS5;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertNMS1ToNMS5 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMS1ToNMS5", "0");
    ConvertNMS1ToNMS5();
};

class ov::pass::ConvertNMS3ToNMS5 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMS3ToNMS5", "0");
    ConvertNMS3ToNMS5();
};

class ov::pass::ConvertNMS4ToNMS5 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertNMS4ToNMS5", "0");
    ConvertNMS4ToNMS5();
};
