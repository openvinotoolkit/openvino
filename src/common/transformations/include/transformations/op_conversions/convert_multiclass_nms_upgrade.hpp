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

class TRANSFORMATIONS_API ConvertMulticlassNms8ToMulticlassNms9;

}  // namespace pass
}  // namespace ov

class ov::pass::ConvertMulticlassNms8ToMulticlassNms9 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMulticlassNms8ToMulticlassNms9");
    ConvertMulticlassNms8ToMulticlassNms9();
};
