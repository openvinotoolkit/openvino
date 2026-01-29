// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace pass {

class SequenceConcatReplacer : public ov::pass::MatcherPass {
public:
    SequenceConcatReplacer();
};

}  // namespace pass
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
