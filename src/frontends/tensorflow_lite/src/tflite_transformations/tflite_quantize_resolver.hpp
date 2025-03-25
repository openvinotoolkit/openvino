// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace pass {

// Fuses Convert into TFLQuantize operation
class TFLQuantizeConvert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::tensorflow_lite::pass::TFLQuantizeConvert");
    TFLQuantizeConvert();
};

// Replaces TFLQuantize operation with FQ or sub-mul pattern if necessary
class TFLQuantizeReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::tensorflow_lite::pass::TFLQuantizeReplacer");
    TFLQuantizeReplacer();
};

// This transformation simplifies type manipulations in the graph
class TFLQuantizeResolver : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::tensorflow_lite::pass::TFLQuantizeResolver");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
