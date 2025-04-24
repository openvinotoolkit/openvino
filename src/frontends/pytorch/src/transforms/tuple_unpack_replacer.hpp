// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

class PrimTupleUnpackReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::pytorch::pass::PrimTupleUnpackReplacer");
    PrimTupleUnpackReplacer();
};

class TupleUnpackInBodyReplacer : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::pytorch::pass::TupleUnpackInBodyReplacer");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
