// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// This transformation fuses Switch-Merge sub-graphs into If operation
// After control flow markers propagation, Switch and Merge nodes
// can contain CF markers so that Switch and Merge nodes with the same marker values
// in new_markers and eliminated markers will be fused into If operation where
// then_branch is represented by a graph below output_true branch of Switch and else_branch is by output_false.
// Moreover, several Switch nodes may have the same new_markers that means the resulted If will have several inputs.
// Merge nodes can have the same eliminated markers that means the fused If will have several outputs.
class SwitchMergeResolver : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::tensorflow::SwitchMergeResolver");
    SwitchMergeResolver() = default;

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
