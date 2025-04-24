// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// This transformation removes isolated subgraph Constant going to the Result node
// It can be case that TensorFlow can remain training artifacts in the form
// of multiple Constant nodes storing training parameter values
// We need to remove them because separate sub-graphs can solidly affect performance
class ConstToResultRemover : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::tensorflow::pass::UnsupportedConstToResultRemover");
    ConstToResultRemover() {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
