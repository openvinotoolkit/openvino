// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
namespace pass {

// This transformation removes isolated subgraph Unsupported constant going to the Result node
class UnsupportedConstToResultRemover : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::tensorflow::pass::UnsupportedConstToResultRemover");
    UnsupportedConstToResultRemover() {}

    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
