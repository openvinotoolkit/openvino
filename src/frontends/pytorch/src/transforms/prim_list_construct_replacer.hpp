// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

class DecomposeListResults : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::pytorch::pass::DecomposeListResults");
    bool run_on_model(const std::shared_ptr<Model>& model) override;
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
