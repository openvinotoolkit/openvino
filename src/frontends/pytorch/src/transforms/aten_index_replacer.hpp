// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/pytorch/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

// This transformation replaces pattern prim::ListConstruct->aten::index
class PYTORCH_API AtenIndexToSelect : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pytorch::pass::AtenIndexToSelect");
    AtenIndexToSelect();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
