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

class AlignTypesRemoval : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pytorch::pass::AlignTypesRemoval");
    AlignTypesRemoval();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
