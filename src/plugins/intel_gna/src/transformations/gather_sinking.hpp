// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace intel_gna {
namespace pass {

class GatherSinkingGeneralForward : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("GatherSinkingGeneralForward", "0");
    GatherSinkingGeneralForward();
};

class GatherSinkingGeneralBackward : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("GatherSinkingGeneralBackward", "0");
    GatherSinkingGeneralBackward();
};

class GatherSinkingGeneral : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("GatherSinkingGeneral", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
