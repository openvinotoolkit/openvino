// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/gather_sinking.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/gather_sinking_binary.hpp"
#include "transformations/gather_sinking_fuse.hpp"
#include "transformations/gather_sinking_matmul.hpp"
#include "transformations/gather_sinking_reshape.hpp"
#include "transformations/gather_sinking_split.hpp"
#include "transformations/gather_sinking_transpose_reshape.hpp"
#include "transformations/gather_sinking_unary.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace ov::intel_gna::pass;

GatherSinkingGeneralForward::GatherSinkingGeneralForward() {
    MATCHER_SCOPE(GatherSinkingGeneralForward);
    add_matcher<GatherSinkingUnaryForward>();
    add_matcher<GatherSinkingBinaryForward>();
    add_matcher<GatherSinkingTransposeReshapeForward>();
    add_matcher<GatherSinkingMatmulForward>();
    add_matcher<GatherSinkingFuse>();
}

GatherSinkingGeneralBackward::GatherSinkingGeneralBackward() {
    MATCHER_SCOPE(GatherSinkingGeneralBackward);
    add_matcher<GatherSinkingUnaryBackward>();
    add_matcher<GatherSinkingBinaryBackward>();
    add_matcher<GatherSinkingTransposeReshapeBackward>();
    add_matcher<GatherSinkingReshapeBackward>();
    add_matcher<GatherSinkingSplitBackward>();
    add_matcher<GatherSinkingMatmulBackward>();
    add_matcher<GatherSinkingFuse>();
}

bool GatherSinkingGeneral::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(GatherSinkingGeneral);
    {
        ov::pass::Manager manager(get_pass_config());
        manager.register_pass<GatherSinkingGeneralForward>();
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    {
        ov::pass::Manager manager(get_pass_config());
        manager.register_pass<GatherSinkingGeneralBackward>();
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    return false;
}
