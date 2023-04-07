// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_general.hpp"

#include "itt.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/disable_shapeof_constant_folding.hpp"
#include "transformations/transpose_sinking/ts_binary.hpp"
#include "transformations/transpose_sinking/ts_concat.hpp"
#include "transformations/transpose_sinking/ts_data_movement.hpp"
#include "transformations/transpose_sinking/ts_fuse.hpp"
#include "transformations/transpose_sinking/ts_interpolate.hpp"
#include "transformations/transpose_sinking/ts_reduction.hpp"
#include "transformations/transpose_sinking/ts_slice.hpp"
#include "transformations/transpose_sinking/ts_split.hpp"
#include "transformations/transpose_sinking/ts_squeeze.hpp"
#include "transformations/transpose_sinking/ts_unary.hpp"
#include "transformations/transpose_sinking/ts_unsqueeze.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::transpose_sinking;

TSGeneralForward::TSGeneralForward() {
    MATCHER_SCOPE(TSGeneralForward);
    add_matcher<TSUnaryForward>();
    add_matcher<TSBinaryForward>();
    add_matcher<TSConcatForward>();
    add_matcher<TSSplitForward>();
    add_matcher<TSDataMovementForward>();
    add_matcher<TSReductionForward>();
    add_matcher<TSSqueezeForward>();
    add_matcher<TSUnsqueezeForward>();
    add_matcher<TSInterpolateForward>();
    add_matcher<TSSliceForward>();
    add_matcher<TSFuse>();
}

TSGeneralBackward::TSGeneralBackward() {
    MATCHER_SCOPE(TSGeneralBackward);
    add_matcher<TSUnaryBackward>();
    add_matcher<TSBinaryBackward>();
    add_matcher<TSConcatBackward>();
    add_matcher<TSSplitBackward>();
    add_matcher<TSDataMovementBackward>();
    add_matcher<TSReductionBackward>();
    add_matcher<TSSqueezeBackward>();
    add_matcher<TSUnsqueezeBackward>();
    add_matcher<TSInterpolateBackward>();
    add_matcher<TSSliceBackward>();
    add_matcher<TSFuse>();
}

bool TSGeneral::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(TSGeneral);
    {
        Manager manager(get_pass_config());
        manager.register_pass<DisableShapeOfConstantFolding>();
        manager.register_pass<TSGeneralForward>();
        manager.register_pass<ConstantFolding>();
        manager.run_passes(f);
    }

    {
        Manager manager(get_pass_config());
        manager.register_pass<DisableShapeOfConstantFolding>();
        manager.register_pass<TSGeneralBackward>();
        manager.register_pass<ConstantFolding>();
        manager.run_passes(f);
    }

    return false;
}
