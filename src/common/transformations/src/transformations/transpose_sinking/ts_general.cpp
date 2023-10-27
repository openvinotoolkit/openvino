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
#include "transformations/common_optimizations/enable_shapeof_constant_folding.hpp"
#include "transformations/transpose_sinking/ts_binary.hpp"
#include "transformations/transpose_sinking/ts_concat.hpp"
#include "transformations/transpose_sinking/ts_cumsum.hpp"
#include "transformations/transpose_sinking/ts_data_movement.hpp"
#include "transformations/transpose_sinking/ts_fuse.hpp"
#include "transformations/transpose_sinking/ts_gather.hpp"
#include "transformations/transpose_sinking/ts_interpolate.hpp"
#include "transformations/transpose_sinking/ts_reduction.hpp"
#include "transformations/transpose_sinking/ts_reset_no_sinking_attribute.hpp"
#include "transformations/transpose_sinking/ts_shape_of.hpp"
#include "transformations/transpose_sinking/ts_slice.hpp"
#include "transformations/transpose_sinking/ts_split.hpp"
#include "transformations/transpose_sinking/ts_squeeze.hpp"
#include "transformations/transpose_sinking/ts_tile.hpp"
#include "transformations/transpose_sinking/ts_unary.hpp"
#include "transformations/transpose_sinking/ts_unsqueeze.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::transpose_sinking;

TSGeneralForward::TSGeneralForward() {
    MATCHER_SCOPE(TSGeneralForward);
    ADD_MATCHER(this, TSUnaryForward);
    ADD_MATCHER(this, TSBinaryForward);
    ADD_MATCHER(this, TSConcatForward);
    ADD_MATCHER(this, TSSplitForward);
    ADD_MATCHER(this, TSDataMovementForward);
    ADD_MATCHER(this, TSReductionForward);
    ADD_MATCHER(this, TSSqueezeForward);
    ADD_MATCHER(this, TSUnsqueezeForward);
    ADD_MATCHER(this, TSInterpolateForward);
    ADD_MATCHER(this, TSSliceForward);
    ADD_MATCHER(this, TSGatherForward);
    ADD_MATCHER(this, TSShapeOfForward);
    ADD_MATCHER(this, TSCumSumForward);
    ADD_MATCHER(this, TSTileForward);
    ADD_MATCHER(this, TSFuse);
}

TSGeneralBackward::TSGeneralBackward() {
    MATCHER_SCOPE(TSGeneralBackward);
    ADD_MATCHER(this, TSUnaryBackward);
    ADD_MATCHER(this, TSUnaryBackward);
    ADD_MATCHER(this, TSBinaryBackward);
    ADD_MATCHER(this, TSConcatBackward);
    ADD_MATCHER(this, TSSplitBackward);
    ADD_MATCHER(this, TSDataMovementBackward);
    ADD_MATCHER(this, TSReductionBackward);
    ADD_MATCHER(this, TSSqueezeBackward);
    ADD_MATCHER(this, TSUnsqueezeBackward);
    ADD_MATCHER(this, TSInterpolateBackward);
    ADD_MATCHER(this, TSSliceBackward);
    ADD_MATCHER(this, TSGatherBackward);
    ADD_MATCHER(this, TSCumSumBackward);
    ADD_MATCHER(this, TSTileBackward);
    ADD_MATCHER(this, TSFuse);
}

bool TSGeneral::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(TSGeneral);
    {
        Manager manager(get_pass_config());
        manager.register_pass<DisableShapeOfConstantFolding>(/* check_shape */ false);
        manager.register_pass<TSGeneralForward>();
        manager.register_pass<ConstantFolding>();
        manager.run_passes(f);
    }

    {
        Manager manager(get_pass_config());
        manager.register_pass<DisableShapeOfConstantFolding>(/* check_shape */ false);
        manager.register_pass<TSGeneralBackward>();
        manager.register_pass<ConstantFolding>();
        manager.register_pass<TSResetNoSinkingAttribute>();
        manager.register_pass<EnableShapeOfConstantFolding>();
        manager.run_passes(f);
    }

    return false;
}
