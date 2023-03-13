// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_sinking_general.hpp"

#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "itt.hpp"
#include "transformations/common_optimizations/transpose_sinking_binary.hpp"
#include "transformations/common_optimizations/transpose_sinking_concat.hpp"
#include "transformations/common_optimizations/transpose_sinking_data_movement.hpp"
#include "transformations/common_optimizations/transpose_sinking_fuse.hpp"
#include "transformations/common_optimizations/transpose_sinking_interpolate.hpp"
#include "transformations/common_optimizations/transpose_sinking_reduction.hpp"
#include "transformations/common_optimizations/transpose_sinking_split.hpp"
#include "transformations/common_optimizations/transpose_sinking_unary.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::TransposeSinkingGeneralForward::TransposeSinkingGeneralForward() {
    MATCHER_SCOPE(TransposeSinkingGeneralForward);
    add_matcher<ov::pass::TransposeSinkingUnaryForward>();
    add_matcher<ov::pass::TransposeSinkingBinaryForward>();
    add_matcher<ov::pass::TransposeSinkingConcatForward>();
    add_matcher<ov::pass::TransposeSinkingSplitForward>();
    add_matcher<ov::pass::TransposeSinkingDataMovementForward>();
    add_matcher<ov::pass::TransposeSinkingReductionForward>();
    add_matcher<ov::pass::TransposeSinkingInterpolateForward>();
    add_matcher<ov::pass::TransposeSinkingFuse>();
}

ov::pass::TransposeSinkingGeneralBackward::TransposeSinkingGeneralBackward() {
    MATCHER_SCOPE(TransposeSinkingGeneralBackward);
    add_matcher<ov::pass::TransposeSinkingUnaryBackward>();
    add_matcher<ov::pass::TransposeSinkingBinaryBackward>();
    add_matcher<ov::pass::TransposeSinkingConcatBackward>();
    add_matcher<ov::pass::TransposeSinkingSplitBackward>();
    add_matcher<ov::pass::TransposeSinkingDataMovementBackward>();
    add_matcher<ov::pass::TransposeSinkingReductionBackward>();
    add_matcher<ov::pass::TransposeSinkingInterpolateBackward>();
    add_matcher<ov::pass::TransposeSinkingFuse>();
}

bool ov::pass::TransposeSinkingGeneral::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(TransposeSinkingGeneral);
    {
        ov::pass::Manager manager(get_pass_config());
        manager.register_pass<ov::pass::TransposeSinkingGeneralForward>();
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    {
        ov::pass::Manager manager(get_pass_config());
        manager.register_pass<ov::pass::TransposeSinkingGeneralBackward>();
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.run_passes(f);
    }

    return false;
}
