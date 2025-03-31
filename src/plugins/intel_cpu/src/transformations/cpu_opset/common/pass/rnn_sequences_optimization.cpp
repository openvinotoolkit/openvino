// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_sequences_optimization.hpp"

#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset8.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {
int64_t getSeqAxis(const std::shared_ptr<ov::Node>& sequenceOp) {
    // Optimization.
    // Plug-ins support seqAxis attribute (value 1 or 0) for Seq ops, but according to the spec we don't
    // support this attribute and should insert Transpose layer before and after Seq op in TI to Sequences
    // transformation. Additional Transpose layers affect the performance, so we try to detect pattern
    // Transpose(axis_order={1,0,2}) -> Seq -> Transpose(axis_order={2,1,0,3}
    // and replace unnecessary Transpose ops with SeqIE (seqAxis = 0) to transfer value
    // of the attribute to plug-ins.
    // todo: specify seqAxis attribute for Sequence ops.
    int64_t seqAxis = 1;  // default
    const auto& target_inputs = sequenceOp->get_output_target_inputs(0);
    if (target_inputs.size() == 1) {
        const auto transpose_before = ov::as_type_ptr<ov::opset1::Transpose>(sequenceOp->get_input_node_shared_ptr(0));
        const auto transpose_after =
            ov::as_type_ptr<ov::opset1::Transpose>(target_inputs.begin()->get_node()->shared_from_this());

        if (transpose_after && transpose_before) {
            auto order_before = ov::as_type_ptr<ov::opset1::Constant>(transpose_before->get_input_node_shared_ptr(1));
            auto order_after = ov::as_type_ptr<ov::opset1::Constant>(transpose_after->get_input_node_shared_ptr(1));
            if (order_before && order_after) {
                auto order_before_values = order_before->cast_vector<int64_t>();
                auto order_after_values = order_after->cast_vector<int64_t>();
                std::vector<int64_t> order_ref_before = {1, 0, 2};
                std::vector<int64_t> order_ref_after = {2, 1, 0, 3};
                if (order_before_values == order_ref_before && order_after_values == order_ref_after) {
                    seqAxis = 0;
                }
            }
        }
    }
    return seqAxis;
}

bool transform(const std::shared_ptr<ov::Node>& sequenceOp) {
    // Detect pattern: Transpose_before -> Seq -> Transpose_after
    auto seqAxis = getSeqAxis(sequenceOp);
    if (seqAxis == 0) {
        ov::Output<ov::Node> in_0 = sequenceOp->get_input_node_shared_ptr(0)->input_value(0);

        auto shapeBeforeTranspose = ov::op::util::make_try_fold<ov::opset1::ShapeOf>(in_0);
        auto newInShape = ov::op::util::make_try_fold<ov::opset8::Gather>(
            shapeBeforeTranspose,
            ov::opset1::Constant::create(ov::element::i32, {3}, {1, 0, 2}),
            ov::opset1::Constant::create(ov::element::i32, {}, {0}));
        auto reshape1 = std::make_shared<ov::opset1::Reshape>(in_0, newInShape, false);
        ov::copy_runtime_info(sequenceOp->get_input_node_shared_ptr(0), reshape1);
        ov::replace_node(sequenceOp->get_input_node_shared_ptr(0), reshape1);

        const auto& seqTargetInputs = sequenceOp->get_output_target_inputs(0);
        if (seqTargetInputs.empty()) {
            return false;
        }
        auto transposeAfter = seqTargetInputs.begin()->get_node()->shared_from_this();

        auto lstmOutShape = ov::op::util::make_try_fold<ov::opset1::ShapeOf>(sequenceOp->output(0));
        auto newOutShape = ov::op::util::make_try_fold<ov::opset8::Gather>(
            lstmOutShape,
            ov::opset1::Constant::create(ov::element::i32, {4}, {2, 1, 0, 3}),
            ov::opset1::Constant::create(ov::element::i32, {}, {0}));

        auto reshape2 = std::make_shared<ov::opset1::Reshape>(sequenceOp->output(0), newOutShape, false);
        reshape2->set_friendly_name(transposeAfter->get_friendly_name());
        ov::copy_runtime_info(transposeAfter, reshape2);
        ov::replace_node(transposeAfter, reshape2);
    }

    sequenceOp->get_rt_info()["seqAxis"] = seqAxis;

    return true;
}
}  // namespace

ov::intel_cpu::OptimizeGRUSequenceTransposes::OptimizeGRUSequenceTransposes() {
    MATCHER_SCOPE(OptimizeGRUSequenceTransposes);
    auto gruSequenceNgraph = ov::pass::pattern::wrap_type<ov::opset5::GRUSequence>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto gruSequence = ov::as_type_ptr<ov::opset5::GRUSequence>(m.get_match_root());
        if (!gruSequence) {
            return false;
        }
        // Bidirectional cases are not supported
        if (gruSequence->get_direction() == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
            return false;
        }

        return transform(gruSequence);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gruSequenceNgraph, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::OptimizeRNNSequenceTransposes::OptimizeRNNSequenceTransposes() {
    MATCHER_SCOPE(OptimizeRNNSequenceTransposes);
    auto rnnSequenceNgraph = ov::pass::pattern::wrap_type<ov::opset5::RNNSequence>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto rnnSequence = ov::as_type_ptr<ov::opset5::RNNSequence>(m.get_match_root());
        if (!rnnSequence) {
            return false;
        }
        // Bidirectional cases are not supported
        if (rnnSequence->get_direction() == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
            return false;
        }

        return transform(rnnSequence);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rnnSequenceNgraph, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::OptimizeLSTMSequenceTransposes::OptimizeLSTMSequenceTransposes() {
    MATCHER_SCOPE(OptimizeLSTMSequenceTransposes);
    auto lstmSequenceNgraph = ov::pass::pattern::wrap_type<ov::opset5::LSTMSequence>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto checkSequence = [](const std::shared_ptr<ov::Node>& node) {
            if (auto lstm5 = ov::as_type_ptr<ov::opset5::LSTMSequence>(node)) {
                return lstm5->get_direction() != ov::op::RecurrentSequenceDirection::BIDIRECTIONAL;
            }
            return false;
        };

        std::shared_ptr<ov::Node> lstmSequence = m.get_match_root();
        return checkSequence(lstmSequence) ? transform(lstmSequence) : false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(lstmSequenceNgraph, matcher_name);
    this->register_matcher(m, callback);
}

// NOLINTNEXTLINE(modernize-use-equals-default)
ov::intel_cpu::OptimizeSequenceTransposes::OptimizeSequenceTransposes() {
    ADD_MATCHER_FOR_THIS(OptimizeLSTMSequenceTransposes)
    ADD_MATCHER_FOR_THIS(OptimizeRNNSequenceTransposes)
    ADD_MATCHER_FOR_THIS(OptimizeGRUSequenceTransposes)
}
