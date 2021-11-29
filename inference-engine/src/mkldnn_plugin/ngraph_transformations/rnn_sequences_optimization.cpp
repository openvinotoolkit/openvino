// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_sequences_optimization.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/variant.hpp>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::OptimizeGRUSequenceTransposes, "OptimizeGRUSequenceTransposes", 0);
NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::OptimizeLSTMSequenceTransposes, "OptimizeLSTMSequenceTransposes", 0);
NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::OptimizeRNNSequenceTransposes, "OptimizeRNNSequenceTransposes", 0);

namespace {
    int64_t getSeqAxis(const std::shared_ptr<ngraph::Node>& sequenceOp) {
        // Optimization.
        // Plug-ins support seqAxis attribute (value 1 or 0) for Seq ops, but according to the spec we don't
        // support this attribute and should insert Transpose layer before and after Seq op in TI to Sequences
        // transformation. Additional Transpose layers affect the performance, so we try to detect pattern
        // Transpose(axis_order={1,0,2}) -> Seq -> Transpose(axis_order={2,1,0,3}
        // and replace unnecessary Transpose ops with SeqIE (seqAxis = 0) to transfer value
        // of the attribute to plug-ins.
        // todo: specify seqAxis attribute for Sequence ops.
        int64_t seqAxis = 1; // default
        const auto& target_inputs = sequenceOp->output(0).get_target_inputs();
        if (target_inputs.size() == 1) {
            const auto& transpose_before = std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(sequenceOp->input_value(0).get_node_shared_ptr());
            const auto& transpose_after = std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(target_inputs.begin()->get_node()->shared_from_this());
            if (transpose_after != nullptr && transpose_before != nullptr) {
                auto order_before = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(
                        transpose_before->input_value(1).get_node_shared_ptr());
                auto order_after = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(
                        transpose_after->input_value(1).get_node_shared_ptr());
                if (order_before != nullptr && order_after != nullptr) {
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

    bool transform(const std::shared_ptr<ngraph::Node>& sequenceOp) {
        // Detect pattern: Transpose_before -> Seq -> Transpose_after
        auto seqAxis = getSeqAxis(sequenceOp);
        if (seqAxis == 0) {
            ngraph::Output<ngraph::Node> in_0 = sequenceOp->get_input_source_output(0).get_node_shared_ptr()->get_input_source_output(0);

            auto newInShape = ngraph::op::v0::Constant::create(ngraph::element::i32, ngraph::Shape{3}, sequenceOp->get_input_shape(0));
            auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(in_0, newInShape, false);
            ngraph::replace_node(sequenceOp->get_input_node_shared_ptr(0), {reshape1->output(0)});

            const auto &gruTargetInputs = sequenceOp->output(0).get_target_inputs();
            if (gruTargetInputs.empty())
                return false;
            auto transposeAfter = gruTargetInputs.begin()->get_node()->shared_from_this();

            auto newOutShape = ngraph::op::v0::Constant::create(ngraph::element::i32, ngraph::Shape{4}, transposeAfter->get_output_shape(0));
            auto reshape2 = std::make_shared<ngraph::op::v1::Reshape>(sequenceOp->output(0), newOutShape, false);
            reshape2->set_friendly_name(transposeAfter->get_friendly_name());
            ngraph::replace_node(transposeAfter, {reshape2->output(0)});
        } else {
            auto originShape = sequenceOp->get_output_shape(0);
            const auto targetInputs = sequenceOp->get_output_target_inputs(0);
            if (targetInputs.empty()) {
                return false;
            }
            auto seqOut = targetInputs.begin()->get_node()->shared_from_this();

            auto tncShape = ngraph::op::v0::Constant::create(ngraph::element::i32, ngraph::Shape{3}, {originShape[2], originShape[0], originShape[3]});
            auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(sequenceOp->output(0), tncShape, false);

            auto order = ngraph::op::v0::Constant::create(ngraph::element::i32, ngraph::Shape{3}, {1, 0, 2});
            auto transpose = std::make_shared<ngraph::op::v1::Transpose>(reshape1->output(0), order);

            auto ndtcShape = ngraph::op::v0::Constant::create(ngraph::element::i32, ngraph::Shape{4}, originShape);
            auto reshape2 = std::make_shared<ngraph::op::v1::Reshape>(transpose->output(0), ndtcShape, false);
            reshape2->set_friendly_name(sequenceOp->get_friendly_name()+".0");

            ngraph::insert_new_node_between(sequenceOp, seqOut, reshape2);
        }

        sequenceOp->get_rt_info()["seqAxis"] = std::make_shared<ngraph::VariantWrapper<int64_t>>(seqAxis);

        return true;
    }
} // namespace

MKLDNNPlugin::OptimizeGRUSequenceTransposes::OptimizeGRUSequenceTransposes() {
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto gruSequence = std::dynamic_pointer_cast<ngraph::op::v5::GRUSequence>(m.get_match_root());
        if (!gruSequence) {
            return false;
        }
        // Bidirectional cases are not supported
        if (gruSequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        return transform(gruSequence);
    };

    auto gruSequenceNgraph = ngraph::pattern::wrap_type<ngraph::op::v5::GRUSequence>();

    auto m = std::make_shared<ngraph::pattern::Matcher>(gruSequenceNgraph, "OptimizeGRUSequenceTransposes");
    this->register_matcher(m, callback);
}

MKLDNNPlugin::OptimizeRNNSequenceTransposes::OptimizeRNNSequenceTransposes() {
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto rnnSequence = std::dynamic_pointer_cast<ngraph::op::v5::RNNSequence>(m.get_match_root());
        if (!rnnSequence) {
            return false;
        }
        // Bidirectional cases are not supported
        if (rnnSequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        return transform(rnnSequence);
    };

    auto rnnSequenceNgraph = ngraph::pattern::wrap_type<ngraph::op::v5::RNNSequence>();

    auto m = std::make_shared<ngraph::pattern::Matcher>(rnnSequenceNgraph, "OptimizeRNNSequenceTransposes");
    this->register_matcher(m, callback);
}

MKLDNNPlugin::OptimizeLSTMSequenceTransposes::OptimizeLSTMSequenceTransposes() {
    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        auto lstmSequence = std::dynamic_pointer_cast<ngraph::op::v5::LSTMSequence>(m.get_match_root());
        if (!lstmSequence) {
            return false;
        }
        // Bidirectional cases are not supported
        if (lstmSequence->get_direction() == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            return false;

        return transform(lstmSequence);
    };

    auto lstmSequenceNgraph_0 = ngraph::pattern::wrap_type<ngraph::op::v0::LSTMSequence>();
    auto lstmSequenceNgraph_5 = ngraph::pattern::wrap_type<ngraph::op::v5::LSTMSequence>();
    const auto lstmSeqInputs = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{lstmSequenceNgraph_0, lstmSequenceNgraph_5});

    auto m = std::make_shared<ngraph::pattern::Matcher>(lstmSeqInputs, "OptimizeLSTMSequenceTransposes");

    this->register_matcher(m, callback);
}
