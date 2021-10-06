// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remarks.hpp"
#include "itt.hpp"

#include "snippets/pass/insert_movebroadcast.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <numeric>

using namespace ngraph;

static std::shared_ptr<ngraph::Node> numpy_broadcast_node(const ngraph::Output<ngraph::Node>& value,
    const ngraph::Shape& output_shape, const ngraph::Shape& source_shape) {
    std::shared_ptr<ngraph::Node> broadcasted_node = value.get_node_shared_ptr();

    if (output_shape == value.get_shape()) {
        return broadcasted_node;
    }

    NGRAPH_CHECK(source_shape.size() == output_shape.size(),
                    "Ranks of source_shape and output_shape dont match: ",
                    source_shape.size(),
                    " vs ",
                    output_shape.size());

    ngraph::AxisVector broadcast_axes;
    ngraph::Shape squeezed_shape;
    for (size_t index = 0; index < output_shape.size(); ++index) {
        if (source_shape.at(index) == 1 && output_shape.at(index) != 1) {
            broadcast_axes.push_back(index);
        } else {
            squeezed_shape.push_back(source_shape.at(index));
        }
    }

    remark(2) << "Insert explicit broadcast " << value.get_node()->get_type_name()
    << " " << broadcast_axes << " " << broadcasted_node->get_shape() << " -> " << output_shape << std::endl;

    // it shouldn't be a probrem for now since we don't consider StridedSlice and Broadcast here
    if (auto constant = ngraph::as_type_ptr<ngraph::opset1::Constant>(broadcasted_node)) {
        if (constant->get_shape() == ngraph::Shape() || ngraph::shape_size(constant->get_shape()) == 1) {
            remark(2) << "Insert explicit broadcast " << value.get_node()->get_type_name()
                       << " to scalar constant " << constant->get_shape() << " -- aborting!" << std::endl;

            return broadcasted_node;
        }
    }

    if (auto constant = ngraph::as_type_ptr<ngraph::snippets::op::Scalar>(broadcasted_node)) {
        if (constant->get_shape() == ngraph::Shape() || ngraph::shape_size(constant->get_shape()) == 1) {
            remark(2) << "Insert explicit broadcast " << value.get_node()->get_type_name()
                       << " to scalar constant " << constant->get_shape() << " -- aborting!" << std::endl;

            return broadcasted_node;
        }
    }

    if (!broadcast_axes.empty()) {
        // ShapeOf
        broadcasted_node = std::make_shared<ngraph::snippets::op::BroadcastMove>(broadcasted_node, output_shape);
    }

    return broadcasted_node;
}

static ngraph::Shape calculate_broadcast_shape(ngraph::Shape lhs_shape, ngraph::Shape rhs_shape) {
    ngraph::Shape result;
    auto lhs_rank = lhs_shape.size();
    auto rhs_rank = rhs_shape.size();
    auto max_rank = std::max(lhs_rank, rhs_rank);

    // left-pad the lhs_shape with ones
    lhs_shape.insert(begin(lhs_shape), max_rank - lhs_rank, 1);
    // left-pad the rhs_shape with ones
    rhs_shape.insert(begin(rhs_shape), max_rank - rhs_rank, 1);

    for (size_t index = 0; index < max_rank; ++index) {
        size_t lhs_dim = lhs_shape.at(index);
        size_t rhs_dim = rhs_shape.at(index);

        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
            throw ngraph::ngraph_error("incompatible shapes");
        }

        result.push_back(std::max(lhs_dim, rhs_dim));
    }
    return result;
}

std::pair<ngraph::Shape, std::vector<ngraph::Shape>> get_numpy_broadcast_shapes(const std::vector<ngraph::Shape>& input_shapes) {
    ngraph::Shape target_shape = std::accumulate(begin(input_shapes), end(input_shapes), ngraph::Shape{}, calculate_broadcast_shape);

    std::vector<ngraph::Shape> full_shapes;
    for (const ngraph::Shape& input : input_shapes) {
        ngraph::Shape padded_shape{input};
        padded_shape.insert(begin(padded_shape), target_shape.size() - padded_shape.size(), 1);
        full_shapes.push_back(move(padded_shape));
    }

    return {target_shape, full_shapes};
}

auto reset_broacast_config(const std::shared_ptr<ngraph::Node>& op) -> void {
    using namespace ngraph;

    bool is_scalar = false;
    for (auto input : op->inputs()) {
        if (input.get_shape() == Shape() || ngraph::shape_size(input.get_shape()) == 1) {
            is_scalar = true;
        }
    }

    if (!is_scalar) {
        if (auto binary = std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(op)) {
            binary->set_autob(ngraph::op::AutoBroadcastSpec::NONE);
        } else if (auto binary = std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseComparison>(op)) {
            binary->set_autob(ngraph::op::AutoBroadcastSpec::NONE);
        } else if (auto binary = std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseLogical>(op)) {
            binary->set_autob(ngraph::op::AutoBroadcastSpec::NONE);
        }
    }
}

// adds explicit broadcasts if needed
// ToDO: this indeed make model not reshapable, need to come up with more clever way to insert fake broadcast,
// well on the other hand, if we replace scalar constant with Scalar op / or ShapeOf, we could have broadcasts that are reshapable
// TODO: generate FakeBroadcast if and only if broadcast is done by w dimension
ngraph::snippets::pass::InsertMoveBroadcast::InsertMoveBroadcast() {
    MATCHER_SCOPE(InsertMoveBroadcast);
    ngraph::graph_rewrite_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto root = m.get_match_root();
        const auto& values = root->input_values();
        if (values.empty()) {
            return false;
        }

        std::vector<ngraph::Shape> input_shapes;
        for (const auto& input : values) {
            input_shapes.push_back(input.get_shape());
        }

        // find the output tensor's shape, then broadcast all inputs so that they are compatible
        auto bcast_shapes = get_numpy_broadcast_shapes(input_shapes);

        ngraph::OutputVector broadcasted_inputs;
        for (size_t i = 0; i < values.size(); ++i) {
            auto node = numpy_broadcast_node(values[i], bcast_shapes.first, bcast_shapes.second[i]);
            ngraph::copy_runtime_info(root, node);
            broadcasted_inputs.push_back(node);
        }

        auto new_args = ngraph::as_node_vector(broadcasted_inputs);
        for (size_t i = 0; i < new_args.size(); i++) {
            root->input(i).replace_source_output(new_args[i]->output(0));
        }

        reset_broacast_config(root);

        return true;
    };

    // only numpy broadcast type is supported currently
    auto any = std::make_shared<pattern::op::Label>(pattern::any_input(),
        [](std::shared_ptr<Node> n) {
            // should add supports_auto_broadcast to SquaredDifference
            return (ngraph::op::supports_auto_broadcast(n) || !!as_type_ptr<opset1::SquaredDifference>(n) || !!as_type_ptr<opset1::Mod>(n))
                && n->get_autob().m_type == ngraph::op::AutoBroadcastType::NUMPY; });

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(any), callback);
}