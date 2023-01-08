// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/pass/insert_movebroadcast.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <numeric>

using namespace ngraph;

namespace {

std::shared_ptr<ngraph::Node> broadcast_node_last_dim(const ngraph::Output<ngraph::Node>& value,
                                                   const ov::Shape& target_shape, const ov::Shape& normalized_shape) {
    std::shared_ptr<ngraph::Node> broadcasted_node = value.get_node_shared_ptr();

    if (target_shape == value.get_shape()) {
        return broadcasted_node;
    }
    // Insert BroadcastMove only if the last dimension needs to be broadcasted. Higher-level dims broadcasting
    // will be handled by pointer arithmetics in TileScheduler
    if (*target_shape.rbegin() != *normalized_shape.rbegin()) {
        ov::Shape broadcasted_shape = normalized_shape;
        *broadcasted_shape.rbegin() = *target_shape.rbegin();
        broadcasted_node = std::make_shared<ngraph::snippets::op::BroadcastMove>(broadcasted_node, broadcasted_shape);
    }

    return broadcasted_node;
}


std::pair<ov::Shape, std::vector<ov::Shape>> get_numpy_broadcast_shapes(const std::vector<ov::Shape>& input_shapes) {
    ov::PartialShape target_shape =  input_shapes.front();
    for (auto i = 1; i < input_shapes.size(); i++) {
        if (!ov::PartialShape::broadcast_merge_into(target_shape, input_shapes[i], op::AutoBroadcastType::NUMPY))
            throw ngraph::ngraph_error("InsertMoveBroadcast: Failed broadcast-merge input shapes");
    }
    std::vector<ov::Shape> normalized_shapes;
    for (const auto& input : input_shapes) {
        ov::Shape padded_shape{input};
        padded_shape.insert(padded_shape.begin(), target_shape.size() - padded_shape.size(), 1);
        normalized_shapes.push_back(std::move(padded_shape));
    }

    return {target_shape.get_shape(), normalized_shapes};
}

} // namespace

ngraph::snippets::pass::InsertMoveBroadcast::InsertMoveBroadcast() {
    MATCHER_SCOPE(InsertMoveBroadcast);
    ngraph::graph_rewrite_callback callback = [this](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertMoveBroadcast")
        auto root = m.get_match_root();
        const auto& values = root->input_values();
        if (values.empty()) {
            return false;
        }

        auto is_scalar_constant = [](const ov::Output<ov::Node>& v){
            if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(v.get_node_shared_ptr())) {
                if (constant->get_shape().empty() || ngraph::shape_size(constant->get_shape()) == 1) {
                    return true;
                }
            }
            return false;
        };
        std::vector<ov::Shape> input_shapes;
        std::vector<bool> ignore_as_scalar;
        for (const auto& val : values) {
            input_shapes.emplace_back(val.get_shape());
            ignore_as_scalar.push_back(is_scalar_constant(val));
        }

        // find the output tensor's shape, then broadcast all inputs so that they are compatible with respect to the last dim
        auto bcast_shapes = get_numpy_broadcast_shapes(input_shapes);

        ngraph::OutputVector broadcasted_inputs;
        for (size_t i = 0; i < values.size(); ++i) {
            if (ignore_as_scalar[i]) {
                broadcasted_inputs.push_back(values[i]);
            } else {
                auto node = broadcast_node_last_dim(values[i], bcast_shapes.first, bcast_shapes.second[i]);
                ngraph::copy_runtime_info(root, node);
                broadcasted_inputs.push_back(node);
            }
        }

        auto new_args = ngraph::as_node_vector(broadcasted_inputs);
        for (size_t i = 0; i < new_args.size(); i++) {
            root->input(i).replace_source_output(new_args[i]->output(0));
        }
        return true;
    };

    // only numpy broadcast type is supported currently
    auto any = std::make_shared<pattern::op::Label>(pattern::any_input(),
        [](std::shared_ptr<Node> n) {
            // should add supports_auto_broadcast to SquaredDifference
            return ((ngraph::op::supports_auto_broadcast(n) || is_type<opset1::SquaredDifference>(n) || is_type<opset1::Mod>(n)) &&
                 n->get_autob().m_type == ngraph::op::AutoBroadcastType::NUMPY) || is_type<opset1::PRelu>(n); });

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(any, matcher_name), callback);
}
