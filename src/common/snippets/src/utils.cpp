// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils.hpp"

#include "snippets/pass/fq_decomposition.hpp"
#include "openvino/core/rt_info.hpp"
#include "snippets/op/subgraph.hpp"


namespace ov {
namespace snippets {
namespace utils {

namespace {
template<typename Shape>
void ordered_shape(const Shape& shape, const std::vector<size_t>& layout, bool is_forward, Shape& reordered_shape) {
    for (size_t i = 0; i < layout.size(); i++) {
        OPENVINO_ASSERT(layout[i] < shape.size(), "layout index is greater than the shape size");
        const auto src_idx = is_forward ? layout[i] : i;
        const auto dst_idx = is_forward ? i : layout[i];
        reordered_shape[dst_idx] = shape[src_idx];
    }
}

// Note:
//   - If `is_forward` is True, `result shape` is ordered `shape` by `layout`
//   - If `is_forward` is False, `result shape` is original shape to which the `layout` was applied
ov::PartialShape get_pshape(const ov::PartialShape& shape, const std::vector<size_t>& layout, bool is_forward) {
    if (layout.empty())
        return shape;
    ov::PartialShape reordered_shape(std::vector<Dimension>(layout.size()));
    if (shape.rank().is_dynamic())
        OPENVINO_THROW("get_reordered_planar_shape can't be called for outputs with dynamic rank");
    const size_t rank = shape.rank().get_length();
    if (layout.size() > rank)
        OPENVINO_THROW("Layout rank can't be larger than tensor rank");
    // Note that it can be smaller though, for example tensor shape can be prepended with 1 for scheduling purposes
    if (std::any_of(layout.begin(), layout.end(), [=](size_t x) {return x >= rank;}))
        OPENVINO_THROW("Invalid layout detected: all layout indexes must be smaller than the tensor rank");
    ordered_shape(shape, layout, is_forward, reordered_shape);
    return reordered_shape;
}
}  // namespace

auto get_non_scalar_constant_count_for_fq(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) -> size_t {
    std::vector<float> cl, ch, isc, ish, osc, osh;
    const bool status = ov::snippets::pass::FakeQuantizeDecomposition::getScalesAndShifts(fq, cl, ch, isc, ish, osc, osh);
    bool is_optimized = false;  // The case when we can calculate only scales
    if (status) {
        const auto out_scales = ov::snippets::pass::FakeQuantizeDecomposition::calculateScales(fq->get_output_element_type(0), cl, ch, isc, ish, osc, osh);
        is_optimized = out_scales.size() != 0;
    }

    const bool only_quantized = is_optimized || (status &&
                                                 std::all_of(osc.cbegin(), osc.cend(),
                                                     [](float val) { return val == 1.f; }) &&
                                                 std::all_of(osh.cbegin(), osh.cend(),
                                                     [](float val) { return val == 0.f; }));
    const bool il = ov::shape_size(fq->input(1).get_shape()) != 1lu;
    const bool ih = ov::shape_size(fq->input(2).get_shape()) != 1lu;
    const bool ol = !only_quantized && ov::shape_size(fq->input(3).get_shape()) != 1lu;
    const bool oh = !only_quantized && ov::shape_size(fq->input(4).get_shape()) != 1lu;

    // FakeQuantize decompoisition has the folowwing formula:
    //      round(x * (levels-1) / (ih - il) - il * (levels-1) / (ih - il)) * (oh - ol) / (levels-1) + ol
    // After the decomposition there is call of ConstantsFolding pass that generates new Constants:
    //      - isc := (levels-1) / (ih - il)
    //      - ish := -il * isc
    //      - osc := (oh - ol) / (levels-1)
    //      - osh := ol
    // New formula:
    //      round(x * isc + ish) * osc + osh
    // Thus, after FakeQuantize decompoisition we have:
    //      - If it's non optimized FQ, 6 Constants instead of original 4:
    //              ih, il (for Max/Min), isc, ish, osc, osh
    //      - If it's optimized FQ, 3 Constants instead of original 4:
    //              ih, il (for Max/Min), isc
    // Some of them can be scalar or non-scalar. It depends on which original 4 Constants are non-scalar
    // To sum it up, below conditions check all possible cases to calculate count of new generated non-scalars
    if (is_optimized) {
        if (il && ih)
            return 3;
        else if (il || ih)
            return 2;
        return 0;
    } else {
        if (ol && il && ih)
            return 6;
        else if ((ol && (il || ih)) || (il && ih && oh))
            return 5;
        else if ((il && oh) || (ih && oh) || (il && ih))
            return 4;
        else if (il || ih)
            return 3;
        else if (ol)
            return 2;
        else if (oh)
            return 1;
        return 0;
    }
}

void broadcast_merge_dim(size_t& dst, const size_t& d1, const size_t& d2) {
    if (d1 == d2 || d1 == 1 || is_dynamic_value(d2)) {
        dst = d2;
    } else if (d2 == 1 || is_dynamic_value(d1)) {
        dst = d1;
    } else {
        OPENVINO_THROW("Failed to broadcast dims: ", d1, " and ", d2);
    }
}

VectorDims pshape_to_vdims(const PartialShape& pshape) {
    VectorDims result;
    result.reserve(pshape.size());
    for (const auto& d : pshape)
        result.push_back(d.is_dynamic() ? get_dynamic_value<VectorDims::value_type>() : d.get_length());
    // Note: PartialShape could be empty which designates scalar value. However, Scalars are represented as {1} in Snippets
    return result.empty() ? VectorDims {1} : result;
}

ov::PartialShape vdims_to_pshape(const VectorDims& vdims) {
    ov::PartialShape result;
    result.reserve(vdims.size());
    for (const auto& v : vdims)
        result.push_back(!is_dynamic_value(v) ? Dimension(static_cast<Dimension::value_type>(v))
                                              : Dimension());
    return result;
}

size_t get_dim_idx(const lowered::ExpressionPort& port, size_t dim_idx) {
    const auto layout = port.get_descriptor_ptr()->get_layout();
    if (port.get_type() == lowered::ExpressionPort::Type::Input)
        return utils::get_input_dim_idx(layout, dim_idx);
    else if (port.get_type() == lowered::ExpressionPort::Type::Output)
        return utils::get_output_dim_idx(layout, dim_idx);
    else
        OPENVINO_THROW("Unsupported type of expression port");
    return 0;
}

ov::PartialShape get_planar_pshape(const ov::PartialShape& shape, const std::vector<size_t>& order) {
    return get_pshape(shape, order, true);
}
ov::PartialShape get_preordered_pshape(const ov::PartialShape& shape, const std::vector<size_t>& order) {
    return get_pshape(shape, order, false);
}

ov::PartialShape get_planar_pshape(const Input<Node>& in) {
    const auto& port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(in);
    return get_planar_pshape(ov::Shape{port->get_shape()}, port->get_layout());
}
ov::PartialShape get_preordered_pshape(const Output<Node>& out) {
    const auto& port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(out);
    return get_preordered_pshape(ov::Shape{port->get_shape()}, port->get_layout());
}

VectorDims get_planar_vdims(const VectorDims& shape, const std::vector<size_t>& order) {
    VectorDims reordered_shape(order.size());
    ordered_shape(shape, order, true, reordered_shape);
    return reordered_shape;
}
VectorDims get_preordered_vdims(const VectorDims& shape, const std::vector<size_t>& order) {
    VectorDims reordered_shape(order.size());
    ordered_shape(shape, order, false, reordered_shape);
    return reordered_shape;
}

VectorDims get_planar_vdims(const snippets::lowered::ExpressionPort& expr_port) {
    OPENVINO_ASSERT(expr_port.get_type() == snippets::lowered::ExpressionPort::Type::Input, "get_planar_vdims expects Expression Input port");
    return get_planar_vdims(expr_port.get_descriptor_ptr()->get_shape(), expr_port.get_descriptor_ptr()->get_layout());
}
VectorDims get_preordered_vdims(const snippets::lowered::ExpressionPort& expr_port) {
    OPENVINO_ASSERT(expr_port.get_type() == snippets::lowered::ExpressionPort::Type::Output, "get_preordered_vdims expects Expression Output port");
    return get_preordered_vdims(expr_port.get_descriptor_ptr()->get_shape(), expr_port.get_descriptor_ptr()->get_layout());
}

std::vector<lowered::ExpressionPtr> get_first_child_shape_infer_expr_seq(const lowered::ExpressionPtr& start_expr) {
    auto get_first_shape_infer_expr = [](const std::set<lowered::ExpressionPort>& consumers) -> lowered::ExpressionPtr {
        for (auto it = consumers.begin(); it != consumers.end(); ++it) {
            auto expr = it->get_expr();
            if (op::Subgraph::is_shape_infer_op(expr->get_node())) {
                return expr;
            }
        }
        return nullptr;
    };
    std::vector<lowered::ExpressionPtr> shape_infer_exprs;
    if (op::Subgraph::is_shape_infer_op(start_expr->get_node())) {
        OPENVINO_ASSERT(start_expr->get_input_port_connector(0)->get_consumers().size() == 1, "Shape infer ops are supposed to be the only consumer.");
        shape_infer_exprs.push_back(start_expr);
    }
    if (start_expr->get_output_count() == 0)
        return shape_infer_exprs;
    auto output_consumers = start_expr->get_output_port_connector(0)->get_consumers();
    while (auto shape_infer_child = get_first_shape_infer_expr(output_consumers)) {
        OPENVINO_ASSERT(output_consumers.size() == 1, "Shape infer ops are supposed to be the only consumer.");
        shape_infer_exprs.push_back(shape_infer_child);
        if (shape_infer_child->get_output_count() == 0)
            break;
        output_consumers = shape_infer_child->get_output_port_connector(0)->get_consumers();
    }
    return shape_infer_exprs;
}

std::vector<lowered::ExpressionPtr> get_first_parent_shape_infer_expr_seq(const lowered::ExpressionPtr& start_expr) {
    std::vector<lowered::ExpressionPtr> shape_infer_exprs;
    auto current_exp = start_expr;
    if (op::Subgraph::is_shape_infer_op(current_exp->get_node())) {
        OPENVINO_ASSERT(current_exp->get_input_port_connector(0)->get_consumers().size() == 1, "Shape infer ops are supposed to be the only consumer.");
        shape_infer_exprs.push_back(current_exp);
    }
    if (current_exp->get_input_count() == 0)
        return shape_infer_exprs;
    auto input = current_exp->get_input_port_connector(0);
    auto first_parent = input->get_source().get_expr();
    while (op::Subgraph::is_shape_infer_op(first_parent->get_node())) {
        shape_infer_exprs.push_back(first_parent);
        current_exp = first_parent;
        if (current_exp->get_input_count() == 0)
            break;
        input = current_exp->get_input_port_connector(0);
        first_parent = input->get_source().get_expr();
        if (!ov::is_type<snippets::op::Store>(first_parent->get_node())) {
            // there are maybe some loopEnd consumers of store as well for loop code gen purpose
            OPENVINO_ASSERT(input->get_consumers().size() == 1, "Shape infer ops are supposed to be the only consumer if it doesn't consume a store ops.");
        }
    }
    return shape_infer_exprs;
}

std::shared_ptr<ov::Node> get_leaf_node_of_first_child_shape_infer_seq(const std::shared_ptr<ov::Node>& start_node)  {
    auto get_first_shape_infer_node = [](const std::set<ov::Input<ov::Node>>& consumers) -> std::shared_ptr<ov::Node> {
        for (auto it = consumers.begin(); it != consumers.end(); ++it) {
            auto node = it->get_node()->shared_from_this();
            if (op::Subgraph::is_shape_infer_op(node)) {
                return node;
            }
        }
        return nullptr;
    };
    std::shared_ptr<ov::Node> leaf_node = nullptr;
    if (op::Subgraph::is_shape_infer_op(start_node)) {
        OPENVINO_ASSERT(start_node->input(0).get_source_output().get_target_inputs().size() == 1, "Shape infer ops are supposed to be the only consumer.");
        leaf_node = start_node;
    }
    if (start_node->get_output_size() == 0)
        return leaf_node;
    auto output_consumers = start_node->get_output_target_inputs(0);
    while (auto first_child = get_first_shape_infer_node(output_consumers)) {
        OPENVINO_ASSERT(output_consumers.size() == 1, "Shape infer ops are supposed to be the only consumer.");
        leaf_node = first_child;
        if (leaf_node->get_output_size() == 0)
            break;
        output_consumers = leaf_node->get_output_target_inputs(0);
    }
    return leaf_node;
}

std::shared_ptr<ov::Node> get_leaf_node_of_first_parent_shape_infer_seq(const std::shared_ptr<ov::Node>& start_node) {
    std::shared_ptr<ov::Node> leaf_node = nullptr;
    if (op::Subgraph::is_shape_infer_op(start_node)) {
        OPENVINO_ASSERT(start_node->input(0).get_source_output().get_target_inputs().size() == 1, "Shape infer ops are supposed to be the only consumer.");
        leaf_node = start_node;
    }
    if (start_node->get_input_size() == 0)
        return leaf_node;
    auto first_parent = start_node->get_input_node_shared_ptr(0);
    while (op::Subgraph::is_shape_infer_op(first_parent)) {
        OPENVINO_ASSERT(first_parent->input(0).get_source_output().get_target_inputs().size() == 1, "Shape infer ops are supposed to be the only consumer.");
        leaf_node = first_parent;
        if (leaf_node->get_input_size() == 0)
            break;
        first_parent = leaf_node->get_input_node_shared_ptr(0);
    }
    return leaf_node;
}

} // namespace utils
} // namespace snippets
} // namespace ov
