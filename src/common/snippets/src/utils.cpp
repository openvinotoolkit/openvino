// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils.hpp"

#include "snippets/pass/fq_decomposition.hpp"

namespace ngraph {
namespace snippets {
namespace utils {

auto get_non_scalar_constant_count_for_fq(const std::shared_ptr<opset1::FakeQuantize>& fq) -> size_t {
    std::vector<float> cl, ch, isc, ish, osc, osh;
    const bool status = ngraph::snippets::pass::FakeQuantizeDecomposition::getScalesAndShifts(fq, cl, ch, isc, ish, osc, osh);
    bool is_optimized = false;  // The case when we can calculate only scales
    if (status) {
        const auto out_scales = ngraph::snippets::pass::FakeQuantizeDecomposition::calculateScales(fq->get_output_element_type(0), cl, ch, isc, ish, osc, osh);
        is_optimized = out_scales.size() != 0;
    }

    const bool only_quantized = is_optimized || (status &&
                                                 std::all_of(osc.cbegin(), osc.cend(),
                                                     [](float val) { return val == 1.f; }) &&
                                                 std::all_of(osh.cbegin(), osh.cend(),
                                                     [](float val) { return val == 0.f; }));
    const bool il = ngraph::shape_size(fq->input(1).get_shape()) != 1lu;
    const bool ih = ngraph::shape_size(fq->input(2).get_shape()) != 1lu;
    const bool ol = !only_quantized && ngraph::shape_size(fq->input(3).get_shape()) != 1lu;
    const bool oh = !only_quantized && ngraph::shape_size(fq->input(4).get_shape()) != 1lu;

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
std::vector<size_t> get_node_output_layout(const std::shared_ptr<Node>& node) {
    return get_node_output_layout(node.get());
}
std::vector<size_t> get_node_output_layout(const Node* node) {
    if (!node)
        return {};
    if (node->is_dynamic())
        throw ngraph_error("It's illegal to call get_node_output_layout for dynamic nodes");
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("Layout");
    if (rinfo != rt.end()) {
        std::vector<size_t> layout(rinfo->second.as<std::vector<size_t>>());
        // This might be a little costy, but still useful sanity check. Remove if proved to be unacceptably heavy.
        std::set<size_t> unique_elements(layout.begin(), layout.end());
        if (unique_elements.size() < layout.size())
            throw ngraph_error("Layout must contain only unique dimension indexes");
        return layout;
    } else {
        return {};
    }
}

ov::PartialShape get_reordered_planar_shape(const ov::PartialShape& shape, const std::vector<size_t>& layout) {
    if (layout.empty())
        return shape;
    std::vector<Dimension> reordered_shape(layout.size());
    if (shape.rank().is_dynamic())
        throw ngraph_error("get_reordered_planar_shape can't be called for outputs with dynamic rank");
    const size_t rank = shape.rank().get_length();
    if (layout.size() > rank)
        throw ngraph_error("Layout rank can't be larger than tensor rank");
    // Note that it can be smaller though, for example tensor shape can be prepended with 1 for scheduling purposes
    if (std::any_of(layout.begin(), layout.end(), [=](size_t x) {return x >= rank;}))
        throw ngraph_error("Invalid layout detected: all layout indexes must be smaller than the tensor rank");
    for (size_t i = 0; i < layout.size(); i++)
        reordered_shape[i] = shape[layout[i]];
    return reordered_shape;
}

ov::PartialShape get_port_planar_shape(const Output<Node>& out) {
    std::vector<size_t> layout = get_node_output_layout(out.get_node_shared_ptr());
    const auto& tensor = out.get_tensor_ptr();
    if (!tensor)
        throw ngraph_error("get_port_planar_shape can't be called for an uninitialized output tensor");
    auto tensor_shape = tensor->get_partial_shape();
    return get_reordered_planar_shape(tensor_shape, layout);
}

} // namespace utils
} // namespace snippets
} // namespace ngraph
