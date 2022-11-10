// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils.hpp"

#include "snippets/pass/fq_decomposition.hpp"

namespace ngraph {
namespace snippets {
namespace utils {

auto get_non_scalar_constant_count_for_fq(const std::shared_ptr<opset1::FakeQuantize>& fq) -> size_t {
    std::vector<float> out_scales;
    std::vector<float> cl, ch, isc, ish, osc, osh;
    const bool status = ngraph::snippets::pass::FakeQuantizeDecomposition::getScalesAndShifts(fq, cl, ch, isc, ish, osc, osh);
    if (status) {
        out_scales = ngraph::snippets::pass::FakeQuantizeDecomposition::calculateScales(fq->get_output_element_type(0), cl, ch, isc, ish, osc, osh);
        if (out_scales.size() != 0) {
            return out_scales.size() != 1;
        }
    }

    const bool only_quantized = status &&
                                std::all_of(osc.cbegin(), osc.cend(),
                                    [](float val) { return val == 1.f; }) &&
                                std::all_of(osh.cbegin(), osh.cend(),
                                    [](float val) { return val == 0.f; });
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
    // Thus, after FakeQuantize decompoisition we have 6 Constants instead of original 4:
    //      ih, il (for Max/Min), isc, ish, osc, osh
    // Some of them can be scalar or non-scalar. It depends on which original 4 Constants are non-scalar
    // To sum it up, below conditions check all possible cases to calculate count of new generated non-scalars
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

auto update_out_tensor_name(std::shared_ptr<ngraph::snippets::op::Subgraph> &subgraph) -> void {
    bool not_set = true;
    for (unsigned int i = 0; i < subgraph->get_output_size() && not_set; i++) {
        for (const auto &in : subgraph->get_output_target_inputs(i)) {
            if (ov::is_type<ov::op::v0::Result>(in.get_node())) {
                const auto& body_result = subgraph->get_body()->get_output_op(i);
                const auto& body_result_input = body_result->get_input_source_output(0);
                ngraph::snippets::op::Subgraph::fill_empty_output_names(
                        subgraph->output(i), body_result_input);
                not_set = false;
                break;
            }
        }
    }
}

std::vector<size_t> get_port_layout(const std::shared_ptr<descriptor::Tensor>& tensor) {
    if (!tensor)
        return {};
    const auto& rank = tensor->get_partial_shape().rank();
    // Strictly speaking, is not a hard limitation but makes no sense, since layout can't be defined in this case
    if (rank.is_dynamic())
        throw ngraph_error("It's illegal to call get_port_layout for outputs with dynamic rank");
    auto &rt = tensor->get_rt_info();
    const auto rinfo = rt.find("Layout");
    if (rinfo != rt.end()) {
        std::vector<size_t> custom_layout(rinfo->second.as<std::vector<size_t>>());
        // Note that it can be smaller though, for example tensor shape can be prepended with 1 for scheduling purposes
        if (custom_layout.size() > rank.get_length())
            throw ngraph_error("Layout rank can't be larger than tensor rank");
        if (std::any_of(custom_layout.begin(), custom_layout.end(), [=](size_t x) {return x >= rank.get_length();}))
            throw ngraph_error("Invalid layout detected: all layout indexes must be smaller than the tensor rank");
        return custom_layout;
    } else {
        return {};
    }
}
std::vector<size_t> get_port_layout(const Output<Node>& out) {
    return get_port_layout(out.get_tensor_ptr());
}

ov::PartialShape get_port_planar_shape(const Output<Node>& out) {
    std::vector<size_t> layout = get_port_layout(out);
    const auto& tensor = out.get_tensor_ptr();
    auto tensor_shape = tensor->get_partial_shape();
    if (!layout.empty()) {
        std::vector<Dimension> reordered_shape(layout.size());
        // Note: layout[i] are guaranteed to fall inside original_shape by utils::get_port_layout(in)
        for (int i = 0; i < layout.size(); i++)
            reordered_shape[i] = tensor_shape[layout[i]];
        tensor_shape = std::move(ov::PartialShape(reordered_shape));
    }
    return tensor_shape;
}

} // namespace utils
} // namespace snippets
} // namespace ngraph
