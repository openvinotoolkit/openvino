// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils.hpp"

#include "snippets/pass/fq_decomposition.hpp"
#include "openvino/core/rt_info.hpp"


namespace ov {
namespace snippets {
namespace utils {

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
    const bool il = ov::shape_size(fq->input(1).get_partial_shape().to_shape()) != 1lu;
    const bool ih = ov::shape_size(fq->input(2).get_partial_shape().to_shape()) != 1lu;
    const bool ol = !only_quantized && ov::shape_size(fq->input(3).get_partial_shape().to_shape()) != 1lu;
    const bool oh = !only_quantized && ov::shape_size(fq->input(4).get_partial_shape().to_shape()) != 1lu;

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

ov::PartialShape get_planar_pshape(const ov::PartialShape& shape, const std::vector<size_t>& layout) {
    if (layout.empty())
        return shape;
    std::vector<Dimension> reordered_shape(layout.size());
    if (shape.rank().is_dynamic())
        OPENVINO_THROW("get_reordered_planar_shape can't be called for outputs with dynamic rank");
    const size_t rank = shape.rank().get_length();
    if (layout.size() > rank)
        OPENVINO_THROW("Layout rank can't be larger than tensor rank");
    // Note that it can be smaller though, for example tensor shape can be prepended with 1 for scheduling purposes
    if (std::any_of(layout.begin(), layout.end(), [=](size_t x) {return x >= rank;}))
        OPENVINO_THROW("Invalid layout detected: all layout indexes must be smaller than the tensor rank");
    for (size_t i = 0; i < layout.size(); i++)
        reordered_shape[i] = shape[layout[i]];
    return reordered_shape;
}

VectorDims pshape_to_vdims(const PartialShape& pshape) {
    VectorDims result;
    result.reserve(pshape.size());
    for (const auto& d : pshape)
        result.push_back(d.is_dynamic() ? IShapeInferSnippets::DYNAMIC_DIMENSION : d.get_length());
    return result;
}

ov::PartialShape vdims_to_pshape(const VectorDims& vdims) {
    ov::PartialShape result;
    result.reserve(vdims.size());
    for (const auto& v : vdims)
        result.push_back(v != IShapeInferSnippets::DYNAMIC_DIMENSION ?
                         Dimension(static_cast<Dimension::value_type>(v)) :
                         Dimension());
    return result;
}

ov::PartialShape get_planar_pshape(const Input<Node>& in) {
    const auto& port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(in);
    return utils::get_planar_pshape(ov::Shape{port->get_shape()}, port->get_layout());
}

ov::PartialShape get_planar_pshape(const Output<Node>& out) {
    const auto& port = snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(out);
    return utils::get_planar_pshape(ov::Shape{port->get_shape()}, port->get_layout());
}

VectorDims get_planar_vdims(const VectorDims& shape, const std::vector<size_t>& layout) {
    VectorDims reordered_shape(shape.size());
    for (size_t i = 0; i < layout.size(); i++) {
        OPENVINO_ASSERT(layout[i] < shape.size(), "get_planar_vdims: layout index is greater than the shape size");
        reordered_shape[i] = shape[layout[i]];
    }
    return reordered_shape;
}

VectorDims get_planar_vdims(const snippets::lowered::PortDescriptorPtr& port_desc) {
    return get_planar_vdims(port_desc->get_shape(), port_desc->get_layout());
}

VectorDims get_planar_vdims(const snippets::lowered::ExpressionPort& expr_port) {
    return get_planar_vdims(expr_port.get_descriptor_ptr());
}

} // namespace utils
} // namespace snippets
} // namespace ov
