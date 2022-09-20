// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils.hpp"

#include "snippets/pass/fq_decomposition.hpp"


auto ngraph::snippets::utils::get_non_scalar_constant_count_for_fq(const std::shared_ptr<ngraph::opset1::FakeQuantize>& fq) -> size_t {
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
