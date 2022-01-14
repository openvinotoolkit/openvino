// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace GNAPluginNS {

/**
 * @brief Convert Unaligned Concat int GNA friendly graph utilizing Convolution operation.
 * This transformation detects unaligned Concat operation and handles its inputs to copy
 * Them using GNA Primitives to desired addresses.
 * Unalignment problem is caused by the GNA operation restriction that tensor can only be written
 * To address that is aligned to 64B.
 * When destination of any of Concat input is not aligned to 64B specific approach must be used.
 * Convolution with specially preparred filters can be used to copy input into output with some offset.
 * Practically 4 filters of the form are used:
 *              0 ... 0 1 0 0 0 0 ... 0
 *              0 ... 0 0 1 0 0 0 ... 0
 *              0 ... 0 0 0 1 0 0 ... 0
 *              0 ... 0 0 0 0 1 0 ... 0
 * The number of "0" before "1" int the first filter denotes the offset from the aligned address.
 * Additional handling of head and tail of each Concat's component may be needed.
 * If so it is done in a form of MatMul Layer and realized using GNA fully connected layer.
 * Component that are aligned (i.e., their destination addresses are) can be handled easier -
 * by using GNA copy operation and optional tail handling (with MatMul).
 * Due to overwriting of the previous component by the not aligned component the order
 * of Concat's input handling must be reversed (last component copied first).
 */
class ConvertUnalignedConcatIntoGnaGraph : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    ConvertUnalignedConcatIntoGnaGraph();
    bool unaligned_concat_converted = false;
};

/**
 * @brief Fold Concat from Strided Slices from a common node.
 * This transformation simplifies the network when detects
 * a pattern containing Concat layer with multiple Strided Slices
 * inputs from the common node. It is applied when Concat's output
 * is effectively the same as the original output node.
 * Pattern:
 *              Original node output
 *                        |
 *              Multiple strided slices
 *                        |
 *                     Concat
 *                        |
 *                     Network
 * is simplified into:
 *              Original node output
 *                        |
 *                     Network
 */
class RemoveTrivialStrideConcatPattern : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    RemoveTrivialStrideConcatPattern();
};

/**
 * @brief Handle Unaligned Concat for GNA.
 * Combination of two transformations. The first one simplifies the graph
 * by removing some unneeded Concat nodes. The second converts Unaligned Concat
 * into GNA friendly graph utilizing Convolution operation.
 */
class TransformConcatForGna : public ngraph::pass::GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    TransformConcatForGna() {
        add_matcher<RemoveTrivialStrideConcatPattern>();
        convUCPass = add_matcher<ConvertUnalignedConcatIntoGnaGraph>();
    }

    bool unalignedConcatIntoGnaGraphConverted() const;

private:
    std::shared_ptr<ConvertUnalignedConcatIntoGnaGraph> convUCPass;
};

} // namespace GNAPluginNS
