// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

// Eliminates redundant Convert(Tnarrow)->Convert(Tsrc) round-trip pairs,
// replacing the outer Convert's output with the original source.
class TRANSFORMATIONS_API EraseRedundantConvertPair : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EraseRedundantConvertPair");
    EraseRedundantConvertPair();
};

}  // namespace ov::pass
