// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace vpu {

class ConvertExtractImagePatchesToReorgYolo : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertExtractImagePatchesToReorgYolo", "0");
    ConvertExtractImagePatchesToReorgYolo();
};

}  // namespace vpu
