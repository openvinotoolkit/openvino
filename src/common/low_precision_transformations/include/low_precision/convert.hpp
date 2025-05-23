// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API ConvertTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("ConvertTransformation", "0", LayerTransformation);
    ConvertTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
