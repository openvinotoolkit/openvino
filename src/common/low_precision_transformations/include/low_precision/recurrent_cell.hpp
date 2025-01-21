// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

class LP_TRANSFORMATIONS_API RecurrentCellTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("RecurrentCellTransformation", "0", LayerTransformation);
    RecurrentCellTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool canBeTransformed(const std::shared_ptr<Node>& layer) const override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    void propagateSkipCleanupAttribute(std::shared_ptr<Node> dequantization_multiply);
    static std::shared_ptr<ov::Node> wrap_fake_quantize(const std::shared_ptr<ov::Node> parameter);
    static std::shared_ptr<ov::Node> wrap_quantization(const std::shared_ptr<ov::Node> parameter);
    static std::shared_ptr<ov::Node> wrap_dequantization(const std::shared_ptr<ov::Node> parameter, const bool with_subtract);

private:
    void propagate(const std::shared_ptr<ov::Node> node);
};

} // namespace low_precision
} // namespace pass
} // namespace ov
