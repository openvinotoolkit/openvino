// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include "low_precision/layer_transformation.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief ShuffleChannelsTransformation propagates dequantization operations through ShuffleChannels operation.
 *
 * For more details about the transformation, refer to
 * [ShuffleChannelsTransformation](@ref openvino_docs_OV_UG_lpt_ShuffleChannelsTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API ShuffleChannelsTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("ShuffleChannelsTransformation", "0", LayerTransformation);
    ShuffleChannelsTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher& m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
    bool canBeTransformed(const std::shared_ptr<Node>& op) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
