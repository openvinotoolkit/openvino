// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "layer_transformation.hpp"
#include "openvino/op/fake_quantize.hpp"

namespace ov {
namespace pass {
namespace low_precision {

namespace fq_decomposition {

/**
 * @brief Extracts output_low and output_high constant values from a FakeQuantize node.
 * @return true if both inputs 3 and 4 are constants and values were extracted, false otherwise.
 */
LP_TRANSFORMATIONS_API bool getOutputRanges(const std::shared_ptr<ov::op::v0::FakeQuantize>& layer,
                                            std::vector<float>& outputLowValues,
                                            std::vector<float>& outputHighValues);

/**
 * @brief Constructs a DataPrecision from a resolved precision and FQ parameters.
 * Computes hasZeroPoint by checking whether the FQ output ranges match the target precision.
 */
LP_TRANSFORMATIONS_API DataPrecision makeDataPrecision(const ov::element::Type& precision,
                                                       size_t levels,
                                                       const std::vector<float>& outputLowValues,
                                                       const std::vector<float>& outputHighValues);

}  // namespace fq_decomposition

/**
 * @ingroup ov_transformation_common_api
 * @brief FakeQuantizeDecompositionTransformation decomposes FakeQuantize operations to quantize
 * (FakeQuantize with changes output intervals and low precision output type) and dequantize operations.
 *
 * For more details about the transformation, refer to
 * [FakeQuantizeDecompositionTransformation](@ref openvino_docs_OV_UG_lpt_FakeQuantizeDecompositionTransformation) page
 * in the OpenVINO Developer Guide.
 */
class LP_TRANSFORMATIONS_API FakeQuantizeDecompositionTransformation : public LayerTransformation {
public:
    OPENVINO_RTTI("FakeQuantizeDecompositionTransformation", "0", LayerTransformation);
    FakeQuantizeDecompositionTransformation(const Params& params = Params());
    bool transform(ov::pass::pattern::Matcher &m) override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;
};

} // namespace low_precision
} // namespace pass
} // namespace ov
