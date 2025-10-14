// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gathermatmul.h"

#include <memory>
#include <string>

#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/util/op_types.hpp"
#include "shape_inference/custom/gathermatmul.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul_compressed.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu::node {

bool GatherMatmul::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        // Check if the operation is BatchGatherMatmul or BatchGatherMatmulCompressed
        const bool isBatchGatherMatmul = ov::is_type<ov::intel_cpu::BatchGatherMatmul>(op);
        const bool isBatchGatherMatmulCompressed = ov::is_type<ov::intel_cpu::BatchGatherMatmulCompressed>(op);

        if (!isBatchGatherMatmul && !isBatchGatherMatmulCompressed) {
            errorMessage = "Only BatchGatherMatmul and BatchGatherMatmulCompressed operations are supported. Got: " +
                           std::string(op->get_type_info().name);
            return false;
        }

        // Check that weights input (port 1) is constant
        if (!ov::op::util::is_on_constant_path(op->input_value(WEIGHTS))) {
            errorMessage = "Only constant weights are supported for GatherMatmul operation";
            return false;
        }

        // For compressed variant, check that scales and zero points are constant
        if (isBatchGatherMatmulCompressed) {
            if (op->get_input_size() > WEIGHT_SCALES) {
                if (!ov::op::util::is_on_constant_path(op->input_value(WEIGHT_SCALES))) {
                    errorMessage = "Only constant weight scales are supported for GatherMatmul operation";
                    return false;
                }
            }

            if (op->get_input_size() > WEIGHT_ZERO_POINTS) {
                if (!ov::op::util::is_on_constant_path(op->input_value(WEIGHT_ZERO_POINTS))) {
                    errorMessage = "Only constant weight zero points are supported for GatherMatmul operation";
                    return false;
                }
            }
        }

        // Check that bias (if present) is constant
        if (op->get_input_size() > BIAS) {
            const auto& biasInput = op->input_value(BIAS);
            // Skip validation if bias is dynamic (empty constant)
            if (biasInput.get_element_type() != ov::element::dynamic) {
                if (!ov::op::util::is_on_constant_path(biasInput)) {
                    errorMessage = "Only constant bias is supported for GatherMatmul operation";
                    return false;
                }
            }
        }

    } catch (...) {
        return false;
    }

    return true;
}

GatherMatmul::GatherMatmul(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, GatherMatmulShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    // Determine the algorithm type
    if (ov::is_type<ov::intel_cpu::BatchGatherMatmulCompressed>(op)) {
        algorithm = Algorithm::GatherMatmulCompressed;
    } else {
        algorithm = Algorithm::GatherMatmulDefault;
    }
}

void GatherMatmul::initSupportedPrimitiveDescriptors() {
    // TODO: implement
}

void GatherMatmul::createPrimitive() {
    // TODO: implement
}

void GatherMatmul::prepareParams() {
    // TODO: implement
}

void GatherMatmul::execute(const dnnl::stream& strm) {
    // TODO: implement
}

void GatherMatmul::executeDynamicImpl(const dnnl::stream& strm) {
    // TODO: implement
}

bool GatherMatmul::created() const {
    return getType() == Type::GatherMatmul;
}

}  // namespace ov::intel_cpu::node
