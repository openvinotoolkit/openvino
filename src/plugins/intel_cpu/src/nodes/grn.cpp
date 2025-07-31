// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grn.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/grn.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool GRN::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto grn = ov::as_type_ptr<const ov::op::v0::GRN>(op);
        if (!grn) {
            errorMessage = "Only v0 GRN operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

GRN::GRN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto grn = ov::as_type_ptr<const ov::op::v0::GRN>(op);
    CPU_NODE_ASSERT(grn, "is not an instance of GRN from v0.");

    CPU_NODE_ASSERT(all_of(1U, inputShapes.size(), outputShapes.size()), "has incorrect number of input/output edges!");

    const auto dataRank = getInputShapeAtPort(0).getRank();

    CPU_NODE_ASSERT(dataRank == getOutputShapeAtPort(0).getRank(), "has input/output rank mismatch");

    bias = grn->get_bias();
}

void GRN::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32, false, 0}},
                         {{LayoutType::ncsp, ov::element::f32, false, 0}},
                         impl_desc_type::ref_any);
}

void GRN::prepareParams() {
    const auto& dataMemPtr = getSrcMemoryAtPort(0);
    const auto& dstMemPtr = getDstMemoryAtPort(0);

    CPU_NODE_ASSERT(dataMemPtr && dataMemPtr->isDefined(), "has undefined input memory");
    CPU_NODE_ASSERT(dstMemPtr && dstMemPtr->isDefined(), "has undefined output memory");
    CPU_NODE_ASSERT(getSelectedPrimitiveDescriptor(), "has unidentified preferable primitive descriptor");

    const VectorDims& dataDims = dataMemPtr->getStaticDims();
    const VectorDims& dstDims = dstMemPtr->getStaticDims();

    for (size_t i = 0; i < dataDims.size(); ++i) {
        CPU_NODE_ASSERT(dataDims[i] == dstDims[i], "hsd input/output tensors dimensions mismatch");
    }

    if (!dataDims.empty()) {
        N = static_cast<int>(dataDims[0]);
    }
    if (dataDims.size() > 1) {
        C = static_cast<int>(dataDims[1]);
    }
    if (dataDims.size() > 2) {
        H = static_cast<int>(dataDims[2]);
    }
    if (dataDims.size() > 3) {
        W = static_cast<int>(dataDims[3]);
    }
}

void GRN::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void GRN::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto* src_data = getSrcDataAtPortAs<const float>(0);
    auto* dst_data = getDstDataAtPortAs<float>(0);

    parallel_for3d(N, H, W, [&](int b, int h, int w) {
        double variance = 0;
        for (int c = 0; c < C; c++) {
            variance += std::pow(src_data[b * C * H * W + c * H * W + h * W + w], 2);
        }
        variance = std::pow(variance + bias, 0.5F);
        for (int c = 0; c < C; c++) {
            dst_data[b * C * H * W + c * H * W + h * W + w] =
                src_data[b * C * H * W + c * H * W + h * W + w] / static_cast<float>(variance);
        }
    });
}

bool GRN::created() const {
    return getType() == Type::GRN;
}

}  // namespace ov::intel_cpu::node
