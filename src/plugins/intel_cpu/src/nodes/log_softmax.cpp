// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "log_softmax.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
#include "openvino/op/log_softmax.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool LogSoftmax::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto logSoftMax = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(op);
        if (!logSoftMax) {
            errorMessage = "Only v5 LogSoftmax operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

LogSoftmax::LogSoftmax(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto logSoftMax = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(op);
    CPU_NODE_ASSERT(logSoftMax, "is not an instance of v5 LogSoftmax.");

    CPU_NODE_ASSERT(all_of(1U, inputShapes.size(), outputShapes.size()), "has incorrect number of input/output edges!");

    auto dimsSize = getInputShapeAtPort(0).getDims().size();
    if (dimsSize == 0) {
        dimsSize += 1;
    }
    axis = static_cast<int>(logSoftMax->get_axis());
    if (axis < 0) {
        axis += dimsSize;
    }

    CPU_NODE_ASSERT(dimsSize >= static_cast<size_t>(1) + axis,
                    "has incorrect input parameters dimensions and axis number!");
}

void LogSoftmax::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, ov::element::f32}},
                         {{LayoutType::ncsp, ov::element::f32}},
                         impl_desc_type::ref_any);
}

void LogSoftmax::prepareParams() {
    const auto& dims = getParentEdgeAt(0)->getMemory().getStaticDims();
    reducedAxisStride = 1;
    axisStep = 1;
    isLastDim = false;

    int j = static_cast<int>(dims.size()) - 1;
    for (; j >= 0; j--) {
        if (dims[j] != 1) {
            break;
        }
    }
    if (j == axis) {
        isLastDim = true;
    }

    for (int i = 0; i < axis; i++) {
        axisStep *= dims[i];
    }
    reducedAxisSize = dims[axis];
    for (size_t i = (axis + 1); i < dims.size(); i++) {
        reducedAxisStride *= dims[i];
    }
}

void LogSoftmax::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void LogSoftmax::execute([[maybe_unused]] const dnnl::stream& strm) {
    const auto* srcData = getSrcDataAtPortAs<const float>(0);
    auto* dstData = getDstDataAtPortAs<float>(0);

    if (isLastDim) {
        parallel_for(axisStep, [&](size_t i) {
            const float* srcDataPtr = &srcData[i * reducedAxisSize];
            float* dstDataPtr = &dstData[i * reducedAxisSize];

            float reduceProd = 0.0F;
            const float max = *std::max_element(srcDataPtr, srcDataPtr + reducedAxisSize);
            for (size_t j = 0; j < reducedAxisSize; ++j) {
                reduceProd += expf(srcDataPtr[j] - max);
            }

            reduceProd = logf(reduceProd);
            for (size_t j = 0; j < reducedAxisSize; ++j) {
                dstDataPtr[j] = srcDataPtr[j] - max - reduceProd;
            }
        });
    } else {
        parallel_for2d(axisStep, reducedAxisStride, [&](size_t k, size_t i) {
            const float* srcDataPtr = &srcData[k * reducedAxisStride * reducedAxisSize + i];
            float* dstDataPtr = &dstData[k * reducedAxisStride * reducedAxisSize + i];

            float reduceProd = 0.0F;
            float max = std::numeric_limits<float>::min();
            for (size_t j = 0; j < reducedAxisSize; ++j) {
                if (srcDataPtr[j * reducedAxisStride] > max) {
                    max = srcDataPtr[j * reducedAxisStride];
                }
            }

            for (size_t j = 0; j < reducedAxisSize; ++j) {
                reduceProd += expf(srcDataPtr[j * reducedAxisStride] - max);
            }

            reduceProd = logf(reduceProd);
            for (size_t j = 0; j < reducedAxisSize; ++j) {
                dstDataPtr[j * reducedAxisStride] = srcDataPtr[j * reducedAxisStride] - max - reduceProd;
            }
        });
    }
}

bool LogSoftmax::created() const {
    return getType() == Type::LogSoftmax;
}

}  // namespace ov::intel_cpu::node
