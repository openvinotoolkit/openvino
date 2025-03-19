// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range.h"

#include <string>

#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset1.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool Range::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v0::Range::get_type_info_static(),
                    ov::op::v4::Range::get_type_info_static())) {
            errorMessage = "Only opset1 and opset4 Range operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Range::Range(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    if (getOriginalInputsNumber() != 3 || getOriginalOutputsNumber() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges!");
    }

    auto start_dims = op->get_input_shape(RANGE_START);
    if (ov::shape_size(start_dims) != 1) {
        THROW_CPU_NODE_ERR("has start scalar with more than 1 value");
    }

    auto limit_dims = op->get_input_shape(RANGE_LIMIT);
    if (ov::shape_size(limit_dims) != 1) {
        THROW_CPU_NODE_ERR("has limit scalar with more than 1 value");
    }

    auto delta_dims = op->get_input_shape(RANGE_DELTA);
    if (ov::shape_size(delta_dims) != 1) {
        THROW_CPU_NODE_ERR("has delta scalar with more than 1 value");
    }

    size_t dstRank = op->get_output_partial_shape(0).size();
    if (dstRank > 1) {
        THROW_CPU_NODE_ERR("has unsupported rank for output: ", dstRank);
    }
}

void Range::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    std::vector<PortConfigurator> inDataConf;
    std::vector<PortConfigurator> outDataConf;

    if (!(getOriginalInputPrecisionAtPort(RANGE_START) == ov::element::i32 &&
          getOriginalInputPrecisionAtPort(RANGE_LIMIT) == ov::element::i32 &&
          getOriginalInputPrecisionAtPort(RANGE_DELTA) == ov::element::i32 &&
          getOriginalOutputPrecisionAtPort(0) == ov::element::i32) &&
        !(getOriginalInputPrecisionAtPort(RANGE_START) == ov::element::f32 &&
          getOriginalInputPrecisionAtPort(RANGE_LIMIT) == ov::element::f32 &&
          getOriginalInputPrecisionAtPort(RANGE_DELTA) == ov::element::f32 &&
          getOriginalOutputPrecisionAtPort(0) == ov::element::f32)) {
        inDataConf.reserve(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); ++i) {
            inDataConf.emplace_back(LayoutType::ncsp, ov::element::f32);
        }
        outDataConf.reserve(1);
        outDataConf.emplace_back(LayoutType::ncsp, ov::element::f32);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
    } else {
        inDataConf.reserve(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); ++i) {
            inDataConf.emplace_back(LayoutType::ncsp);
        }
        outDataConf.reserve(1);
        outDataConf.emplace_back(LayoutType::ncsp);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref_any);
    }
}

void Range::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void Range::execute(const dnnl::stream& strm) {
    StatusCode retcode = OK;
    switch (getParentEdgeAt(0)->getMemory().getDesc().getPrecision()) {
    case ov::element::f32:
        retcode = rangeKernel<float>();
        break;
    case ov::element::i32:
        retcode = rangeKernel<int32_t>();
        break;
    default:
        THROW_CPU_NODE_ERR("Incorrect output precision. Only FP32 and I32 are supported!");
    }
    if (retcode == PARAMETER_MISMATCH) {
        THROW_CPU_NODE_ERR("Range indexes exceeds data tensor dimension");
    }
}

template <typename data_t>
size_t Range::getWorkAmount(data_t* startPtr, data_t* stopPtr, data_t* stepPtr) const {
    data_t start = 0, limit = 0, delta = 0;
    if (startPtr == nullptr) {
        startPtr = &start;
    }
    if (stopPtr == nullptr) {
        stopPtr = &limit;
    }
    if (stepPtr == nullptr) {
        stepPtr = &delta;
    }
    *startPtr = getSrcDataAtPortAs<const data_t>(RANGE_START)[0];
    *stopPtr = getSrcDataAtPortAs<const data_t>(RANGE_LIMIT)[0];
    *stepPtr = getSrcDataAtPortAs<const data_t>(RANGE_DELTA)[0];
    const data_t span = *stopPtr - *startPtr;
    const data_t step = *stepPtr;
    if (std::is_same<data_t, int>::value) {
        auto iSpan = static_cast<int>(span);
        auto iStep = static_cast<int>(step);
        return static_cast<size_t>(div_up(iSpan < 0 ? -iSpan : iSpan, iStep < 0 ? -iStep : iStep));
    }
    return static_cast<size_t>(std::ceil(std::fabs(span) / std::fabs(step)));
}

template <typename data_t>
Range::StatusCode Range::rangeKernel() {
    data_t start = 0, delta = 0;
    size_t work_amount_dst = getWorkAmount<data_t>(&start, nullptr, &delta);
    if (isDynamicNode()) {
        VectorDims newOutputShape{work_amount_dst};
        redefineOutputMemory({newOutputShape});
    }
    auto* dst_data = getDstDataAtPortAs<data_t>(0);
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t iwork = 0, end = 0;
        splitter(work_amount_dst, nthr, ithr, iwork, end);
        data_t dst_value = start + iwork * delta;
        for (; iwork < end; ++iwork, dst_value += delta) {
            dst_data[iwork] = dst_value;
        }
    });
    return OK;
}

bool Range::created() const {
    return getType() == Type::Range;
}

}  // namespace ov::intel_cpu::node
