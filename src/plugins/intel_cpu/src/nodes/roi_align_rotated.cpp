// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_align_rotated.h"

#include <openvino/opsets/opset15.hpp>

#include "common/cpu_convert.h"
#include "openvino/reference/roi_align.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

ROIAlignRotated::ROIAlignRotated(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    const auto roiAlign = ov::as_type_ptr<const ov::opset15::ROIAlignRotated>(op);
    pooledH = roiAlign->get_pooled_h();
    pooledW = roiAlign->get_pooled_w();
    spatialScale = roiAlign->get_spatial_scale();
    samplingRatio = roiAlign->get_sampling_ratio();
    clockwiseMode = roiAlign->get_clockwise_mode();
}

void ROIAlignRotated::getSupportedDescriptors() {
    // Validation is already done in the ov::opset14::ROIAlignRotated.
}

void ROIAlignRotated::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    ov::element::Type inputPrec0 = getOriginalInputPrecisionAtPort(0);
    ov::element::Type outputPrec = getOriginalOutputPrecisionAtPort(0);

    addSupportedPrimDesc(
        {{LayoutType::ncsp, inputPrec0}, {LayoutType::ncsp, ov::element::f32}, {LayoutType::ncsp, ov::element::i32}},
        {{LayoutType::ncsp, outputPrec}},
        impl_desc_type::ref);
}

bool ROIAlignRotated::created() const {
    return getType() == Type::ROIAlignRotated;
}

bool ROIAlignRotated::needPrepareParams() const {
    return false;
}

void ROIAlignRotated::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <ov::element::Type_t OV_TYPE>
void ROIAlignRotated::executeImpl() {
    using T = typename ov::element_type_traits<OV_TYPE>::value_type;

    const size_t batch_indices_size = getSrcMemoryAtPort(2)->getShape().getElementsCount();

    std::vector<int64_t> batch_indices_vec_scaled_up(batch_indices_size);
    cpu_convert(getSrcMemoryAtPort(2)->getData(),
                batch_indices_vec_scaled_up.data(),
                getSrcMemoryAtPort(2)->getPrecision(),
                ov::element::i64,
                batch_indices_size);

    ov::reference::roi_align<T, ov::reference::roi_policy::ROIAlignRotatedOpDefPolicy>(
        getSrcDataAtPortAs<const T>(0),
        getSrcDataAtPortAs<const T>(1),
        batch_indices_vec_scaled_up.data(),
        getDstDataAtPortAs<T>(0),
        ov::Shape{getSrcMemoryAtPort(0)->getStaticDims()},
        ov::Shape{getSrcMemoryAtPort(1)->getStaticDims()},
        ov::Shape{getSrcMemoryAtPort(2)->getStaticDims()},
        ov::Shape{getDstMemoryAtPort(0)->getStaticDims()},
        pooledH,
        pooledW,
        samplingRatio,
        spatialScale,
        ov::op::v3::ROIAlign::PoolingMode::AVG,
        ov::op::v9::ROIAlign::AlignedMode::ASYMMETRIC,
        clockwiseMode);
}

void ROIAlignRotated::execute(dnnl::stream) {
    const ov::element::Type type = getOriginalInputPrecisionAtPort(0);
    executeImpl<ov::element::f32>();

#define CASE(OV_TYPE)                        \
    case ov::element::OV_TYPE:               \
        executeImpl<ov::element::OV_TYPE>(); \
        break;

    switch (type) {
        CASE(bf16);
        CASE(f16);
        CASE(f32);
        CASE(f64);
    default:
        OPENVINO_THROW("[ROIAlignRotated]: Unhandled data type ", type, " in execute()");
    }
#undef CASE
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
