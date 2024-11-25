// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stft.h"

#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/stft.hpp"
#include "openvino/reference/stft.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool STFT::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v15::STFT::get_type_info_static()) {
            errorMessage = "Only STFT operation from the opset15 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

STFT::STFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op, PortMask(2, 3))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    const auto stft_op = as_type_ptr<op::v15::STFT>(op);
    m_transpose_frames = stft_op->get_transpose_frames();

    m_is_frame_size_const = is_type<op::v0::Constant>(stft_op->get_input_node_ptr(FRAME_SIZE_IDX));
    m_is_frame_step_const = is_type<op::v0::Constant>(stft_op->get_input_node_ptr(FRAME_STEP_IDX));
}

void STFT::getSupportedDescriptors() {
    if (getParentEdges().size() != 4) {
        THROW_CPU_NODE_ERR("STFT has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("STFT has incorrect number of output edges.");
    }
}

void STFT::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto dataPrecision = getOriginalInputPrecisionAtPort(DATA_IDX);
    if (!one_of(dataPrecision, ov::element::f32)) {
        dataPrecision = ov::element::f32;
    }

    std::vector<PortConfigurator> configurators({{LayoutType::ncsp, dataPrecision},
                                                 {LayoutType::ncsp, dataPrecision},
                                                 {LayoutType::ncsp, ov::element::i32},
                                                 {LayoutType::ncsp, ov::element::i32}});

    addSupportedPrimDesc(configurators, {{LayoutType::ncsp, dataPrecision}}, impl_desc_type::ref_any);
}

bool STFT::needPrepareParams() const {
    return false;
}

bool STFT::created() const {
    return getType() == Type::STFT;
}

void STFT::execute(dnnl::stream strm) {
    ov::reference::stft(getSrcDataAtPortAs<const float>(DATA_IDX),
                        getSrcDataAtPortAs<const float>(WINDOW_IDX),
                        getDstDataAtPortAs<float>(0),
                        ov::Shape{getSrcMemoryAtPort(DATA_IDX)->getStaticDims()},
                        ov::Shape{getSrcMemoryAtPort(WINDOW_IDX)->getStaticDims()},
                        (getSrcDataAtPortAs<const int32_t>(FRAME_SIZE_IDX))[0],
                        (getSrcDataAtPortAs<const int32_t>(FRAME_STEP_IDX))[0],
                        m_transpose_frames);
}

void STFT::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool STFT::needShapeInfer() const {
    return !(m_is_frame_size_const && m_is_frame_step_const) || Node::needShapeInfer();
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
