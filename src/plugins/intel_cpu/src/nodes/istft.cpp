// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "istft.h"

#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/istft.hpp"
#include "openvino/reference/istft.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool ISTFT::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v16::ISTFT::get_type_info_static()) {
            errorMessage = "Only ISTFT operation from the opset16 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ISTFT::ISTFT(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto istft_op = as_type_ptr<op::v16::ISTFT>(op);

    m_is_frame_size_const = is_type<op::v0::Constant>(istft_op->get_input_node_ptr(FRAME_SIZE_IDX));
    m_is_frame_step_const = is_type<op::v0::Constant>(istft_op->get_input_node_ptr(FRAME_STEP_IDX));
    if (istft_op->get_input_size() > SIGNAL_LENGTH_IDX) {
        m_has_signal_length_input = true;
        m_is_signal_length_const = is_type<op::v0::Constant>(istft_op->get_input_node_ptr(SIGNAL_LENGTH_IDX));
    }
    m_center = istft_op->get_center();
    m_normalized = istft_op->get_normalized();
}

void ISTFT::getSupportedDescriptors() {
    const auto input_size = getParentEdges().size();
    if (input_size < 4 || input_size > 5) {
        THROW_CPU_NODE_ERR("ISTFT has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("ISTFT has incorrect number of output edges.");
    }
}

void ISTFT::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto dataPrecision = getOriginalInputPrecisionAtPort(DATA_IDX);
    if (!one_of(dataPrecision, ov::element::f32)) {
        dataPrecision = ov::element::f32;
    }

    std::vector<PortConfigurator> configurators({{LayoutType::ncsp, dataPrecision},
                                                 {LayoutType::ncsp, dataPrecision},
                                                 {LayoutType::ncsp, ov::element::i32},
                                                 {LayoutType::ncsp, ov::element::i32}});
    if (m_has_signal_length_input) {
        configurators.emplace_back(LayoutType::ncsp, ov::element::i32);
    }

    addSupportedPrimDesc(configurators, {{LayoutType::ncsp, dataPrecision}}, impl_desc_type::ref_any);
}

bool ISTFT::needPrepareParams() const {
    return false;
}

bool ISTFT::created() const {
    return getType() == Type::ISTFT;
}

void ISTFT::execute(const dnnl::stream& strm) {
    const auto signal_length =
        m_has_signal_length_input ? (getSrcDataAtPortAs<const int32_t>(SIGNAL_LENGTH_IDX))[0] : -1;
    ov::reference::istft(getSrcDataAtPortAs<const float>(DATA_IDX),
                         getSrcDataAtPortAs<const float>(WINDOW_IDX),
                         getDstDataAtPortAs<float>(0),
                         ov::Shape{getSrcMemoryAtPort(DATA_IDX)->getStaticDims()},
                         ov::Shape{getSrcMemoryAtPort(WINDOW_IDX)->getStaticDims()},
                         (getSrcDataAtPortAs<const int32_t>(FRAME_SIZE_IDX))[0],
                         (getSrcDataAtPortAs<const int32_t>(FRAME_STEP_IDX))[0],
                         signal_length,
                         m_center,
                         m_normalized);
}

void ISTFT::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool ISTFT::needShapeInfer() const {
    return (m_has_signal_length_input && !m_is_signal_length_const) ||
           (!m_has_signal_length_input && !(m_is_frame_size_const && m_is_frame_step_const)) || Node::needShapeInfer();
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
