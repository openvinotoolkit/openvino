// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "identity.hpp"

#include "openvino/core/parallel.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "openvino/op/constant.hpp"
#include "openvino/op/identity.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

bool Identity::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v16::Identity::get_type_info_static()) {
            errorMessage = "Only Identity operation from the opset16 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Identity::Identity(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
        : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    auto identity_op = as_type_ptr<op::v16::Identity>(op);

    if (is_type<op::v0::Constant>(identity_op->get_input_node_ptr(0))) {
        m_const_input = true;
        constant = ConstantType::Const; // Node always produces the same output
    } else {
        m_const_input = false;
        constant = ConstantType::NoConst; // Node produces output based on input
    }
}

void Identity::getSupportedDescriptors() {
    if (getParentEdges().size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void Identity::initSupportedPrimitiveDescriptors() {
    auto shape_prc = getOriginalInputPrecisionAtPort(0);
    auto out_prc = getOriginalOutputPrecisionAtPort(0);

    if (shape_prc != out_prc) {
        THROW_CPU_NODE_ERR("has to have the same dtype for input and output nodes.");
    }

    m_out_prc = out_prc;

    addSupportedPrimDesc({{LayoutType::ncsp, shape_prc, m_const_input}},
                         {{LayoutType::ncsp, out_prc}},
                         ref_any);
}

bool Identity::needPrepareParams() const {
    if (m_out_shape != getDstMemoryAtPort(0)->getShape().getStaticDims()) {
        return true;
    }
    return false;
}

void Identity::prepareParams() {
    m_out_shape = getDstMemoryAtPort(0)->getShape().getStaticDims();
}

bool Identity::needShapeInfer() const {
    return !m_const_input;
}

bool Identity::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

bool Identity::created() const {
    return getType() == Type::Identity;
}

bool Identity::canBeInPlace() const { 
    return getSrcMemoryAtPort(0) == getDstMemoryAtPort(0); 
}

std::string Identity::getPrimitiveDescriptorType() const {
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    impl_desc_type type = impl_desc_type::undef;
    if (selectedPrimitiveDesc) {
        type = selectedPrimitiveDesc->getImplementationType();
    }

    std::string str_type;
    if (type == impl_desc_type::unknown)
        str_type = "unknown";
    else if (type == impl_desc_type::ref_any)
        str_type = "ref_any";
    else
        str_type = "undef";

    if (selectedPrimitiveDesc) {
        if (selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision() != ov::element::u8) {
            str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision().get_type_name());
        } else {
            str_type += "_I8";
        }
    }

    return str_type;
}

void Identity::execute(dnnl::stream strm) {
    const auto out_el_num = std::accumulate(m_out_shape.begin(), m_out_shape.end(), 1lu, std::multiplies<Dim>());

    if (!canBeInPlace()) {
        auto input = getSrcDataAtPort(0);
        auto output = getDstDataAtPort(0);

        cpu_parallel_memcpy(output, input, m_out_prc.size() * out_el_num);
    }
}

void Identity::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
