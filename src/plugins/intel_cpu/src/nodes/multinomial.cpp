// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"

#include "ie_ngraph_utils.hpp"
#include "openvino/op/multinomial.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

Multinomial::Multinomial(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op, PortMask(NUM_SAMPLES_PORT))) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    auto multinomial_op = as_type_ptr<op::v13::Multinomial>(op);
    m_with_replacement = multinomial_op->get_with_replacement();
    m_global_seed = multinomial_op->get_global_seed();
    m_log_probs = multinomial_op->get_log_probs();
    m_op_seed = multinomial_op->get_op_seed();
    m_output_precision = multinomial_op->get_convert_type();

    for (size_t i = 0lu; i < op->get_input_size(); i++) {
        if (is_type<op::v0::Constant>(op->get_input_node_ptr(i))) {
            m_const_inputs[i] = true;
        }
    }
}

bool Multinomial::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v13::Multinomial::get_type_info_static()) {
            errorMessage = "Only Multinomial operation from the opset13 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void Multinomial::getSupportedDescriptors() {
    if (getParentEdges().size() != 2) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    }
    if (getChildEdges().size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void Multinomial::initSupportedPrimitiveDescriptors() {
    m_probs_precision = getOriginalInputPrecisionAtPort(PROBS_PORT);
    if (!one_of(m_probs_precision, ov::element::f32, ov::element::f16, ov::element::bf16)) {
        m_probs_precision = ov::element::f32;
    }

    m_num_samples_precision = getOriginalInputPrecisionAtPort(NUM_SAMPLES_PORT);
    if (!one_of(m_num_samples_precision, ov::element::i64, ov::element::i32)) {
        m_num_samples_precision = ov::element::i32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, m_probs_precision, m_const_inputs[PROBS_PORT]},
                          {LayoutType::ncsp, m_num_samples_precision, m_const_inputs[NUM_SAMPLES_PORT]}},
                         {{LayoutType::ncsp, m_output_precision}},
                         ref_any);
}

std::string Multinomial::getPrimitiveDescriptorType() const {
    std::string str_type;
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    impl_desc_type type = impl_desc_type::undef;
    if (selectedPrimitiveDesc) {
        type = selectedPrimitiveDesc->getImplementationType();
    }

    if (type == impl_desc_type::unknown)
        str_type += "unknown_";
    if ((type & impl_desc_type::jit) == impl_desc_type::jit)
        str_type += "jit_";
    if ((type & impl_desc_type::ref) == impl_desc_type::ref)
        str_type += "ref_";
    if ((type & impl_desc_type::avx512) == impl_desc_type::avx512)
        str_type += "avx512_";
    if ((type & impl_desc_type::avx2) == impl_desc_type::avx2)
        str_type += "avx2_";
    if ((type & impl_desc_type::sse42) == impl_desc_type::sse42)
        str_type += "sse42_";
    if ((type & impl_desc_type::any) == impl_desc_type::any)
        str_type += "any_";

    if (str_type.empty())
        str_type += "undef_";

    if (selectedPrimitiveDesc) {
        str_type += m_output_precision.get_type_name();
    } else {
        str_type.pop_back();
    }

    return str_type;
}

bool Multinomial::needShapeInfer() const {
    return true;
}

bool Multinomial::needPrepareParams() const {
    return true;
}

void Multinomial::prepareParams() {
    const auto& probs_shape = getParentEdgeAt(PROBS_PORT)->getMemory().getStaticDims();
    const auto& num_samples_shape = getParentEdgeAt(NUM_SAMPLES_PORT)->getMemory().getStaticDims();

    if (probs_shape.size() != 2) {
        THROW_CPU_NODE_ERR("has incompatible 'probs' shape ",
                           PartialShape(probs_shape),
                           ". Only 2D tensors are allowed.");
    }

    if (num_samples_shape.size() != 1) {
        THROW_CPU_NODE_ERR("has incompatible 'num_samples' shape ",
                           PartialShape(num_samples_shape),
                           ". Only scalar and 1D single element tensors are allowed.");
    }

    if (m_num_samples_precision == ov::element::i32) {
        m_samples_count =
            reinterpret_cast<const int32_t*>(getParentEdgeAt(NUM_SAMPLES_PORT)->getMemoryPtr()->getData())[0];
    } else {
        m_samples_count =
            reinterpret_cast<const int64_t*>(getParentEdgeAt(NUM_SAMPLES_PORT)->getMemoryPtr()->getData())[0];
    }

    m_batches_count = probs_shape[0];
    m_probs_count = probs_shape[1];
    m_samples_probs_count = m_samples_count * m_probs_count;
    m_input_elements_count = m_batches_count * m_probs_count;
    m_output_elements_count = m_batches_count * m_samples_count;
    m_batches_samples_probs_count = m_output_elements_count * m_probs_count;
}

void Multinomial::execute(dnnl::stream strm) {
    switch (m_probs_precision) {
    case ov::element::f32:
        return execute_types<float>();
    case ov::element::f16:
        return execute_types<float16>();
    case ov::element::bf16:
        return execute_types<bfloat16>();
    default:
        THROW_CPU_NODE_ERR("Multinomial CPU implementation does not support probs element type: ", m_probs_precision);
    }
}

void Multinomial::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Multinomial::isExecutable() const {
    return !isInputTensorAtPortEmpty(PROBS_PORT) && !isInputTensorAtPortEmpty(NUM_SAMPLES_PORT);
}

bool Multinomial::created() const {
    return getType() == Type::Multinomial;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
