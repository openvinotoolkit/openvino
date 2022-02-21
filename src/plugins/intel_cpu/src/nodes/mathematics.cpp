// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>

#include <ngraph/ops.hpp>
#include "ie_parallel.hpp"
#include "mathematics.h"
#include "utils/general_utils.h"

using namespace ov::intel_cpu;
using namespace InferenceEngine;

bool MKLDNNMathNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (initializers.find(op->get_type_info()) == initializers.end()) {
            errorMessage = "Unsupported Math layer type.";
            return false;
        }

        if (ov::intel_cpu::one_of(op->get_type_info(), ngraph::op::v0::HardSigmoid::get_type_info_static(), ngraph::op::v0::Selu::get_type_info_static())) {
            auto firstConst = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1));
            auto secondConst = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
            if (!firstConst || !secondConst) {
                errorMessage = "Constant expected as the second and third inputs.";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNMathNode::MKLDNNMathNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache), alpha(0.f), beta(0.f), gamma(0.f) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    initializers[op->get_type_info()](op, *this);
}

void MKLDNNMathNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (int i = 0; i < inputShapes.size(); ++i)
        inDataConf.emplace_back(LayoutType::ncsp, Precision::FP32);

    addSupportedPrimDesc(inDataConf,
                         {{LayoutType::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

std::vector<VectorDims> MKLDNNMathNode::shapeInfer() const {
    return std::vector<VectorDims>{getParentEdgesAtPort(0)[0]->getMemory().getStaticDims()};
}

void MKLDNNMathNode::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

void MKLDNNMathNode::execute(mkldnn::stream strm) {
    size_t dataSize = getChildEdgesAtPort(0)[0]->getMemory().GetShape().getElementsCount();
    const float *src_data = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    float* dst_data = reinterpret_cast<float *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    switch (getAlgorithm()) {
        case ov::intel_cpu::MathAbs:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = (std::abs)(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathAcos:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = acosf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathAcosh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = acoshf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathAsin:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = asinf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathAsinh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = asinhf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathAtan:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = atanf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathAtanh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = atanhf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathCeiling:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = ceilf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathCos:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = cosf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathCosh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = coshf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathFloor:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = floorf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathHardSigmoid:
            alpha = (alpha == 0.0f) ? 0.2f : alpha;
            beta = (beta == 0.0f) ? 0.5f : beta;
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = (std::max)(0.f, (std::min)(1.f, alpha * src_data[i] + beta));
            });
            break;
        case ov::intel_cpu::MathLog:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = logf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathNegative:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = -src_data[i];
            });
            break;
        case ov::intel_cpu::MathReciprocal:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = 1.0f / src_data[i];
            });
            break;
        case ov::intel_cpu::MathSelu:
            alpha = (alpha == 0.0f) ? 1.67326f : alpha;
            gamma = (gamma == 0.0f) ? 1.0507f : gamma;
            parallel_for(dataSize, [&](size_t i) {
                float x = src_data[i];
                dst_data[i] = (x > 0.0f) ? (gamma * x) : (gamma * alpha * (exp(x) - 1.0f));
            });
            break;
        case ov::intel_cpu::MathSign:
            parallel_for(dataSize, [&](size_t i) {
                if (src_data[i] > 0.0f)
                    dst_data[i] = 1.0f;
                else if (src_data[i] < 0.0f)
                    dst_data[i] = -1.0f;
                else
                    dst_data[i] = 0.0f;
            });
            break;
        case ov::intel_cpu::MathSin:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = sinf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathSinh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = sinhf(src_data[i]);
            });
            break;
        case ov::intel_cpu::MathSoftPlus:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = logf(expf(src_data[i]) + 1);
            });
            break;
        case ov::intel_cpu::MathSoftsign:
            parallel_for(dataSize, [&](size_t i) {
                float x = src_data[i];
                dst_data[i] = x / (1.f + (std::abs)(x));
            });
            break;
        case ov::intel_cpu::MathTan:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = tanf(src_data[i]);
            });
            break;
        default:
            IE_THROW() << "Incorrect Reduce layer type";
    }
}

bool MKLDNNMathNode::created() const {
    return getType() == Math;
}

std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, MKLDNNMathNode& node)>> MKLDNNMathNode::initializers {
        {ngraph::op::v0::Abs::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathAbs;
        }},
        {ngraph::op::v0::Acos::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathAcos;
        }},
        {ngraph::op::v3::Acosh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathAcosh;
        }},
        {ngraph::op::v0::Asin::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathAsin;
        }},
        {ngraph::op::v3::Asinh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathAsinh;
        }},
        {ngraph::op::v0::Atan::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathAtan;
        }},
        {ngraph::op::v0::Ceiling::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathCeiling;
        }},
        {ngraph::op::v0::Cos::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathCos;
        }},
        {ngraph::op::v0::Cosh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathCosh;
        }},
        {ngraph::op::v0::Floor::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathFloor;
        }},
        {ngraph::op::v0::HardSigmoid::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathHardSigmoid;
            node.alpha = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
            node.beta = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
        }},
        {ngraph::op::v0::Log::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathLog;
        }},
        {ngraph::op::v0::Negative::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathNegative;
        }},
        {ngraph::op::v0::Selu::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathSelu;
            node.alpha = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
            node.gamma = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
        }},
        {ngraph::op::v0::Sign::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathSign;
        }},
        {ngraph::op::v0::Sin::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathSin;
        }},
        {ngraph::op::v0::Sinh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathSinh;
        }},
        {ngraph::op::v4::SoftPlus::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathSoftPlus;
        }},
        {ngraph::op::v0::Tan::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathTan;
        }},
        {ngraph::op::v3::Atanh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = ov::intel_cpu::MathAtanh;
        }}
};

REG_MKLDNN_PRIM_FOR(MKLDNNMathNode, Math);
