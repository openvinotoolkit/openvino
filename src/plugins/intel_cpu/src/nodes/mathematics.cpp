// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>

#include <ngraph/ops.hpp>
#include "ie_parallel.hpp"
#include "mathematics.h"
#include "utils/general_utils.h"
#include <shape_inference/shape_inference_pass_through.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Math::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (getInitializers().find(op->get_type_info()) == getInitializers().end()) {
            errorMessage = "Unsupported Math layer type.";
            return false;
        }

        if (one_of(op->get_type_info(), ngraph::op::v0::HardSigmoid::get_type_info_static(), ngraph::op::v0::Selu::get_type_info_static())) {
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

Math::Math(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, PassThroughShapeInferFactory()),
      alpha(0.f),
      beta(0.f),
      gamma(0.f) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    getInitializers()[op->get_type_info()](op, *this);
}

void Math::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); ++i)
        inDataConf.emplace_back(LayoutType::ncsp, Precision::FP32);

    addSupportedPrimDesc(inDataConf,
                         {{LayoutType::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void Math::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void Math::execute(dnnl::stream strm) {
    size_t dataSize = getChildEdgesAtPort(0)[0]->getMemory().getShape().getElementsCount();
    const float *src_data = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->getData());
    float* dst_data = reinterpret_cast<float *>(getChildEdgeAt(0)->getMemoryPtr()->getData());

    switch (getAlgorithm()) {
        case Algorithm::MathAbs:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = (std::abs)(src_data[i]);
            });
            break;
        case Algorithm::MathAcos:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = acosf(src_data[i]);
            });
            break;
        case Algorithm::MathAcosh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = acoshf(src_data[i]);
            });
            break;
        case Algorithm::MathAsin:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = asinf(src_data[i]);
            });
            break;
        case Algorithm::MathAsinh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = asinhf(src_data[i]);
            });
            break;
        case Algorithm::MathAtan:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = atanf(src_data[i]);
            });
            break;
        case Algorithm::MathAtanh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = atanhf(src_data[i]);
            });
            break;
        case Algorithm::MathCeiling:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = ceilf(src_data[i]);
            });
            break;
        case Algorithm::MathCos:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = cosf(src_data[i]);
            });
            break;
        case Algorithm::MathCosh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = coshf(src_data[i]);
            });
            break;
        case Algorithm::MathFloor:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = floorf(src_data[i]);
            });
            break;
        case Algorithm::MathHardSigmoid:
            alpha = (alpha == 0.0f) ? 0.2f : alpha;
            beta = (beta == 0.0f) ? 0.5f : beta;
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = (std::max)(0.f, (std::min)(1.f, alpha * src_data[i] + beta));
            });
            break;
        case Algorithm::MathNegative:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = -src_data[i];
            });
            break;
        case Algorithm::MathReciprocal:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = 1.0f / src_data[i];
            });
            break;
        case Algorithm::MathSelu:
            alpha = (alpha == 0.0f) ? 1.67326f : alpha;
            gamma = (gamma == 0.0f) ? 1.0507f : gamma;
            parallel_for(dataSize, [&](size_t i) {
                float x = src_data[i];
                dst_data[i] = (x > 0.0f) ? (gamma * x) : (gamma * alpha * (exp(x) - 1.0f));
            });
            break;
        case Algorithm::MathSign:
            parallel_for(dataSize, [&](size_t i) {
                if (src_data[i] > 0.0f)
                    dst_data[i] = 1.0f;
                else if (src_data[i] < 0.0f)
                    dst_data[i] = -1.0f;
                else
                    dst_data[i] = 0.0f;
            });
            break;
        case Algorithm::MathSin:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = sinf(src_data[i]);
            });
            break;
        case Algorithm::MathSinh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = sinhf(src_data[i]);
            });
            break;
        case Algorithm::MathSoftPlus:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = logf(expf(src_data[i]) + 1);
            });
            break;
        case Algorithm::MathSoftsign:
            parallel_for(dataSize, [&](size_t i) {
                float x = src_data[i];
                dst_data[i] = x / (1.f + (std::abs)(x));
            });
            break;
        case Algorithm::MathTan:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = tanf(src_data[i]);
            });
            break;
        default:
            IE_THROW() << "Incorrect Reduce layer type";
    }
}

bool Math::created() const {
    return getType() == Type::Math;
}

std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, Math& node)>>& Math::getInitializers() {
    static std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, Math& node)>> initializers {
            {ngraph::op::v0::Abs::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathAbs;
            }},
            {ngraph::op::v0::Acos::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathAcos;
            }},
            {ngraph::op::v3::Acosh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathAcosh;
            }},
            {ngraph::op::v0::Asin::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathAsin;
            }},
            {ngraph::op::v3::Asinh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathAsinh;
            }},
            {ngraph::op::v0::Atan::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathAtan;
            }},
            {ngraph::op::v0::Ceiling::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathCeiling;
            }},
            {ngraph::op::v0::Cos::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathCos;
            }},
            {ngraph::op::v0::Cosh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathCosh;
            }},
            {ngraph::op::v0::Floor::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathFloor;
            }},
            {ngraph::op::v0::HardSigmoid::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathHardSigmoid;
                node.alpha = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
                node.beta = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
            }},
            {ngraph::op::v0::Negative::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathNegative;
            }},
            {ngraph::op::v0::Selu::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathSelu;
                node.alpha = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
                node.gamma = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
            }},
            {ngraph::op::v0::Sign::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathSign;
            }},
            {ngraph::op::v0::Sin::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathSin;
            }},
            {ngraph::op::v0::Sinh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathSinh;
            }},
            {ngraph::op::v4::SoftPlus::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathSoftPlus;
            }},
            {ngraph::op::v0::Tan::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathTan;
            }},
            {ngraph::op::v3::Atanh::get_type_info_static(), [](const std::shared_ptr<ngraph::Node>& op, Math& node) {
                node.algorithm = Algorithm::MathAtanh;
            }}
    };
    return initializers;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
