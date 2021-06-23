// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>

#include <ngraph/ops.hpp>
#include "ie_parallel.hpp"
#include "mkldnn_math_node.h"
#include "utils/general_utils.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNMathNode::isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (initializers.find(op->get_type_info()) == initializers.end()) {
            errorMessage = "Unsupported Math layer type.";
            return false;
        }

        if (MKLDNNPlugin::one_of(op->get_type_info(), ngraph::op::v0::HardSigmoid::type_info, ngraph::op::v0::Selu::type_info)) {
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

    std::vector<DataConfigurator> inDataConf;
    inDataConf.reserve(getOriginalInputsNumber());
    for (int i = 0; i < getOriginalInputsNumber(); ++i)
        inDataConf.emplace_back(TensorDescCreatorTypes::ncsp, Precision::FP32);

    addSupportedPrimDesc(inDataConf,
                         {{TensorDescCreatorTypes::ncsp, Precision::FP32}},
                         impl_desc_type::ref_any);
}

void MKLDNNMathNode::execute(mkldnn::stream strm) {
    size_t dataSize = getChildEdgeAt(0)->getBlob()->size();
    const float *src_data = reinterpret_cast<const float *>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    float* dst_data = reinterpret_cast<float *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    switch (getAlgorithm()) {
        case MKLDNNPlugin::MathAbs:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = (std::abs)(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathAcos:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = acosf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathAcosh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = acoshf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathAsin:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = asinf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathAsinh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = asinhf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathAtan:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = atanf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathAtanh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = atanhf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathCeiling:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = ceilf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathCos:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = cosf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathCosh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = coshf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathFloor:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = floorf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathHardSigmoid:
            alpha = (alpha == 0.0f) ? 0.2f : alpha;
            beta = (beta == 0.0f) ? 0.5f : beta;
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = (std::max)(0.f, (std::min)(1.f, alpha * src_data[i] + beta));
            });
            break;
        case MKLDNNPlugin::MathLog:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = logf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathNegative:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = -src_data[i];
            });
            break;
        case MKLDNNPlugin::MathReciprocal:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = 1.0f / src_data[i];
            });
            break;
        case MKLDNNPlugin::MathSelu:
            alpha = (alpha == 0.0f) ? 1.67326f : alpha;
            gamma = (gamma == 0.0f) ? 1.0507f : gamma;
            parallel_for(dataSize, [&](size_t i) {
                float x = src_data[i];
                dst_data[i] = (x > 0.0f) ? (gamma * x) : (gamma * alpha * (exp(x) - 1.0f));
            });
            break;
        case MKLDNNPlugin::MathSign:
            parallel_for(dataSize, [&](size_t i) {
                if (src_data[i] > 0.0f)
                    dst_data[i] = 1.0f;
                else if (src_data[i] < 0.0f)
                    dst_data[i] = -1.0f;
                else
                    dst_data[i] = 0.0f;
            });
            break;
        case MKLDNNPlugin::MathSin:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = sinf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathSinh:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = sinhf(src_data[i]);
            });
            break;
        case MKLDNNPlugin::MathSoftPlus:
            parallel_for(dataSize, [&](size_t i) {
                dst_data[i] = logf(expf(src_data[i]) + 1);
            });
            break;
        case MKLDNNPlugin::MathSoftsign:
            parallel_for(dataSize, [&](size_t i) {
                float x = src_data[i];
                dst_data[i] = x / (1.f + (std::abs)(x));
            });
            break;
        case MKLDNNPlugin::MathTan:
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
        {ngraph::op::v0::Abs::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathAbs;
        }},
        {ngraph::op::v0::Acos::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathAcos;
        }},
        {ngraph::op::v3::Acosh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathAcosh;
        }},
        {ngraph::op::v0::Asin::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathAsin;
        }},
        {ngraph::op::v3::Asinh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathAsinh;
        }},
        {ngraph::op::v0::Atan::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathAtan;
        }},
        {ngraph::op::v0::Ceiling::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathCeiling;
        }},
        {ngraph::op::v0::Cos::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathCos;
        }},
        {ngraph::op::v0::Cosh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathCosh;
        }},
        {ngraph::op::v0::Floor::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathFloor;
        }},
        {ngraph::op::v0::HardSigmoid::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathHardSigmoid;
            node.alpha = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
            node.beta = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
        }},
        {ngraph::op::v0::Log::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathLog;
        }},
        {ngraph::op::v0::Negative::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathNegative;
        }},
        {ngraph::op::v0::Selu::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathSelu;
            node.alpha = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
            node.gamma = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
        }},
        {ngraph::op::v0::Sign::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathSign;
        }},
        {ngraph::op::v0::Sin::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathSin;
        }},
        {ngraph::op::v0::Sinh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathSinh;
        }},
        {ngraph::op::v4::SoftPlus::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathSoftPlus;
        }},
        {ngraph::op::v0::Tan::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathTan;
        }},
        {ngraph::op::v3::Atanh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MKLDNNMathNode& node) {
            node.algorithm = MKLDNNPlugin::MathAtanh;
        }}
};

REG_MKLDNN_PRIM_FOR(MKLDNNMathNode, Math);
