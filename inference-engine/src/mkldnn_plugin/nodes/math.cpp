// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "base.hpp"

#include <cmath>
#include <string>
#include <vector>
#include <cassert>

#include "ie_parallel.hpp"
#include "common/tensor_desc_creator.h"
#include "utils/general_utils.h"
#include <ngraph/ops.hpp>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

using MKLDNNPlugin::TensorDescCreatorTypes;

class MathImpl: public ExtLayerBase {
public:
    bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
        try {
            if (initializers.find(op->get_type_info()) == initializers.end()) {
                errorMessage = "Unsupported Math layer type.";
                return false;
            }

            if (MKLDNNPlugin::one_of(op->get_type_info(),
                    ngraph::op::v0::HardSigmoid::type_info,
                    ngraph::op::v0::Selu::type_info)) {
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

    explicit MathImpl(const std::shared_ptr<ngraph::Node>& op) :
            alpha(0.f), beta(0.f), gamma(0.f) {
        try {
            std::string errorMessage;
            if (!isSupportedOperation(op, errorMessage)) {
                IE_THROW(NotImplemented) << errorMessage;
            }

            initializers[op->get_type_info()](op, *this);

            if (MKLDNNPlugin::one_of(op->get_type_info(),
                    ngraph::op::v0::HardSigmoid::type_info,
                    ngraph::op::v0::Selu::type_info)) {
                addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::FP32},
                               {TensorDescCreatorTypes::ncsp, Precision::FP32}},
                              {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
            } else {
                addConfig(op, {{TensorDescCreatorTypes::ncsp, Precision::FP32}},
                              {{TensorDescCreatorTypes::ncsp, Precision::FP32}});
            }
        } catch (InferenceEngine::Exception &ex) {
            errorMsg = ex.what();
            throw;
        }
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs, ResponseDesc *resp) noexcept override {
        size_t dataSize = outputs[0]->size();
        const float *src_data = inputs[0]->cbuffer().as<const float *>() +
            inputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();
        float* dst_data = outputs[0]->cbuffer().as<float *>() +
            outputs[0]->getTensorDesc().getBlockingDesc().getOffsetPadding();

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
            if (resp) {
                std::string errorMsg = "Incorrect Reduce layer type";
                errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            }
            return GENERAL_ERROR;
        }
        return OK;
    }

private:
    static std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, MathImpl& node)>> initializers;

    float alpha = 0.0f;
    float beta = 0.0f;
    float gamma = 0.0f;
};

std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, MathImpl& node)>> MathImpl::initializers = {
    {ngraph::op::v0::Abs::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathAbs;
    }},
    {ngraph::op::v0::Acos::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathAcos;
    }},
    {ngraph::op::v3::Acosh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathAcosh;
    }},
    {ngraph::op::v0::Asin::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathAsin;
    }},
    {ngraph::op::v3::Asinh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathAsinh;
    }},
    {ngraph::op::v0::Atan::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathAtan;
    }},
    {ngraph::op::v0::Ceiling::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathCeiling;
    }},
    {ngraph::op::v0::Cos::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathCos;
    }},
    {ngraph::op::v0::Cosh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathCosh;
    }},
    {ngraph::op::v0::Floor::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathFloor;
    }},
    {ngraph::op::v0::HardSigmoid::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathHardSigmoid;
        node.alpha = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
        node.beta = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
    }},
    {ngraph::op::v0::Log::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathLog;
    }},
    {ngraph::op::v0::Negative::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathNegative;
    }},
    {ngraph::op::v0::Selu::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathSelu;
        node.alpha = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
        node.gamma = ngraph::as_type_ptr<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
    }},
    {ngraph::op::v0::Sign::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathSign;
    }},
    {ngraph::op::v0::Sin::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathSin;
    }},
    {ngraph::op::v0::Sinh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathSinh;
    }},
    {ngraph::op::v4::SoftPlus::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathSoftPlus;
    }},
    {ngraph::op::v0::Tan::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathTan;
    }},
    {ngraph::op::v3::Atanh::type_info, [](const std::shared_ptr<ngraph::Node>& op, MathImpl& node) {
        node.algorithm = MKLDNNPlugin::MathAtanh;
    }}
};

REG_FACTORY_FOR(MathImpl, Abs);
REG_FACTORY_FOR(MathImpl, Acos);
REG_FACTORY_FOR(MathImpl, Acosh);
REG_FACTORY_FOR(MathImpl, Asin);
REG_FACTORY_FOR(MathImpl, Asinh);
REG_FACTORY_FOR(MathImpl, Atan);
REG_FACTORY_FOR(MathImpl, Atanh);
REG_FACTORY_FOR(MathImpl, Ceil);
REG_FACTORY_FOR(MathImpl, Ceiling);
REG_FACTORY_FOR(MathImpl, Cos);
REG_FACTORY_FOR(MathImpl, Cosh);
REG_FACTORY_FOR(MathImpl, Floor);
REG_FACTORY_FOR(MathImpl, HardSigmoid);
REG_FACTORY_FOR(MathImpl, Log);
REG_FACTORY_FOR(MathImpl, Neg);
REG_FACTORY_FOR(MathImpl, Reciprocal);
REG_FACTORY_FOR(MathImpl, Selu);
REG_FACTORY_FOR(MathImpl, Sign);
REG_FACTORY_FOR(MathImpl, Sin);
REG_FACTORY_FOR(MathImpl, Sinh);
REG_FACTORY_FOR(MathImpl, SoftPlus);
REG_FACTORY_FOR(MathImpl, Softsign);
REG_FACTORY_FOR(MathImpl, Tan);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
