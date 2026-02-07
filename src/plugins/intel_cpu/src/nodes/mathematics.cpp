// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mathematics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <shape_inference/shape_inference_pass_through.hpp>
#include <string>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/tan.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

bool Math::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (getInitializers().find(op->get_type_info()) == getInitializers().end()) {
            errorMessage = "Unsupported Math layer type.";
            return false;
        }

        if (any_of(op->get_type_info(),
                   ov::op::v0::HardSigmoid::get_type_info_static(),
                   ov::op::v0::Selu::get_type_info_static())) {
            auto firstConst = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
            auto secondConst = ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
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

Math::Math(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    getInitializers()[op->get_type_info()](op, *this);
}

void Math::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    // Respect the original input precision to avoid precision mismatches
    // that can cause numerical inconsistencies in FP16 mode
    ov::element::Type precision = getOriginalInputPrecisionAtPort(0);
    if (none_of(precision, ov::element::f32, ov::element::bf16, ov::element::f16)) {
        precision = ov::element::f32;
    }

    std::vector<PortConfigurator> inDataConf;
    inDataConf.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        inDataConf.emplace_back(LayoutType::ncsp, precision);
    }

    addSupportedPrimDesc(inDataConf, {{LayoutType::ncsp, precision}}, impl_desc_type::ref_any);
}

void Math::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

template <typename T>
void Math::executeImpl() {
    size_t dataSize = getChildEdgeAt(0)->getMemory().getShape().getElementsCount();
    const auto* src_data = getSrcDataAtPortAs<const T>(0);
    auto* dst_data = getDstDataAtPortAs<T>(0);
    const auto& cpu_parallel = context->getCpuParallel();

    switch (getAlgorithm()) {
    case Algorithm::MathAbs:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>((std::abs)(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathAcos:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(acosf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathAcosh:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(acoshf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathAsin:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(asinf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathAsinh:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(asinhf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathAtan:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(atanf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathAtanh:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(atanhf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathCeiling:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(ceilf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathCos:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(cosf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathCosh:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(coshf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathFloor:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(floorf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathHardSigmoid: {
        float a = (alpha == 0.0F) ? 0.2F : alpha;
        float b = (beta == 0.0F) ? 0.5F : beta;
        cpu_parallel->parallel_for(dataSize, [&, a, b](size_t i) {
            float val = a * static_cast<float>(src_data[i]) + b;
            dst_data[i] = static_cast<T>((std::max)(0.F, (std::min)(1.F, val)));
        });
        break;
    }
    case Algorithm::MathNegative:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(-static_cast<float>(src_data[i]));
        });
        break;
    case Algorithm::MathReciprocal:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(1.0F / static_cast<float>(src_data[i]));
        });
        break;
    case Algorithm::MathSelu: {
        float a = (alpha == 0.0F) ? 1.67326F : alpha;
        float g = (gamma == 0.0F) ? 1.0507F : gamma;
        cpu_parallel->parallel_for(dataSize, [&, a, g](size_t i) {
            float x = static_cast<float>(src_data[i]);
            dst_data[i] = static_cast<T>((x > 0.0F) ? (g * x) : (g * a * (std::exp(x) - 1.0F)));
        });
        break;
    }
    case Algorithm::MathSign:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            float val = static_cast<float>(src_data[i]);
            if (val > 0.0F) {
                dst_data[i] = static_cast<T>(1.0F);
            } else if (val < 0.0F) {
                dst_data[i] = static_cast<T>(-1.0F);
            } else if (std::isnan(val)) {
                dst_data[i] = src_data[i];
            } else {
                dst_data[i] = static_cast<T>(0.0F);
            }
        });
        break;
    case Algorithm::MathSin:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(sinf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathSinh:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(sinhf(static_cast<float>(src_data[i])));
        });
        break;
    case Algorithm::MathSoftPlus:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(logf(expf(static_cast<float>(src_data[i])) + 1));
        });
        break;
    case Algorithm::MathSoftsign:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            float x = static_cast<float>(src_data[i]);
            dst_data[i] = static_cast<T>(x / (1.F + (std::abs)(x)));
        });
        break;
    case Algorithm::MathTan:
        cpu_parallel->parallel_for(dataSize, [&](size_t i) {
            dst_data[i] = static_cast<T>(tanf(static_cast<float>(src_data[i])));
        });
        break;
    default:
        OPENVINO_THROW("Incorrect Math layer type");
    }
}

void Math::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto precision = getChildEdgeAt(0)->getMemory().getDesc().getPrecision();

    if (precision == ov::element::f32) {
        executeImpl<float>();
    } else if (precision == ov::element::bf16) {
        executeImpl<ov::bfloat16>();
    } else if (precision == ov::element::f16) {
        executeImpl<ov::float16>();
    } else {
        OPENVINO_THROW("Unsupported precision: ", precision);
    }
}

bool Math::created() const {
    return getType() == Type::Math;
}

std::map<const ov::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ov::Node>&, Math& node)>>&
Math::getInitializers() {
    static std::map<const ov::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ov::Node>&, Math& node)>>
        initializers{
            {ov::op::v0::Abs::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathAbs;
             }},
            {ov::op::v0::Acos::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathAcos;
             }},
            {ov::op::v3::Acosh::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathAcosh;
             }},
            {ov::op::v0::Asin::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathAsin;
             }},
            {ov::op::v3::Asinh::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathAsinh;
             }},
            {ov::op::v0::Atan::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathAtan;
             }},
            {ov::op::v0::Ceiling::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathCeiling;
             }},
            {ov::op::v0::Cos::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathCos;
             }},
            {ov::op::v0::Cosh::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathCosh;
             }},
            {ov::op::v0::Floor::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathFloor;
             }},
            {ov::op::v0::HardSigmoid::get_type_info_static(),
             [](const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathHardSigmoid;
                 node.alpha =
                     ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
                 node.beta =
                     ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
             }},
            {ov::op::v0::Negative::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathNegative;
             }},
            {ov::op::v0::Selu::get_type_info_static(),
             [](const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathSelu;
                 node.alpha =
                     ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1))->cast_vector<float>()[0];
                 node.gamma =
                     ov::as_type_ptr<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2))->cast_vector<float>()[0];
             }},
            {ov::op::v0::Sign::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathSign;
             }},
            {ov::op::v0::Sin::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathSin;
             }},
            {ov::op::v0::Sinh::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathSinh;
             }},
            {ov::op::v4::SoftPlus::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathSoftPlus;
             }},
            {ov::op::v0::Tan::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathTan;
             }},
            {ov::op::v3::Atanh::get_type_info_static(),
             []([[maybe_unused]] const std::shared_ptr<ov::Node>& op, Math& node) {
                 node.algorithm = Algorithm::MathAtanh;
             }}};
    return initializers;
}

}  // namespace ov::intel_cpu::node
