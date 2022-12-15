// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reference.h"
#include <ie_ngraph_utils.hpp>
#include <dnnl_extension_utils.h>
#include "openvino/runtime/tensor.hpp"
#include "common/blocked_desc_creator.h"
#include <ngraph/opsets/opset1.hpp>

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
namespace node {

Reference::Reference(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache,
                                         const std::string& errorMessage) :
        Node(op, eng, cache, NgraphShapeInferFactory(op, FULL_PORT_MASK)), ngraphOp(op), additionalErrorMessage(errorMessage) {
    if (!op->has_evaluate()) {
        IE_THROW(NotImplemented) << "Cannot fallback on ngraph reference implementation (Ngraph::Node::evaluate() is not implemented)";
    }
    setType(Type::Reference);
    setTypeStr("Reference");

    // RandomUniform should generate new sequence each run even if all inputs are constants. So that method Node::IsConstant()
    // doesn't return 'True' for RandomUniform with all constant inputs and the node generates new values for each inference,
    // we set 'NoConst' value for 'ConstantType' in ctor
    if (ov::is_type<ngraph::op::v8::RandomUniform>(ngraphOp)) {
        constant = ConstantType::NoConst;
    }

    // rt_info "inputsSupportBF16"/"outputsSupportBF16"
    // indicates that the reference implementation support
    // BF16 data type at runtime.
    const auto & rtInfo = ngraphOp->get_rt_info();
    auto it = rtInfo.find("inputsSupportBF16");
    if (it != rtInfo.end()) {
        inputsSupportBF16 = it->second.as<std::set<int>>();
    }
    it = rtInfo.find("outputsSupportBF16");
    if (it != rtInfo.end()) {
        outputsSupportBF16 = it->second.as<std::set<int>>();
    }
    it = rtInfo.find("internalDynamismShapeInfer");
    if (it != rtInfo.end()) {
        internalDynamismShapeInfer = it->second.as<bool>();
    } else {
        internalDynamismShapeInfer = true;
    }
}

void Reference::getSupportedDescriptors() {}

void Reference::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto addConfig = [&](bool useBF16) {
        std::vector<PortConfigurator> inputConfigurators;
        inputConfigurators.reserve(inputShapes.size());
        for (size_t i = 0; i < inputShapes.size(); i++) {
            auto prec = convertPrecision(ngraphOp->get_input_element_type(i));
            if (useBF16 && inputsSupportBF16.count(i)) {
                prec = Precision::BF16;
            }
            inputConfigurators.emplace_back(LayoutType::ncsp, prec, inputShapes[i]);
        }

        std::vector<PortConfigurator> outputConfigurators;
        outputConfigurators.reserve(inputShapes.size());
        for (size_t i = 0; i < outputShapes.size(); i++) {
            auto prec = convertPrecision(ngraphOp->get_output_element_type(i));
            if (useBF16 && outputsSupportBF16.count(i)) {
                prec = Precision::BF16;
            }
            outputConfigurators.emplace_back(LayoutType::ncsp, prec, outputShapes[i]);
        }

        addSupportedPrimDesc(inputConfigurators, outputConfigurators, impl_desc_type::ref);
    };

    addConfig(false);

    if (!inputsSupportBF16.empty() || !outputsSupportBF16.empty()) {
        addConfig(true);
    }
}

void Reference::createPrimitive() {}

void Reference::execute(dnnl::stream strm) {
    auto getTensor = [this](TensorCache& tensorCache, int port, const Memory& mem) -> const ov::Tensor& {
        auto prec = mem.getDesc().getPrecision();
        auto dims = mem.getStaticDims();
        void* ptr = mem.GetPtr();
        auto it = tensorCache.find(port);
        if (it != tensorCache.end()) {
            TensorEntry& tentry = it->second;
            // cache hit
            if (std::get<0>(tentry) == prec && std::get<1>(tentry) == dims && std::get<2>(tentry) == ptr)
                return std::get<3>(tentry);
        }
        tensorCache[port] = TensorEntry{prec, dims, ptr, ov::Tensor(convertPrecision(prec), dims, ptr)};
        return std::get<3>(tensorCache[port]);
    };

    ov::TensorVector inputs;
    for (size_t i = 0; i < inputShapes.size(); i++) {
        const Memory& mem = getParentEdgesAtPort(i)[0]->getMemory();
        inputs.push_back(getTensor(inputTensorCache, i, mem));
    }

    ov::TensorVector outputs;
    for (size_t i = 0; i < outputShapes.size(); i++) {
        const Memory& mem = getChildEdgesAtPort(i)[0]->getMemory();
        outputs.push_back(getTensor(outputTensorCache, i, mem));
    }

    if (!ngraphOp->evaluate(outputs, inputs)) {
        IE_THROW() << "Evaluation failed on node of type: " << std::string(ngraphOp->get_type_name()) << " name: " << getName();
    }
}

void Reference::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool Reference::created() const {
    return getType() == Type::Reference;
}

bool Reference::needShapeInfer() const {
    if (internalDynamismShapeInfer)
        return true;
    return Node::needShapeInfer();
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
