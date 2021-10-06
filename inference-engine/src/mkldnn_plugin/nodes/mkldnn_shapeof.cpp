// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_shapeof.h"
#include "ie_parallel.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>
#include <math.h>
#include <mkldnn.hpp>
#include <mkldnn_extension_utils.h>
#include <mkldnn_selective_build.h>
#include <mkldnn_types.h>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <string>
#include <utils/bfloat16.hpp>
#include <utils/general_utils.h>
#include <vector>

using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace mkldnn;
using namespace mkldnn::impl::cpu::x64;

bool MKLDNNShapeOfNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ngraph::op::v0::ShapeOf::type_info,
                    ngraph::op::v3::ShapeOf::type_info)) {
            errorMessage = "Node is not an instance of ShapeOf form the operation set v1 or v3.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNShapeOfNode::MKLDNNShapeOfNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                     MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "ShapeOf layer with name '" + getName() + "' ";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (op->get_output_element_type(0) != ngraph::element::i32) {
        IE_THROW() << errorPrefix << "doesn't support demanded output precision";
    }
}

void MKLDNNShapeOfNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    if (getParentEdges().size() != 1)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().size() != 1)
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getParentEdges().size();
}

void MKLDNNShapeOfNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    precision = getOriginalInputPrecisionAtPort(0);

    std::vector<LayoutType> dataFormats{ LayoutType::ncsp, LayoutType::nspc, LayoutType::nCsp16c, LayoutType::nCsp8c };
    for (const auto &df : dataFormats) {
        addSupportedPrimDesc({{df, precision}},
                             {{LayoutType::ncsp, Precision::I32}},
                             impl_desc_type::ref);
    }
}

void MKLDNNShapeOfNode::execute(mkldnn::stream strm) {
    auto inPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto outPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto inDims = inPtr->GetShape().getDims();
    size_t dimsCount = inDims.size();
    if (dimsCount != outPtr->GetShape().getElementsCount())
        IE_THROW() << errorPrefix << "has inconsistent input shape and output size";

    auto *dst = reinterpret_cast<int *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    for (size_t i = 0; i < dimsCount; i++) {
        dst[i] = inDims[i];
    }
}

bool MKLDNNShapeOfNode::created() const {
    return getType() == ShapeOf;
}

void MKLDNNShapeOfNode::createPrimitive() {}

REG_MKLDNN_PRIM_FOR(MKLDNNShapeOfNode, ShapeOf)
