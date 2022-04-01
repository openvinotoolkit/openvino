// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eyelike.h"
#include <ie_ngraph_utils.hpp>
#include <utils/bfloat16.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
// using namespace mkldnn;
// using namespace InferenceEngine;
using namespace InferenceEngine::details;
bool Eye::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != ngraph::op::v9::Eye::get_type_info_static()) {
            errorMessage = "Node is not an instance of Eye form the operation set v9.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Eye::Eye(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                     WeightsSharing::Ptr &cache) : Node(op, eng, cache) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Eye layer with name '" + getName() + "' ";
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
    outType = op->get_output_element_type(0);
    if (!one_of(outType, ngraph::element::f32, ngraph::element::bf16,
        ngraph::element::i32, ngraph::element::u32,
        ngraph::element::i8, ngraph::element::u8)) {
        IE_THROW() << errorPrefix << "doesn't support demanded output precision";
    }
}

void Eye::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    if (!one_of(getParentEdges().size(), 3, 4))
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

namespace {
struct EyeContext {
    Eye &node;
};
}
template<typename T>
struct Eye::EyeExecute {
    void operator()(EyeContext & ctx) {
        ctx.node.executeSpecified<T>();
    }
};

void Eye::execute(mkldnn::stream strm) {
    auto outputPrec = getChildEdgesAtPort(0)[0]->getMemory().getDesc().getPrecision();
    EyeContext ctx = { *this };
    OV_SWITCH(intel_cpu, EyeExecute, ctx, outputPrec,
              OV_CASE(Precision::FP32, float),
              OV_CASE(Precision::BF16, bfloat16_t),
              OV_CASE(Precision::I32, int),
              OV_CASE(Precision::U32, uint32_t),
              OV_CASE(Precision::I8, int8_t),
              OV_CASE(Precision::U8, uint8_t))
}

void Eye::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    std::vector<PortConfigurator> inDataConf;
    std::vector<PortConfigurator> outDataConf;

    inDataConf.reserve(inputShapes.size());
    for (int i = 0; i < inputShapes.size(); ++i)
        inDataConf.emplace_back(LayoutType::ncsp, Precision::I32);
    outDataConf.reserve(1);
    outDataConf.emplace_back(LayoutType::ncsp, convertPrecision(outType));

    addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref);
}

bool Eye::isExecutable() const {
    return true;
}

template <typename T>
void Eye::executeSpecified() {
    size_t rowNum = getRowNum();
    size_t colNum = getColNum();
    int shift = getDiagIndex();
    if (isDynamicNode()) {
        VectorDims newDims{getRowNum(), getColNum()};
        redefineOutputMemory({newDims});
    }

    auto outPtr = getChildEdgeAt(0)->getMemoryPtr();

    T *dst = reinterpret_cast<T *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    // std::cout << "\nssize=" << getChildEdgeAt(0)->getMemoryPtr()->GetSize() << "\n";
    memset(dst, 0, colNum * rowNum * sizeof(int));

    size_t minSide = std::min(rowNum, colNum);
    size_t maxSide = std::max(rowNum, colNum);
    size_t absShift = std::abs(shift);
    size_t onesPerBatchNum = (absShift <= maxSide - minSide ? minSide :
                              absShift < maxSide ? minSide - absShift : 0);
    size_t dataShift = (shift >= 0 ? shift : -shift * colNum);
    for (size_t i = 0; i < onesPerBatchNum; i++) {
        dst[dataShift + i + i * colNum] = 1;
    }
}

bool Eye::created() const {
    return getType() == Type::Eye;
}
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
