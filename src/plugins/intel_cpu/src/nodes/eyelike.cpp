// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eyelike.h"
#include <ie_ngraph_utils.hpp>

// #include <ngraph/opsets/opset1.hpp>

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
// using namespace mkldnn;
// using namespace InferenceEngine;
using namespace InferenceEngine::details;
bool Eye::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (false) {
            // !one_of(op->get_type_info(),
            //         ngraph::op::v0::Eye::get_type_info_static(),
            //         ngraph::op::v3::Eye::get_type_info_static())) {
            errorMessage = "Node is not an instance of Eye form the operation set v9.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Eye::Eye(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                     WeightsSharing::Ptr &cache) : Node(op, eng, cache), ngraphOp(op) {
    std::string errorMessage;
    if (isSupportedOperation(op, errorMessage)) {
        errorPrefix = "Eye layer with name '" + getName() + "' ";
        // if (ngraphOp->get_input_partial_shape(0).size() == 0)
        //     IE_THROW() << errorPrefix << "gets unsupported input 0D tensor (scalar)";  // prepare support
    } else {
        IE_THROW(NotImplemented) << errorMessage;
    }
    if (ngraphOp->get_output_element_type(0) != ngraph::element::i32) {
        IE_THROW() << errorPrefix << "doesn't support demanded output precision";
    }
}

void Eye::getSupportedDescriptors() {
    if (!descs.empty())
        return;
    if (!one_of(getParentEdges().size(), 1, 2, 3))  // alse check scalar
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

// void Eye::initSupportedPrimitiveDescriptors() {
//     if (!supportedPrimitiveDescriptors.empty())
//         return;
//     // outputConfigurators.emplace_back(LayoutType::ncsp, , outputShapes[i]);
//     // OV_SWITCH(intel_cpu, NonZeroExecute, ctx, inputPrec,
//     //           OV_CASE(Precision::FP32, float),
//     //           OV_CASE(Precision::BF16, bfloat16_t),
//     //           OV_CASE(Precision::I32, int),
//     //           OV_CASE(Precision::U32, uint32_t),
//     //           OV_CASE(Precision::I8, int8_t),
//     //           OV_CASE(Precision::U8, uint8_t))
//     addSupportedPrimDesc({{LayoutType::ncsp, Precision::I32}}, {{LayoutType::ncsp,
//         convertPrecision(ngraphOp->get_output_element_type(0))}}, impl_desc_type::ref);
// }

// for scalar
void Eye::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    std::vector<PortConfigurator> inDataConf;
    std::vector<PortConfigurator> outDataConf;

    if (true || !(getOriginalInputPrecisionAtPort(0) == Precision::I32 &&
            getOriginalInputPrecisionAtPort(1) == Precision::I32 &&
            getOriginalOutputPrecisionAtPort(0) == Precision::I32)) {
        inDataConf.reserve(inputShapes.size());
        for (int i = 0; i < inputShapes.size(); ++i)
            inDataConf.emplace_back(LayoutType::ncsp, Precision::I32);
        outDataConf.reserve(1);
        outDataConf.emplace_back(LayoutType::ncsp, Precision::I32);
        addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref);
    }
}
bool Eye::isExecutable() const {
    return true;
}

void Eye::execute(mkldnn::stream strm) {
    std::cout << "\nexeexe\n";
    // add scalar case
    // add i32 checking  - compare maxpooling
    // int ?
    size_t rowNum = getRowNum();
    size_t colNum = getColNum();
    int shift = 0; // get from attrs

    // memset(dst, 0, 1  * sizeof(int));  // fix
    std::cout << "\ncol, row=" << getRowNum() << "_" << getColNum() << "\n";
    // *dst = 1;
    if (isDynamicNode()) {
        VectorDims newDims{getRowNum(), getColNum()};
        redefineOutputMemory({newDims});
    }

    auto outPtr = getChildEdgeAt(0)->getMemoryPtr();

    int *dst = reinterpret_cast<int *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    std::cout << "\nssize=" << getChildEdgeAt(0)->getMemoryPtr()->GetSize() << "\n";

    // get real 1's count
    for (size_t i = 0 * shift; i < std::max(rowNum, colNum); i++) {
        dst[i + i * colNum] = 1;
    }
    // dst[0] = 1;
}

bool Eye::created() const {
    return getType() == Type::Eye;
}
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
