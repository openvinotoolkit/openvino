// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eye.h"
#include "openvino/op/eye.hpp"
#include <utils/bfloat16.hpp>
#include "openvino/core/parallel.hpp"
#include "shape_inference/shape_inference_ngraph.hpp"
#include "utils/bfloat16.hpp"

#define THROW_ERROR(...) OPENVINO_THROW(NameFromType(getType()), " node with name '", getName(), "' ", __VA_ARGS__)

namespace ov {
namespace intel_cpu {
namespace node {
using namespace ov::intel_cpu;

bool Eye::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != ov::op::v9::Eye::get_type_info_static()) {
            errorMessage = "Node is not an instance of Eye form the operation set v9.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

namespace {
class EyeShapeInferFactory : public ShapeInferFactory {
public:
    EyeShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        IShapeInfer::port_mask_t port_mask = EMPTY_PORT_MASK;
        if (m_op->get_input_size() == 4) {
            port_mask =  PortMask(Eye::ROWS_NUM, Eye::COLS_NUM, Eye::DIAGONAL_INDEX, Eye::BATCH_SHAPE);
        } else {
            port_mask =  PortMask(Eye::ROWS_NUM, Eye::COLS_NUM, Eye::DIAGONAL_INDEX);
        }
        return std::make_shared<NgraphShapeInfer>(make_shape_inference(m_op), port_mask);
    }
private:
    std::shared_ptr<ov::Node> m_op;
};
} // namespace

Eye::Eye(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) : Node(op, context, EyeShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
            OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    outType = op->get_output_element_type(0);
    withBatchShape = (op->get_input_size() == 4);
    if (!one_of(outType, ov::element::f32, ov::element::bf16,
        ov::element::i32, ov::element::i8, ov::element::u8)) {
        THROW_ERROR(errorPrefix, "doesn't support demanded output precision");
    }
}

void Eye::getSupportedDescriptors() {
    if (!one_of(getParentEdges().size(), 3u, 4u))
        THROW_ERROR(errorPrefix, "has incorrect number of input edges: ", getParentEdges().size());
    if (getChildEdges().empty())
        THROW_ERROR(errorPrefix, "has incorrect number of output edges: ", getChildEdges().size());
}

template<typename T>
struct Eye::EyeExecute {
    void operator()(Eye *node) {
        node->executeSpecified<T>();
    }
};

void Eye::execute(dnnl::stream strm) {
    auto outputPrec = getChildEdgeAt(0)->getMemory().getDesc().getPrecision();
    OV_SWITCH(intel_cpu, EyeExecute, this, outputPrec,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::bf16, bfloat16_t),
              OV_CASE(ov::element::i32, int),
              OV_CASE(ov::element::i8, int8_t),
              OV_CASE(ov::element::u8, uint8_t))
}

void Eye::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    std::vector<PortConfigurator> inDataConf;
    std::vector<PortConfigurator> outDataConf;

    inDataConf.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); ++i)
        inDataConf.emplace_back(LayoutType::ncsp, ov::element::i32);
    outDataConf.reserve(1);
    outDataConf.emplace_back(LayoutType::ncsp, outType);

    addSupportedPrimDesc(inDataConf, outDataConf, impl_desc_type::ref);
}

template <typename T>
void Eye::executeSpecified() {
    const size_t rowNum = getRowNum();
    const size_t colNum = getColNum();
    const int64_t shift = getDiagIndex();
    auto outPtr = getDstMemoryAtPort(0);
    if (!outPtr || !outPtr ->isDefined())
        THROW_ERROR(errorPrefix, "Destination memory is undefined.");
    T *dst = outPtr->getDataAs<T>();

    const size_t batchVolume = getBatchVolume(getBatchShape());
    const size_t spatialCount = colNum * rowNum;
    const size_t spatialSize = spatialCount * sizeof(T);
    const size_t l2CacheSize = dnnl::utils::get_cache_size(2, true);
    const size_t elementsCount = colNum * rowNum * batchVolume;

    const int64_t countByColumns = std::max(int64_t(colNum) - std::abs(shift), int64_t(0));
    const int64_t countByRows = std::max(int64_t(rowNum) - std::abs(shift), int64_t(0));
    const size_t onesPerBatchNum =
        static_cast<size_t>(shift > 0 ? std::min(countByColumns, int64_t(rowNum)) : std::min(countByRows, int64_t(colNum)));
    const size_t dataShift = static_cast<size_t>(shift >= 0 ? shift : -shift * colNum);

    if (spatialSize >= l2CacheSize) {
        parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
            size_t start = 0, end = 0;
            splitter(elementsCount, nthr, ithr, start, end);
            memset(dst + start, 0, (end - start) * sizeof(T));
        });
        if (onesPerBatchNum == 0) return;
        for (size_t bShift = 0; bShift < batchVolume * spatialCount; bShift += spatialCount) {
            parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
                size_t start = 0, end = 0;
                splitter(onesPerBatchNum, nthr, ithr, start, end);
                for (size_t j = start; j < end; j++) {
                    dst[dataShift + j * (colNum + 1) + bShift] = static_cast<T>(1);
                }
            });
        }
    } else {
        parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
            size_t start = 0, end = 0;
            splitter(batchVolume, nthr, ithr, start, end);
            memset(dst + start * spatialCount, 0, (end - start) * spatialSize);
            if (onesPerBatchNum == 0) return;
            for (size_t spShift = start * spatialCount; spShift < end * spatialCount; spShift += spatialCount) {
                for (size_t j = 0; j < onesPerBatchNum; j++) {
                    dst[dataShift + j * (colNum + 1) + spShift] = static_cast<T>(1);
                }
            }
        });
    }
}

bool Eye::created() const {
    return getType() == Type::Eye;
}
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
