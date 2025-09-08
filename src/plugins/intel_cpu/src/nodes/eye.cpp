// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eye.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utils/bfloat16.hpp>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "onednn/dnnl.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/eye.hpp"
#include "selective_build.h"
#include "shape_inference/shape_inference_cpu.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {
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

Eye::Eye(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    outType = op->get_output_element_type(0);
    withBatchShape = (op->get_input_size() == 4);
    if (none_of(outType, ov::element::f32, ov::element::bf16, ov::element::i32, ov::element::i8, ov::element::u8)) {
        CPU_NODE_THROW("doesn't support demanded output precision");
    }
}

void Eye::getSupportedDescriptors() {
    if (none_of(getParentEdges().size(), 3U, 4U)) {
        CPU_NODE_THROW("has incorrect number of input edges: ", getParentEdges().size());
    }
    if (getChildEdges().empty()) {
        CPU_NODE_THROW("has incorrect number of output edges: ", getChildEdges().size());
    }
}

template <typename T>
struct Eye::EyeExecute {
    void operator()(Eye* node) {
        node->executeSpecified<T>();
    }
};

void Eye::execute([[maybe_unused]] const dnnl::stream& strm) {
    auto outputPrec = getChildEdgeAt(0)->getMemory().getDesc().getPrecision();
    OV_SWITCH(intel_cpu,
              EyeExecute,
              this,
              outputPrec,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::bf16, bfloat16_t),
              OV_CASE(ov::element::i32, int),
              OV_CASE(ov::element::i8, int8_t),
              OV_CASE(ov::element::u8, uint8_t))
}

void Eye::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    std::vector<PortConfigurator> inDataConf;
    std::vector<PortConfigurator> outDataConf;

    inDataConf.reserve(inputShapes.size());
    for (size_t i = 0; i < inputShapes.size(); ++i) {
        inDataConf.emplace_back(LayoutType::ncsp, ov::element::i32);
    }
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
    if (!outPtr || !outPtr->isDefined()) {
        CPU_NODE_THROW("Destination memory is undefined.");
    }
    T* dst = outPtr->getDataAs<T>();

    const size_t batchVolume = getBatchVolume(getBatchShape());
    const size_t spatialCount = colNum * rowNum;
    const size_t spatialSize = spatialCount * sizeof(T);
    const size_t l2CacheSize = dnnl::utils::get_cache_size(2, true);
    const size_t elementsCount = colNum * rowNum * batchVolume;

    const int64_t countByColumns = std::max(static_cast<int64_t>(colNum) - std::abs(shift), static_cast<int64_t>(0));
    const int64_t countByRows = std::max(static_cast<int64_t>(rowNum) - std::abs(shift), static_cast<int64_t>(0));
    const size_t onesPerBatchNum =
        static_cast<size_t>(shift > 0 ? std::min(countByColumns, static_cast<int64_t>(rowNum))
                                      : std::min(countByRows, static_cast<int64_t>(colNum)));
    const auto dataShift = (shift >= 0 ? shift : -shift * colNum);

    if (spatialSize >= l2CacheSize) {
        parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
            size_t start = 0;
            size_t end = 0;
            splitter(elementsCount, nthr, ithr, start, end);
            std::fill(dst + start, dst + end, static_cast<T>(0));
        });
        if (onesPerBatchNum == 0) {
            return;
        }
        for (size_t bShift = 0; bShift < batchVolume * spatialCount; bShift += spatialCount) {
            parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
                size_t start = 0;
                size_t end = 0;
                splitter(onesPerBatchNum, nthr, ithr, start, end);
                for (size_t j = start; j < end; j++) {
                    dst[dataShift + j * (colNum + 1) + bShift] = static_cast<T>(1);
                }
            });
        }
    } else {
        parallel_nt(0, [&](const size_t ithr, const size_t nthr) {
            size_t start = 0;
            size_t end = 0;
            splitter(batchVolume, nthr, ithr, start, end);
            std::fill(dst + start * spatialCount, dst + end * spatialCount, static_cast<T>(0));
            if (onesPerBatchNum == 0) {
                return;
            }
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
}  // namespace ov::intel_cpu::node
