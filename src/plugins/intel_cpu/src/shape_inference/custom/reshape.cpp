// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape.hpp"
#include <vector>
#include "utils.hpp"
#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {
namespace node {

Result ReshapeShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    static constexpr size_t RESHAPE_SRC = 0, RESHAPE_PATTERN = 1;
    const auto& inputShape = input_shapes[RESHAPE_SRC].get();
    const size_t inputShapeSize = inputShape.size();
    const auto memPtr = data_dependency.at(RESHAPE_PATTERN);
    const auto data = memPtr->getData();
    const auto& dims = memPtr->getStaticDims();
    const auto outputPatternSize = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<Dim>());
    std::vector<int64_t> outPattern = ov::get_raw_data_as<int64_t>(
                                          memPtr->getDesc().getPrecision(),
                                          data,
                                          outputPatternSize,
                                          ov::util::Cast<int64_t>());
    VectorDims outputShape(outputPatternSize);
    size_t outputProduct = 1;
    int32_t minusOneIdx = -1;
    int32_t minusOneCount = 0;
    for (int32_t i = 0; i < outputPatternSize; ++i) {
        if (outPattern[i] == 0 && m_specialZero && i < static_cast<int32_t>(inputShapeSize)) {
            outputShape[i] = inputShape[i];
        } else if (outPattern[i] == -1) {
            minusOneIdx = i;
            minusOneCount++;
        } else {
            outputShape[i] = outPattern[i];
            outputProduct *= outputShape[i];
        }
    }
    size_t inputProduct = 1;
    for (size_t i = 0; i < inputShapeSize; ++i) {
        if (static_cast<int>(i) < outputPatternSize && outPattern[i] == 0 && m_specialZero) {
            continue;
        }
        inputProduct *= inputShape[i];
    }
    if (minusOneIdx >= 0) {
        if (outputProduct != 0) {
            outputShape[minusOneIdx] = inputProduct / outputProduct;
            outputProduct *= outputShape[minusOneIdx];
        } else {
            outputShape[minusOneIdx] = 0;
        }
    }
    if (minusOneCount > 1  || inputProduct != outputProduct) {
        OPENVINO_THROW("[cpu]reshape: the shape of input data ", ov::intel_cpu::vec2str(inputShape),
                    " conflicts with the reshape pattern ", ov::intel_cpu::vec2str(outPattern));
    }
    return {{std::move(outputShape)}, ShapeInferStatus::success};
}

Result SqueezeShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    static constexpr size_t SQUEEZE_SRC = 0, SQUEEZE_PATTERN = 1;
    const auto& inputShape = input_shapes[SQUEEZE_SRC].get();
    const size_t inputShapeSize = inputShape.size();
    auto itr = data_dependency.find(SQUEEZE_PATTERN);
    VectorDims outputShape;
    outputShape.reserve(inputShapeSize);
    if (itr != data_dependency.end()) {
        const auto memPtr = data_dependency.at(SQUEEZE_PATTERN);
        const auto data = memPtr->getData();
        const auto& dims = memPtr->getStaticDims();
        if (dims.size() != 0) {
            const size_t outputPatternSize = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<Dim>());
            std::vector<int64_t> outPattern = ov::get_raw_data_as<int64_t>(
                                                  memPtr->getDesc().getPrecision(),
                                                  data,
                                                  outputPatternSize,
                                                  ov::util::Cast<int64_t>());
            std::vector<int64_t> originOutPattern = outPattern;
            std::vector<bool> removeMask(inputShapeSize, false);
            for (size_t i = 0; i < outputPatternSize; i++) {
                if (outPattern[i] < 0) {
                    outPattern[i] = inputShapeSize + outPattern[i];
                }
                if (outPattern[i] >= 0 && outPattern[i] < static_cast<int64_t>(inputShapeSize) && inputShape[outPattern[i]] == 1) {
                    removeMask[outPattern[i]] = true;
                }
            }
            for (size_t i = 0; i < inputShapeSize; i++) {
                if (!removeMask[i]) {
                    outputShape.push_back(inputShape[i]);
                }
            }
        } else {
            for (size_t i = 0; i < inputShapeSize; i++) {
                 if (inputShape[i] != 1) {
                     outputShape.push_back(inputShape[i]);
                 }
            }
        }
    } else {
        for (size_t i = 0; i < inputShapeSize; i++) {
            if (inputShape[i] != 1) {
                outputShape.push_back(inputShape[i]);
            }
        }
    }
    return {{std::move(outputShape)}, ShapeInferStatus::success};
}

Result UnsqueezeShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                  const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    static constexpr size_t UNSQUEEZE_SRC = 0, UNSQUEEZE_PATTERN = 1;
    const auto& inputShape = input_shapes[UNSQUEEZE_SRC].get();
    const size_t inputShapeSize = inputShape.size();
    const auto memPtr = data_dependency.at(UNSQUEEZE_PATTERN);
    const auto data = memPtr->getData();
    const auto& dims = memPtr->getStaticDims();
    size_t outputPatternSize = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<Dim>());
    std::vector<int64_t> originOutPattern = ov::get_raw_data_as<int64_t>(
                                          memPtr->getDesc().getPrecision(),
                                          data,
                                          outputPatternSize,
                                          ov::util::Cast<int64_t>());
    // remove repeated pattern
    std::unordered_set<int64_t> tmp(originOutPattern.begin(), originOutPattern.end());
    std::vector<int64_t> outPattern = std::vector<int64_t>(tmp.begin(), tmp.end());
    outputPatternSize = outPattern.size();
    size_t outputShapeSize = inputShapeSize + outputPatternSize;
    VectorDims outputShape(outputShapeSize, 0);
    bool existError = false;
    for (size_t i = 0; i < outputPatternSize; i++) {
        if (outPattern[i] < 0) {
            outPattern[i] = outputShapeSize + outPattern[i];
        }
        if (outPattern[i] >= 0 && outPattern[i] < static_cast<int64_t>(outputShapeSize)) {
            outputShape[outPattern[i]] = 1;
        } else {
            existError = true;
            break;
        }
    }
    for (size_t i = 0, y = 0; i < outputShapeSize; i++) {
        if (outputShape[i] == 0) {
            if (y < inputShapeSize) {
                outputShape[i] = inputShape[y];
                y++;
            } else {
                existError = true;
                break;
            }
        }
    }
    if (existError) {
        OPENVINO_THROW("[cpu]unsqueeze: the shape of input data ", ov::intel_cpu::vec2str(inputShape),
                " conflicts with the unsqueeze pattern ", ov::intel_cpu::vec2str(originOutPattern));
    }
    return {{std::move(outputShape)}, ShapeInferStatus::success};
}

ShapeInferPtr ReshapeShapeInferFactory::makeShapeInfer() const {
    if (const auto reshapeOp = ov::as_type_ptr<const ov::op::v1::Reshape>(m_op)) {
        return std::make_shared<ReshapeShapeInfer>(reshapeOp->get_special_zero());
    } else if (ov::is_type<ov::op::v0::Squeeze>(m_op)) {
        return std::make_shared<SqueezeShapeInfer>();
    } else if (ov::is_type<ov::op::v0::Unsqueeze>(m_op)) {
        return std::make_shared<UnsqueezeShapeInfer>();
    } else {
        OPENVINO_THROW("[cpu]reshape: ", m_op->get_type_name(), " is not implemented");
    }
}

} // namespace node
} // namespace intel_cpu
} // namespace ov
