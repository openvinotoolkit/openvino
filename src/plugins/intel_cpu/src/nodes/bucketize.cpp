// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bucketize.h"

#include <algorithm>
#include <shape_inference/shape_inference_pass_through.hpp>
#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "openvino/opsets/opset3.hpp"

namespace ov::intel_cpu::node {

bool Bucketize::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto bucketsize = ov::as_type_ptr<const ov::opset3::Bucketize>(op);
        if (!bucketsize) {
            errorMessage = "Only opset3 Bucketize operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Bucketize::Bucketize(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto bucketsize = ov::as_type_ptr<const ov::opset3::Bucketize>(op);
    if (bucketsize == nullptr) {
        THROW_CPU_NODE_ERR("is not an instance of Bucketize from opset3.");
    }

    if (getOriginalInputsNumber() != 2 || getOriginalOutputsNumber() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input/output edges!");
    }

    // check one attribute
    with_right = bucketsize->get_with_right_bound();
}

void Bucketize::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    // check precisions for input and output tensors
    input_precision = getOriginalInputPrecisionAtPort(INPUT_TENSOR_PORT);
    if (input_precision != ov::element::f32 && input_precision != ov::element::i32 &&
        input_precision != ov::element::i64) {
        input_precision = ov::element::f32;
    }
    boundaries_precision = getOriginalInputPrecisionAtPort(INPUT_BINS_PORT);
    if (boundaries_precision != ov::element::f32 && boundaries_precision != ov::element::i32 &&
        boundaries_precision != ov::element::i64) {
        boundaries_precision = ov::element::f32;
    }
    output_precision = getOriginalOutputPrecisionAtPort(OUTPUT_TENSOR_PORT);
    if (output_precision != ov::element::i32 && output_precision != ov::element::i64) {
        output_precision = ov::element::i32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, input_precision}, {LayoutType::ncsp, boundaries_precision}},
                         {{LayoutType::ncsp, output_precision}},
                         impl_desc_type::ref_any);
}

inline constexpr uint32_t getElementsMask(ov::element::Type precision1,
                                          ov::element::Type precision2,
                                          ov::element::Type precision3 = ov::element::dynamic,
                                          ov::element::Type precision4 = ov::element::dynamic) {
    return static_cast<uint32_t>(ov::element::Type_t(precision1)) |
           (static_cast<uint32_t>(ov::element::Type_t(precision2)) << 8) |
           (static_cast<uint32_t>(ov::element::Type_t(precision3)) << 16) |
           (static_cast<uint32_t>(ov::element::Type_t(precision4)) << 24);
}

void Bucketize::execute(const dnnl::stream& strm) {
    auto precision_mask = getElementsMask(input_precision, boundaries_precision, output_precision);

    switch (precision_mask) {
    case getElementsMask(ov::element::f32, ov::element::f32, ov::element::i32):
        bucketize<element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::f32, ov::element::f32, ov::element::i64):
        bucketize<element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    case getElementsMask(ov::element::f32, ov::element::i32, ov::element::i32):
        bucketize<element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::f32, ov::element::i32, ov::element::i64):
        bucketize<element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    case getElementsMask(ov::element::f32, ov::element::i64, ov::element::i32):
        bucketize<element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::f32, ov::element::i64, ov::element::i64):
        bucketize<element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    case getElementsMask(ov::element::i32, ov::element::f32, ov::element::i32):
        bucketize<element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::i32, ov::element::f32, ov::element::i64):
        bucketize<element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    case getElementsMask(ov::element::i32, ov::element::i32, ov::element::i32):
        bucketize<element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::i32, ov::element::i32, ov::element::i64):
        bucketize<element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    case getElementsMask(ov::element::i32, ov::element::i64, ov::element::i32):
        bucketize<element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::i32, ov::element::i64, ov::element::i64):
        bucketize<element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    case getElementsMask(ov::element::i64, ov::element::f32, ov::element::i32):
        bucketize<element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::i64, ov::element::f32, ov::element::i64):
        bucketize<element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::f32>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    case getElementsMask(ov::element::i64, ov::element::i32, ov::element::i32):
        bucketize<element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::i64, ov::element::i32, ov::element::i64):
        bucketize<element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i32>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    case getElementsMask(ov::element::i64, ov::element::i64, ov::element::i32):
        bucketize<element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i32>::value_type>();
        break;
    case getElementsMask(ov::element::i64, ov::element::i64, ov::element::i64):
        bucketize<element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i64>::value_type,
                  element_type_traits<ov::element::i64>::value_type>();
        break;
    default:
        THROW_CPU_NODE_ERR("has unsupported precision: ", precision_mask);
    }
}

void Bucketize::prepareParams() {
    auto inputTensorMemPtr = getSrcMemoryAtPort(INPUT_TENSOR_PORT);
    auto inputBinsMemPtr = getSrcMemoryAtPort(INPUT_BINS_PORT);
    auto dstMemPtr = getDstMemoryAtPort(0);
    if (!dstMemPtr || !dstMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has destination memory undefined.");
    }
    if (!inputTensorMemPtr || !inputTensorMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has input tensor undefined.");
    }
    if (!inputBinsMemPtr || !inputBinsMemPtr->isDefined()) {
        THROW_CPU_NODE_ERR("has input bins undefined.");
    }
    if (getSelectedPrimitiveDescriptor() == nullptr) {
        THROW_CPU_NODE_ERR("has preferable primitive descriptors unset.");
    }

    // update with_bins/num_values/num_bin_values
    auto input_tensor_dims = inputTensorMemPtr->getStaticDims();
    if (input_tensor_dims.size() < 1) {
        THROW_CPU_NODE_ERR("has incorrect dimensions of the input.");
    }
    auto input_bin_dims = inputBinsMemPtr->getStaticDims();
    if (input_bin_dims.size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect dimensions of the boundaries tensor.");
    }
    if (input_bin_dims[0] != 0) {
        with_bins = true;
    }
    num_bin_values = input_bin_dims[0];

    num_values = std::accumulate(input_tensor_dims.begin(),
                                 input_tensor_dims.end(),
                                 static_cast<size_t>(1),
                                 std::multiplies<>());
}

bool Bucketize::neverExecute() const {
    return getSelectedPrimitiveDescriptor()->hasZeroInputDimsAtPort(0);
}

bool Bucketize::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

template <typename T, typename T_BOUNDARIES, typename T_IND>
void Bucketize::bucketize() {
    const auto* input_data = getSrcDataAtPortAs<const T>(0);
    const auto* boundaries_data = getSrcDataAtPortAs<const T_BOUNDARIES>(1);
    auto* output_data = getDstDataAtPortAs<T_IND>(0);

    if (!with_bins) {
        memset(output_data, 0, num_values * sizeof(T_IND));
        return;
    }

    // boundaries are assumed to be sorted and to have unique elements
    parallel_for(num_values, [&](size_t ind) {
        T value = input_data[ind];
        if (with_right) {
            auto low = std::lower_bound(boundaries_data, boundaries_data + num_bin_values, value);
            output_data[ind] = static_cast<T_IND>(low - boundaries_data);
        } else {
            auto up = std::upper_bound(boundaries_data, boundaries_data + num_bin_values, value);
            output_data[ind] = static_cast<T_IND>(up - boundaries_data);
        }
    });
}

bool Bucketize::created() const {
    return getType() == Type::Bucketize;
}

}  // namespace ov::intel_cpu::node
