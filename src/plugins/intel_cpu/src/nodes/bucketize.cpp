// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <algorithm>

#include <openvino/opsets/opset3.hpp>
#include <shape_inference/shape_inference_pass_through.hpp>
#include "openvino/core/parallel.hpp"
#include "bucketize.h"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

bool Bucketize::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto bucketsize = std::dynamic_pointer_cast<const ov::opset3::Bucketize>(op);
        if (!bucketsize) {
            errorMessage = "Only opset3 Bucketize operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

Bucketize::Bucketize(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Bucketize layer with name '" + op->get_friendly_name() + "' ";
    const auto bucketsize = std::dynamic_pointer_cast<const ov::opset3::Bucketize>(op);
    if (bucketsize == nullptr)
        IE_THROW() << "Operation with name '" << op->get_friendly_name() <<
            "' is not an instance of Bucketize from opset3.";

    if (getOriginalInputsNumber() != 2 || getOriginalOutputsNumber() != 1) {
        IE_THROW() << errorPrefix << " has incorrect number of input/output edges!";
    }

    // check one attribute
    with_right = bucketsize->get_with_right_bound();
}

void Bucketize::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    // check precisions for input and output tensors
    input_precision = getOriginalInputPrecisionAtPort(INPUT_TENSOR_PORT);
    if (input_precision != Precision::FP32 && input_precision != Precision::I32 &&
        input_precision != Precision::I64) {
        input_precision = Precision::FP32;
    }
    boundaries_precision = getOriginalInputPrecisionAtPort(INPUT_BINS_PORT);
    if (boundaries_precision != Precision::FP32 && boundaries_precision != Precision::I32 &&
        boundaries_precision != Precision::I64) {
        boundaries_precision = Precision::FP32;
    }
    output_precision = getOriginalOutputPrecisionAtPort(OUTPUT_TENSOR_PORT);
    if (output_precision != Precision::I32 && output_precision != Precision::I64) {
        output_precision = Precision::I32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, input_precision},
                          {LayoutType::ncsp, boundaries_precision}},
                         {{LayoutType::ncsp, output_precision}},
                         impl_desc_type::ref_any);
}

void Bucketize::execute(dnnl::stream strm) {
    auto precision_mask = getPrecisionMask(input_precision, boundaries_precision, output_precision);

    switch (precision_mask) {
        case getPrecisionMask(Precision::FP32, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::FP32, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        case getPrecisionMask(Precision::FP32, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::FP32, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        case getPrecisionMask(Precision::FP32, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::FP32, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::I32, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::I32, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        case getPrecisionMask(Precision::I32, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::I32, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        case getPrecisionMask(Precision::I64, Precision::FP32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::I64, Precision::FP32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::FP32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        case getPrecisionMask(Precision::I64, Precision::I32, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::I64, Precision::I32, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        case getPrecisionMask(Precision::I64, Precision::I64, Precision::I32):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I32>::value_type>();
            break;
        case getPrecisionMask(Precision::I64, Precision::I64, Precision::I64):
            bucketize<PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type,
                    PrecisionTrait<Precision::I64>::value_type>();
            break;
        default:
            IE_THROW() << errorPrefix << " has unsupported precision: " << precision_mask;
    }
}

void Bucketize::prepareParams() {
    auto inputTensorMemPtr = getParentEdgeAt(INPUT_TENSOR_PORT)->getMemoryPtr();
    auto inputBinsMemPtr = getParentEdgeAt(INPUT_BINS_PORT)->getMemoryPtr();
    auto dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->isAllocated())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!inputTensorMemPtr || !inputTensorMemPtr->isAllocated())
        IE_THROW() << "Input tensor didn't allocate.";
    if (!inputBinsMemPtr || !inputBinsMemPtr->isAllocated())
        IE_THROW() << "Input bins didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    // update with_bins/num_values/num_bin_values
    auto input_tensor_dims = inputTensorMemPtr->getStaticDims();
    if (input_tensor_dims.size() < 1) {
        IE_THROW() << errorPrefix << " has incorrect dimensions of the input.";
    }
    auto input_bin_dims = inputBinsMemPtr->getStaticDims();
    if (input_bin_dims.size() != 1) {
        IE_THROW() << errorPrefix << " has incorrect dimensions of the boundaries tensor.";
    }
    if (input_bin_dims[0] != 0) {
        with_bins = true;
    }
    num_bin_values = input_bin_dims[0];

    num_values =
        std::accumulate(input_tensor_dims.begin(), input_tensor_dims.end(), size_t(1), std::multiplies<size_t>());
}

bool Bucketize::isExecutable() const {
    return !isInputTensorAtPortEmpty(0);
}

template <typename T, typename T_BOUNDARIES, typename T_IND>
void Bucketize::bucketize() {
    const auto *input_data = reinterpret_cast<const T *>(getParentEdgeAt(0)->getMemoryPtr()->getData());
    const auto *boundaries_data = reinterpret_cast<const T_BOUNDARIES *>(getParentEdgeAt(1)->getMemoryPtr()->getData());
    auto *output_data = reinterpret_cast<T_IND *>(getChildEdgesAtPort(0)[0]->getMemoryPtr()->getData());

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

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
