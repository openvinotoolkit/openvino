//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "tensor_utils.hpp"

#include "data_type_converters.hpp"

#include <openvino/core/except.hpp>
#include <openvino/core/type.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/element_type_traits.hpp>

#include <fstream>

namespace {

template <class InT, class OutT>
void convertTensorPrecisionImpl(const ov::Tensor& in, const ov::Tensor& out) {
    const auto inputBuffer = in.data<const InT>();
    OPENVINO_ASSERT(inputBuffer != nullptr, "Tensor was not allocated");

    const auto outputBuffer = out.data<OutT>();
    OPENVINO_ASSERT(outputBuffer != nullptr, "Tensor was not allocated");

    for (size_t index = 0; index < in.get_size(); ++index) {
        outputBuffer[index] = npu::utils::convertValuePrecision<OutT>(inputBuffer[index]);
    }
}

}  // namespace

namespace npu {
namespace utils {

void copyTensor(const ov::Tensor& in, const ov::Tensor& out) {
    OPENVINO_ASSERT(in.get_element_type() == out.get_element_type(), "Precision mismatch");
    OPENVINO_ASSERT(in.get_shape() == out.get_shape(), "Shape mismatch");

    const auto inputBuffer = in.data<const uint8_t>();
    OPENVINO_ASSERT(inputBuffer != nullptr, "Tensor was not allocated");

    const auto outputBuffer = out.data<uint8_t>();
    OPENVINO_ASSERT(outputBuffer != nullptr, "Tensor was not allocated");

    std::copy_n(inputBuffer, in.get_byte_size(), outputBuffer);
}

void convertTensorPrecision(const ov::Tensor& in, const ov::Tensor& out) {
    OPENVINO_ASSERT(in.get_shape() == out.get_shape(), "Mismatch in Dims");

    const ov::element::Type& inPrecision = in.get_element_type();
    const ov::element::Type& outPrecision = out.get_element_type();

    if (inPrecision == outPrecision) {
        copyTensor(in, out);
        return;
    }

#define CASE(InT, OutT)                                                                       \
    convertTensorPrecisionImpl<ov::fundamental_type_for<ov::element::Type_t::InT>,            \
                               ov::fundamental_type_for<ov::element::Type_t::OutT>>(in, out); \
    break

    switch (inPrecision) {
    case ov::element::Type_t::f64: {
        switch (outPrecision) {
        case ov::element::Type_t::f32:
            CASE(f64, f32);
        case ov::element::Type_t::u64:
            CASE(f64, u64);
        case ov::element::Type_t::i64:
            CASE(f64, i64);
        case ov::element::Type_t::u32:
            CASE(f64, u32);
        case ov::element::Type_t::i32:
            CASE(f64, i32);
        case ov::element::Type_t::u16:
            CASE(f64, u16);
        case ov::element::Type_t::i16:
            CASE(f64, i16);
        case ov::element::Type_t::u8:
            CASE(f64, u8);
        case ov::element::Type_t::i8:
            CASE(f64, i8);
        case ov::element::Type_t::f16:
            CASE(f64, f16);
        case ov::element::Type_t::bf16:
            CASE(f64, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::f32: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(f32, f64);
        case ov::element::Type_t::u64:
            CASE(f32, u64);
        case ov::element::Type_t::i64:
            CASE(f32, i64);
        case ov::element::Type_t::u32:
            CASE(f32, u32);
        case ov::element::Type_t::i32:
            CASE(f32, i32);
        case ov::element::Type_t::u16:
            CASE(f32, u16);
        case ov::element::Type_t::i16:
            CASE(f32, i16);
        case ov::element::Type_t::u8:
            CASE(f32, u8);
        case ov::element::Type_t::i8:
            CASE(f32, i8);
        case ov::element::Type_t::f16:
            CASE(f32, f16);
        case ov::element::Type_t::bf16:
            CASE(f32, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::f16: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(f16, f64);
        case ov::element::Type_t::f32:
            CASE(f16, f32);
        case ov::element::Type_t::bf16:
            CASE(f16, bf16);
        case ov::element::Type_t::u64:
            CASE(f16, u64);
        case ov::element::Type_t::i64:
            CASE(f16, i64);
        case ov::element::Type_t::u32:
            CASE(f16, u32);
        case ov::element::Type_t::i32:
            CASE(f16, i32);
        case ov::element::Type_t::u16:
            CASE(f16, u16);
        case ov::element::Type_t::i16:
            CASE(f16, i16);
        case ov::element::Type_t::u8:
            CASE(f16, u8);
        case ov::element::Type_t::i8:
            CASE(f16, i8);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::bf16: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(bf16, f64);
        case ov::element::Type_t::f32:
            CASE(bf16, f32);
        case ov::element::Type_t::f16:
            CASE(bf16, f16);
        case ov::element::Type_t::u64:
            CASE(bf16, u64);
        case ov::element::Type_t::i64:
            CASE(bf16, i64);
        case ov::element::Type_t::u32:
            CASE(bf16, u32);
        case ov::element::Type_t::i32:
            CASE(bf16, i32);
        case ov::element::Type_t::u16:
            CASE(bf16, u16);
        case ov::element::Type_t::i16:
            CASE(bf16, i16);
        case ov::element::Type_t::u8:
            CASE(bf16, u8);
        case ov::element::Type_t::i8:
            CASE(bf16, i8);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::u64: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(u64, f64);
        case ov::element::Type_t::f32:
            CASE(u64, f32);
        case ov::element::Type_t::i64:
            CASE(u64, i64);
        case ov::element::Type_t::u32:
            CASE(u64, u32);
        case ov::element::Type_t::i32:
            CASE(u64, i32);
        case ov::element::Type_t::u16:
            CASE(u64, u16);
        case ov::element::Type_t::i16:
            CASE(u64, i16);
        case ov::element::Type_t::u8:
            CASE(u64, u8);
        case ov::element::Type_t::i8:
            CASE(u64, i8);
        case ov::element::Type_t::f16:
            CASE(u64, f16);
        case ov::element::Type_t::bf16:
            CASE(u64, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::i64: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(i64, f64);
        case ov::element::Type_t::f32:
            CASE(i64, f32);
        case ov::element::Type_t::u64:
            CASE(i64, u64);
        case ov::element::Type_t::u32:
            CASE(i64, u32);
        case ov::element::Type_t::i32:
            CASE(i64, i32);
        case ov::element::Type_t::u16:
            CASE(i64, u16);
        case ov::element::Type_t::i16:
            CASE(i64, i16);
        case ov::element::Type_t::u8:
            CASE(i64, u8);
        case ov::element::Type_t::i8:
            CASE(i64, i8);
        case ov::element::Type_t::f16:
            CASE(i64, f16);
        case ov::element::Type_t::bf16:
            CASE(i64, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::u32: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(u32, f64);
        case ov::element::Type_t::f32:
            CASE(u32, f32);
        case ov::element::Type_t::u64:
            CASE(u32, u64);
        case ov::element::Type_t::i64:
            CASE(u32, i64);
        case ov::element::Type_t::i32:
            CASE(u32, i32);
        case ov::element::Type_t::u16:
            CASE(u32, u16);
        case ov::element::Type_t::i16:
            CASE(u32, i16);
        case ov::element::Type_t::u8:
            CASE(u32, u8);
        case ov::element::Type_t::i8:
            CASE(u32, i8);
        case ov::element::Type_t::f16:
            CASE(u32, f16);
        case ov::element::Type_t::bf16:
            CASE(u32, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::i32: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(i32, f64);
        case ov::element::Type_t::f32:
            CASE(i32, f32);
        case ov::element::Type_t::u64:
            CASE(i32, u64);
        case ov::element::Type_t::i64:
            CASE(i32, i64);
        case ov::element::Type_t::u32:
            CASE(i32, u32);
        case ov::element::Type_t::u16:
            CASE(i32, u16);
        case ov::element::Type_t::i16:
            CASE(i32, i16);
        case ov::element::Type_t::u8:
            CASE(i32, u8);
        case ov::element::Type_t::i8:
            CASE(i32, i8);
        case ov::element::Type_t::f16:
            CASE(i32, f16);
        case ov::element::Type_t::bf16:
            CASE(i32, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::u16: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(u16, f64);
        case ov::element::Type_t::f32:
            CASE(u16, f32);
        case ov::element::Type_t::u64:
            CASE(u16, u64);
        case ov::element::Type_t::i64:
            CASE(u16, i64);
        case ov::element::Type_t::u32:
            CASE(u16, u32);
        case ov::element::Type_t::i32:
            CASE(u16, i32);
        case ov::element::Type_t::i16:
            CASE(u16, i16);
        case ov::element::Type_t::u8:
            CASE(u16, u8);
        case ov::element::Type_t::i8:
            CASE(u16, i8);
        case ov::element::Type_t::f16:
            CASE(u16, f16);
        case ov::element::Type_t::bf16:
            CASE(u16, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::i16: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(i16, f64);
        case ov::element::Type_t::f32:
            CASE(i16, f32);
        case ov::element::Type_t::u64:
            CASE(i16, u64);
        case ov::element::Type_t::i64:
            CASE(i16, i64);
        case ov::element::Type_t::u32:
            CASE(i16, u32);
        case ov::element::Type_t::i32:
            CASE(i16, i32);
        case ov::element::Type_t::u16:
            CASE(i16, u16);
        case ov::element::Type_t::u8:
            CASE(i16, u8);
        case ov::element::Type_t::i8:
            CASE(i16, i8);
        case ov::element::Type_t::f16:
            CASE(i16, f16);
        case ov::element::Type_t::bf16:
            CASE(i16, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::u8: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(u8, f64);
        case ov::element::Type_t::f32:
            CASE(u8, f32);
        case ov::element::Type_t::u64:
            CASE(u8, u64);
        case ov::element::Type_t::i64:
            CASE(u8, i64);
        case ov::element::Type_t::u32:
            CASE(u8, u32);
        case ov::element::Type_t::i32:
            CASE(u8, i32);
        case ov::element::Type_t::u16:
            CASE(u8, u16);
        case ov::element::Type_t::i16:
            CASE(u8, i16);
        case ov::element::Type_t::i8:
            CASE(u8, i8);
        case ov::element::Type_t::f16:
            CASE(u8, f16);
        case ov::element::Type_t::bf16:
            CASE(u8, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    case ov::element::Type_t::i8: {
        switch (outPrecision) {
        case ov::element::Type_t::f64:
            CASE(i8, f64);
        case ov::element::Type_t::f32:
            CASE(i8, f32);
        case ov::element::Type_t::u64:
            CASE(i8, u64);
        case ov::element::Type_t::i64:
            CASE(i8, i64);
        case ov::element::Type_t::u32:
            CASE(i8, u32);
        case ov::element::Type_t::i32:
            CASE(i8, i32);
        case ov::element::Type_t::u16:
            CASE(i8, u16);
        case ov::element::Type_t::i16:
            CASE(i8, i16);
        case ov::element::Type_t::u8:
            CASE(i8, u8);
        case ov::element::Type_t::f16:
            CASE(i8, f16);
        case ov::element::Type_t::bf16:
            CASE(i8, bf16);
        default:
            OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                           outPrecision.get_type_name());
        }
        break;
    }
    default:
        OPENVINO_THROW("Unsupported combination of precisions ", inPrecision.get_type_name(), " -> ",
                       outPrecision.get_type_name());
    }

#undef CASE
}

ov::Tensor toPrecision(const ov::Tensor& in, const ov::element::Type& precision, void* ptr) {
    if (in.get_element_type() == precision && ptr == nullptr) {
        return in;
    }

    ov::Tensor out;

    if (ptr == nullptr) {
        out = ov::Tensor(precision, in.get_shape());
    } else {
        out = ov::Tensor(precision, in.get_shape(), ptr);
    }

    convertTensorPrecision(in, out);

    return out;
}

std::vector<std::vector<float>> parseTensorsAsFP32(const std::map<std::string, ov::Tensor>& tensors) {
    std::vector<std::vector<float>> results;

    for (const auto& tensor : tensors) {
        const ov::Tensor tensorFP32 = toFP32(tensor.second);
        const auto dataBuffer = tensorFP32.data<float>();
        OPENVINO_ASSERT(dataBuffer != nullptr);

        const size_t size = tensorFP32.get_size();
        std::vector<float> result(size);
        std::copy_n(dataBuffer, size, result.begin());

        results.push_back(result);
    }

    return results;
}

ov::Tensor joinTensors(const std::list<ov::Tensor>& tensors, const ov::Layout& layout) {
    if (tensors.empty()) {
        OPENVINO_THROW("Cannot join tensors: nothing to join");
    }
    if (!ov::layout::has_batch(layout)) {
        OPENVINO_THROW("Cannot join tensors: has no batch_idx in layout", layout.to_string());
    }
    auto pivotShape = tensors.front().get_shape();
    auto pivotPrecision = tensors.front().get_element_type();
    if (!std::all_of(tensors.begin(), tensors.end(), [&pivotShape, &pivotPrecision](const auto& t) {
            return t.get_shape() == pivotShape && t.get_element_type() == pivotPrecision;
        })) {
        OPENVINO_THROW("Cannot join tensors with different shapes, expected: ", pivotPrecision, ", ", pivotShape);
    }
    pivotShape[ov::layout::batch_idx(layout)] *= tensors.size();
    ov::Tensor out(pivotPrecision, pivotShape);
    const auto outputBuffer = out.data();
    size_t bytesOffset = 0;
    for (const auto& t : tensors) {
        memcpy(reinterpret_cast<unsigned char*>(outputBuffer) + bytesOffset, t.data(), t.get_byte_size());
        bytesOffset += t.get_byte_size();
    }
    return out;
}

std::list<ov::Tensor> splitBatchedTensor(const ov::Tensor &tensor, const ov::Layout& layout, size_t parts) {
    if (!parts) {
        OPENVINO_THROW("Cannot split tensor on parts: ", parts);
    }
    auto pivotShape = tensor.get_shape();
    if (!ov::layout::has_batch(layout)) {
        OPENVINO_THROW("Cannot split tensor: has no batch_idx in layout", layout.to_string());
    }
    auto pivotPrecision = tensor.get_element_type();
    if (pivotShape[ov::layout::batch_idx(layout)] % parts != 0) {
        OPENVINO_THROW("Cannot split tensor with batch size: ", pivotShape[ov::layout::batch_idx(layout)], " on: ", parts ," equal tensors");
    }
    pivotShape[ov::layout::batch_idx(layout)] /= parts;
    std::list<ov::Tensor> ret;
    const auto *inputBuffer = tensor.data<unsigned char>();
    for (size_t i = 0; i < parts; i ++) {
        ov::Tensor out(pivotPrecision, pivotShape);
        memcpy(out.data<unsigned char>(), inputBuffer, out.get_byte_size());
        inputBuffer += out.get_byte_size();
        ret.push_back(std::move(out));
    }
    return ret;
}
}  // namespace utils
}  // namespace npu
