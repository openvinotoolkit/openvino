// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nrmse.hpp"

#include <algorithm>
#include <openvino/runtime/make_tensor.hpp>
#include <openvino/core/parallel.hpp>

namespace {
template <typename InT>
void to_f32(const ov::Tensor& in, ov::Tensor& out) {
    OPENVINO_ASSERT(in.is_continuous());
    OPENVINO_ASSERT(out.is_continuous());
    OPENVINO_ASSERT(in.get_shape() == out.get_shape());

    if (ov::element::Type_t::f32 == in.get_element_type()) {
        in.copy_to(out);
        return;
    }

    const InT* in_buffer = in.data<InT>();
    OPENVINO_ASSERT(in_buffer != nullptr);
    const auto out_buffer = out.data<float>();
    OPENVINO_ASSERT(out_buffer != nullptr);

    // NOTE: ov::parallel_for takes care of splitting the work among threads such way,
    //       that the passed lambda function will be called sequentially
    //       on some part of "in.get_size()" range inside the each thread
    ov::parallel_for(in.get_size(), [in_buffer, out_buffer](int64_t index) {
        out_buffer[index] = static_cast<float>(in_buffer[index]);
    });
}

void to_f32(const ov::Tensor& in, ov::Tensor& out) {
    switch (in.get_element_type()) {
    case ov::element::Type_t::f32:
        ::to_f32<float>(in, out);
        break;
    case ov::element::Type_t::u64:
        ::to_f32<uint64_t>(in, out);
        break;
    case ov::element::Type_t::i64:
        ::to_f32<int64_t>(in, out);
        break;
    case ov::element::Type_t::u32:
        ::to_f32<uint32_t>(in, out);
        break;
    case ov::element::Type_t::i32:
        ::to_f32<int32_t>(in, out);
        break;
    case ov::element::Type_t::u16:
        ::to_f32<uint16_t>(in, out);
        break;
    case ov::element::Type_t::i16:
        ::to_f32<int16_t>(in, out);
        break;
    case ov::element::Type_t::u8:
        ::to_f32<uint8_t>(in, out);
        break;
    case ov::element::Type_t::i8:
        ::to_f32<int8_t>(in, out);
        break;
    case ov::element::Type_t::f16:
        ::to_f32<ov::float16>(in, out);
        break;
    case ov::element::Type_t::bf16:
        ::to_f32<ov::bfloat16>(in, out);
        break;
    default:
        OPENVINO_THROW("Unsupported precision {0}", in.get_element_type().get_type_name());
        break;
    }
}
} // anynomous namespace

ov::npuw::tests::metrics::NRMSE::NRMSE(double threshold) : m_threshold(threshold) {}

bool ov::npuw::tests::metrics::NRMSE::operator()(const ov::Tensor& actual,
                                                 const ov::Tensor& reference) const {
    OPENVINO_ASSERT(actual.is_continuous());
    OPENVINO_ASSERT(reference.is_continuous());
    OPENVINO_ASSERT(actual.get_shape() == reference.get_shape());
    // Check for alignment:
    OPENVINO_ASSERT(actual.get_byte_size() == reference.get_byte_size());
    // FIXME: Check for strides

    ov::Tensor actual_f32;
    ov::Tensor reference_f32;

    if (ov::element::Type_t::f32 == actual.get_element_type()) {
        actual_f32 = actual;
    } else {
        ov::Tensor dst(ov::element::Type_t::f32, actual.get_shape());
        to_f32(actual, dst);
        actual_f32 = std::move(dst);
    }

    if (ov::element::Type_t::f32 == reference.get_element_type()) {
        reference_f32 = reference;
    } else {
        ov::Tensor dst(ov::element::Type_t::f32, reference.get_shape());
        to_f32(reference, dst);
        reference_f32 = dst;
    }

    float* actual_data = actual_f32.data<float>();
    float* reference_data = reference_f32.data<float>();
    const std::size_t size = actual_f32.get_size();

    double squared_error{};
    for (size_t i = 0; i < size; ++i) {
        double diff = (actual_data[i] - reference_data[i]);
        squared_error += (diff * diff);
    }

    if (squared_error <= std::numeric_limits<double>::epsilon()) {
        std::cout << "NRMSE loss: 0.0, threshold: " << m_threshold << ".\n";
        return true;
    }

    double rmse = sqrt(squared_error / size);
    OPENVINO_ASSERT(rmse >= 0.0);

    auto actual_min_max = std::minmax_element(actual_data, actual_data + size);
    auto reference_min_max = std::minmax_element(reference_data, reference_data + size);
    double den = std::max({0.001f,
                           std::max(0.f, *reference_min_max.second) - std::min(0.f, *reference_min_max.first),
                           std::max(0.f, *actual_min_max.second) - std::min(0.f, *actual_min_max.first)});

    double nrmse = rmse / den;
    std::cout << "NRMSE loss: " << nrmse << ", threshold: " << m_threshold << ".\n";

    bool success = nrmse <= m_threshold;
    return success;
}
