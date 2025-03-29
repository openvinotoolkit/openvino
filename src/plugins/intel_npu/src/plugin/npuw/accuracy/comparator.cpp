// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "comparator.hpp"

#include <algorithm>
#include <openvino/runtime/make_tensor.hpp>

#include "../logging.hpp"
#include "../util.hpp"

ov::npuw::metrics::NRMSE::NRMSE(double threshold) : m_threshold(threshold) {}

bool ov::npuw::metrics::NRMSE::operator()(const ov::SoPtr<ov::ITensor>& actual,
                                          const ov::SoPtr<ov::ITensor>& reference) const {
    NPUW_ASSERT(actual->is_continuous());
    NPUW_ASSERT(reference->is_continuous());
    NPUW_ASSERT(actual->get_shape() == reference->get_shape());
    // Check for alignment:
    NPUW_ASSERT(actual->get_byte_size() == reference->get_byte_size());
    // FIXME: Check for strides

    ov::Tensor actual_f32;
    ov::Tensor reference_f32;

    if (ov::element::Type_t::f32 == actual->get_element_type()) {
        actual_f32 = ov::make_tensor(actual);
    } else {
        ov::Tensor dst(ov::element::Type_t::f32, actual->get_shape());
        ov::npuw::util::to_f32(ov::make_tensor(actual), dst);
        actual_f32 = std::move(dst);
    }

    if (ov::element::Type_t::f32 == reference->get_element_type()) {
        reference_f32 = ov::make_tensor(reference);
    } else {
        ov::Tensor dst(ov::element::Type_t::f32, reference->get_shape());
        ov::npuw::util::to_f32(ov::make_tensor(reference), dst);
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
        LOG_INFO("NRMSE loss: 0.0, threshold: " << m_threshold << ".");
        LOG_INFO("PASS");
        return true;
    }

    double rmse = sqrt(squared_error / size);
    NPUW_ASSERT(rmse >= 0.0);

    auto actual_min_max = std::minmax_element(actual_data, actual_data + size);
    auto reference_min_max = std::minmax_element(reference_data, reference_data + size);
    double den = std::max({0.001f,
                           std::max(0.f, *reference_min_max.second) - std::min(0.f, *reference_min_max.first),
                           std::max(0.f, *actual_min_max.second) - std::min(0.f, *actual_min_max.first)});

    double nrmse = rmse / den;
    LOG_INFO("NRMSE loss: " << nrmse << ", threshold: " << m_threshold << ".");

    bool success = nrmse <= m_threshold;
    LOG_INFO((success ? "PASS" : "FAIL"));
    return success;
}
