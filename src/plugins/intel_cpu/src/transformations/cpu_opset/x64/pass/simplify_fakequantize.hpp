// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/opsets/opset8.hpp>
#include <vector>

#include "openvino/opsets/opset1.hpp"

namespace ov::intel_cpu {

inline std::vector<float> simplifyToScale(const std::shared_ptr<ov::opset8::FakeQuantize>& fq_node,
                                          float threshold = 0.0001f) {
    auto levels = fq_node->get_levels();
    auto input_low = ov::as_type_ptr<ov::opset8::Constant>(fq_node->get_input_node_shared_ptr(1))->cast_vector<float>();
    auto input_high =
        ov::as_type_ptr<ov::opset8::Constant>(fq_node->get_input_node_shared_ptr(2))->cast_vector<float>();
    auto output_low =
        ov::as_type_ptr<ov::opset8::Constant>(fq_node->get_input_node_shared_ptr(3))->cast_vector<float>();
    auto output_high =
        ov::as_type_ptr<ov::opset8::Constant>(fq_node->get_input_node_shared_ptr(4))->cast_vector<float>();

    std::vector<float> cl, ch, isc, ish, osc, osh;
    for (float i : input_low) {
        cl.push_back(i);
    }
    for (float i : input_high) {
        ch.push_back(i);
    }

    for (size_t i = 0; i < std::max(input_low.size(), input_high.size()); i++) {
        float il = input_low[input_low.size() == 1 ? 0 : i];
        float ih = input_high[input_high.size() == 1 ? 0 : i];

        isc.push_back((levels - 1) / (ih - il));
        ish.push_back(-il * (levels - 1) / (ih - il));
    }

    for (size_t i = 0; i < std::max(output_low.size(), output_high.size()); i++) {
        float ol = output_low[output_low.size() == 1 ? 0 : i];
        float oh = output_high[output_high.size() == 1 ? 0 : i];

        osc.push_back((oh - ol) / (levels - 1));
        osh.push_back(ol);
    }

    std::vector<float> outScale;

    if (fq_node->get_output_element_type(0) == ov::element::u8 &&
        std::all_of(cl.cbegin(),
                    cl.cend(),
                    [](float val) {
                        return val == 0.0f;
                    }) &&
        std::all_of(ish.cbegin(),
                    ish.cend(),
                    [](float val) {
                        return val == 0.0f;
                    }) &&
        std::all_of(osc.cbegin(),
                    osc.cend(),
                    [](float val) {
                        return val == 1.0f;
                    }) &&
        std::all_of(osh.cbegin(), osh.cend(), [](float val) {
            return val == 0.0f;
        })) {
        outScale = isc;
    }

    if (fq_node->get_output_element_type(0) == ov::element::i8 &&
        std::all_of(ish.cbegin(),
                    ish.cend(),
                    [&threshold](float val) {
                        return std::abs(val - 128.f) < threshold;
                    }) &&
        std::all_of(osc.cbegin(),
                    osc.cend(),
                    [](float val) {
                        return val == 1.f;
                    }) &&
        std::all_of(osh.cbegin(), osh.cend(), [&threshold](float val) {
            return std::abs(val + 128.f) < threshold;
        })) {
        bool isCropAligned = true;
        for (size_t i = 0; i < std::max(cl.size(), isc.size()); i++) {
            if (std::abs(cl[cl.size() == 1 ? 0 : i] * isc[isc.size() == 1 ? 0 : i] + 128.f) > threshold) {
                isCropAligned = false;
            }
        }

        for (size_t i = 0; i < std::max(ch.size(), isc.size()); i++) {
            if (std::abs(ch[ch.size() == 1 ? 0 : i] * isc[isc.size() == 1 ? 0 : i] - 127.f) > threshold) {
                isCropAligned = false;
            }
        }

        if (isCropAligned) {
            outScale = isc;
        }
    }

    return outScale;
}

}  // namespace ov::intel_cpu
