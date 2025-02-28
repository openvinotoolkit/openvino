// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "ov_ops/rms.hpp"

#include "shared_test_classes/base/utils/compare_results.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

namespace ov {
namespace test {
namespace utils {

namespace {
void compare(const std::shared_ptr<ov::Node> &node,
             size_t port,
             const ov::element::Type& inference_precision,
             const ov::Tensor &expected,
             const ov::Tensor &actual,
             double abs_threshold,
             double rel_threshold,
             double topk_threshold,
             double mvn_threshold) {
    ov::test::utils::compare(expected, actual, inference_precision, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);
}

void compare(const std::shared_ptr<ov::op::v0::DetectionOutput> &node,
             size_t port,
             const ov::element::Type& inference_precision,
             const ov::Tensor &expected,
             const ov::Tensor &actual,
             double abs_threshold,
             double rel_threshold,
             double topk_threshold,
             double mvn_threshold) {
        ASSERT_EQ(expected.get_size(), actual.get_size());

        size_t expSize = 0;
        size_t actSize = 0;

        const float* expBuf = expected.data<const float>();
        const float* actBuf = actual.data<const float>();
        ASSERT_NE(expBuf, nullptr);
        ASSERT_NE(actBuf, nullptr);

        for (size_t i = 0; i < actual.get_size(); i+=7) {
            if (expBuf[i] == -1)
                break;
            expSize += 7;
        }
        for (size_t i = 0; i < actual.get_size(); i+=7) {
            if (actBuf[i] == -1)
                break;
            actSize += 7;
        }
        ASSERT_EQ(expSize, actSize);
        ov::test::utils::compare(expected, actual, inference_precision, 1e-2f, rel_threshold, topk_threshold, mvn_threshold);
}

namespace color_conversion {
template <typename T>
inline void validate_colors(const T* expected, const T* actual, size_t size, float dev_threshold, float abs_threshold = 0.01f) {
    size_t mismatches = 0;
    for (size_t i = 0; i < size; i++) {
        if (std::abs(static_cast<float>(expected[i] - actual[i])) > abs_threshold) {
            mismatches++;
        }
    }
    ASSERT_LT(mismatches / size, dev_threshold) << mismatches <<
        " out of " << size << " color mismatches found which exceeds allowed threshold " << dev_threshold;
}

inline void validate_colors(const ov::Tensor& expected, const ov::Tensor& actual, float dev_threshold, float abs_threshold = 0.01f) {
    OPENVINO_ASSERT(expected.get_size() == actual.get_size());
    OPENVINO_ASSERT(expected.get_element_type() == actual.get_element_type());

#define CASE(X)                                                             \
    case X:                                                                 \
        validate_colors(                                                    \
                static_cast<ov::fundamental_type_for<X>*>(expected.data()), \
                static_cast<ov::fundamental_type_for<X>*>(actual.data()),   \
                                                     expected.get_size(),   \
                                                     dev_threshold,         \
                                                     abs_threshold);        \
        break;
    switch (expected.get_element_type()) {
        CASE(ov::element::Type_t::i8)
        CASE(ov::element::Type_t::i16)
        CASE(ov::element::Type_t::i32)
        CASE(ov::element::Type_t::i64)
        CASE(ov::element::Type_t::u8)
        CASE(ov::element::Type_t::u16)
        CASE(ov::element::Type_t::u32)
        CASE(ov::element::Type_t::u64)
        CASE(ov::element::Type_t::bf16)
        CASE(ov::element::Type_t::f16)
        CASE(ov::element::Type_t::f32)
        CASE(ov::element::Type_t::f64)
    default:
        OPENVINO_THROW("Unsupported element type: ", expected.get_element_type());
    }
#undef CASE
}
} // namespace color_conversion

void compare(const std::shared_ptr<ov::op::v8::I420toRGB> &node,
             size_t port,
             const ov::element::Type& inference_precision,
             const ov::Tensor &expected,
             const ov::Tensor &actual,
             double abs_threshold,
             double rel_threshold,
             double topk_threshold,
             double mvn_threshold) {
    ov::test::utils::compare(expected, actual, inference_precision, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);

    // Allow less than 2% of deviations with 1 color step. 2% is experimental value
    // For different calculation methods - 1.4% deviation is observed
    color_conversion::validate_colors(expected, actual, 0.02);
}

void compare(const std::shared_ptr<ov::op::v8::I420toBGR> &node,
             size_t port,
             const ov::element::Type& inference_precision,
             const ov::Tensor &expected,
             const ov::Tensor &actual,
             double abs_threshold,
             double rel_threshold,
             double topk_threshold,
             double mvn_threshold) {
    ov::test::utils::compare(expected, actual, inference_precision, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);

    // Allow less than 2% of deviations with 1 color step. 2% is experimental value
    // For different calculation methods - 1.4% deviation is observed
    color_conversion::validate_colors(expected, actual, 0.02);
}

void compare(const std::shared_ptr<ov::op::v8::NV12toRGB> &node,
             size_t port,
             const ov::element::Type& inference_precision,
             const ov::Tensor &expected,
             const ov::Tensor &actual,
             double abs_threshold,
             double rel_threshold,
             double topk_threshold,
             double mvn_threshold) {
    ov::test::utils::compare(expected, actual, inference_precision, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);

    // Allow less than 2% of deviations with 1 color step. 2% is experimental value
    // For different calculation methods - 1.4% deviation is observed
    color_conversion::validate_colors(expected, actual, 0.02);
}

void compare(const std::shared_ptr<ov::op::v8::NV12toBGR> &node,
             size_t port,
             const ov::element::Type& inference_precision,
             const ov::Tensor &expected,
             const ov::Tensor &actual,
             double abs_threshold,
             double rel_threshold,
             double topk_threshold,
             double mvn_threshold) {
    ov::test::utils::compare(expected, actual, inference_precision, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);

    // Allow less than 2% of deviations with 1 color step. 2% is experimental value
    // For different calculation methods - 1.4% deviation is observed
    color_conversion::validate_colors(expected, actual, 0.02);
}

template<typename T>
void compareResults(const std::shared_ptr<ov::Node> &node,
                    size_t port,
                    const ov::element::Type& inference_precision,
                    const ov::Tensor &expected,
                    const ov::Tensor &actual,
                    double abs_threshold,
                    double rel_threshold,
             double topk_threshold,
             double mvn_threshold) {
    return compare(ov::as_type_ptr<T>(node), port, inference_precision, expected, actual, abs_threshold, rel_threshold, topk_threshold, mvn_threshold);
}

} // namespace

CompareMap getCompareMap() {
OPENVINO_SUPPRESS_DEPRECATED_START
    CompareMap compareMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), compareResults<NAMESPACE::NAME>},
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#include "openvino/opsets/opset9_tbl.hpp"
#include "openvino/opsets/opset10_tbl.hpp"
#include "openvino/opsets/opset11_tbl.hpp"
#include "openvino/opsets/opset12_tbl.hpp"
#include "openvino/opsets/opset13_tbl.hpp"
#include "openvino/opsets/opset14_tbl.hpp"
#include "openvino/opsets/opset15_tbl.hpp"
#include "openvino/opsets/opset16_tbl.hpp"

#include "ov_ops/opset_private_tbl.hpp"
#undef _OPENVINO_OP_REG
    };
OPENVINO_SUPPRESS_DEPRECATED_END
    return compareMap;
}

} // namespace utils
} // namespace test
} // namespace ov
