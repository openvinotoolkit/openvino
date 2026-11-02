// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <plugin/transformations/transpose_fc_decompression_params.hpp>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

const ov::Shape weights_shape{32, 16};

// Values below 16 fit every tested type; with transposed = true the [rows, cols] data is stored as [cols, rows]
std::shared_ptr<ov::op::v0::Constant> make_param(const ov::element::Type& type, const ov::Shape& shape, bool transposed) {
    const size_t rows = shape[0], cols = shape[1];
    std::vector<float> values(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const auto value = static_cast<float>((r * cols + c) % 15 + 1);
            values[transposed ? c * rows + r : r * cols + c] = value;
        }
    }
    return ov::op::v0::Constant::create(type, transposed ? ov::Shape{cols, rows} : shape, values);
}

// zp_type == dynamic: no zero point
std::shared_ptr<ov::Model> getModel(const ov::Shape& param_shape, const ov::element::Type& zp_type, bool scale_transposed, bool zp_transposed) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(weights_shape[1])});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, weights_shape, {1});
    auto no_bias = std::make_shared<op::Placeholder>();
    auto scale = make_param(ov::element::f16, param_shape, scale_transposed);
    std::shared_ptr<op::FullyConnectedCompressed> fc;
    if (zp_type == ov::element::dynamic) {
        fc = std::make_shared<op::FullyConnectedCompressed>(input, weights, no_bias, scale, ov::element::f16, true, !scale_transposed);
    } else {
        auto zp = make_param(zp_type, param_shape, zp_transposed);
        fc = std::make_shared<op::FullyConnectedCompressed>(input, weights, no_bias, scale, zp, ov::element::f16, true, !scale_transposed, !zp_transposed);
    }
    return std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input});
}

// scale / zp shape [N, G], zp type, scale transposed, zp transposed
using TransposeFCDecompressionParamsParams = std::tuple<ov::Shape, ov::element::Type, bool, bool>;

class TransposeFCDecompressionParamsTests : public testing::WithParamInterface<TransposeFCDecompressionParamsParams>, public TransformationTestsF {
public:
    TransposeFCDecompressionParamsTests() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

    static std::string getTestCaseName(const testing::TestParamInfo<TransposeFCDecompressionParamsParams>& info) {
        const auto& [param_shape, zp_type, scale_transposed, zp_transposed] = info.param;
        std::ostringstream ss;
        ss << "param_shape=" << param_shape << "_zp=" << zp_type << "_scale_transposed=" << scale_transposed << "_zp_transposed=" << zp_transposed;
        return ss.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [param_shape, zp_type, scale_transposed, zp_transposed] = GetParam();
        model = getModel(param_shape, zp_type, false, false);
        if (scale_transposed || zp_transposed)
            model_ref = getModel(param_shape, zp_type, scale_transposed, zp_transposed);
        manager.register_pass<TransposeFCDecompressionParams>();
    }
};

TEST_P(TransposeFCDecompressionParamsTests, Transform) {}

INSTANTIATE_TEST_SUITE_P(smoke_TransposeFCDecompressionParams,
                         TransposeFCDecompressionParamsTests,
                         ::testing::Values(TransposeFCDecompressionParamsParams{{32, 4}, ov::element::dynamic, true, false},
                                           TransposeFCDecompressionParamsParams{{32, 4}, ov::element::u8, true, true},
                                           TransposeFCDecompressionParamsParams{{32, 4}, ov::element::f16, true, true},
                                           // sub-byte zero point keeps its layout
                                           TransposeFCDecompressionParamsParams{{32, 4}, ov::element::u4, true, false},
                                           // a single group has the same memory order in both layouts
                                           TransposeFCDecompressionParamsParams{{32, 1}, ov::element::u8, false, false},
                                           // broadcast over N
                                           TransposeFCDecompressionParamsParams{{1, 4}, ov::element::u8, false, false}),
                         TransposeFCDecompressionParamsTests::getTestCaseName);

TEST_F(TransformationTestsF, TransposeFCDecompressionParams_AlreadyGroupMajor) {
    model = getModel({32, 4}, ov::element::u8, true, true);
    manager.register_pass<TransposeFCDecompressionParams>();
}

// The transposed constant keeps the weights-file attribute and records the transpose for the weightless cache
TEST_F(TransformationTestsF, TransposeFCDecompressionParams_WeightlessCacheAttribute) {
    auto with_weightless_attr = [](std::shared_ptr<ov::Model> m, bool transposed) {
        for (const auto& op : m->get_ordered_ops()) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
            if (!constant || constant->get_element_type() != ov::element::f16)
                continue;
            constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
                ov::WeightlessCacheAttribute(constant->get_byte_size(), 128, ov::element::f16);
            if (transposed)
                constant->get_rt_info()["gpu_weightless_transpose_order"] = std::vector<size_t>{1, 0};
        }
        return m;
    };
    model = with_weightless_attr(getModel({32, 4}, ov::element::dynamic, false, false), false);
    model_ref = with_weightless_attr(getModel({32, 4}, ov::element::dynamic, true, false), true);
    manager.register_pass<TransposeFCDecompressionParams>();
}

// A scale shared by two FCs and a Multiply: each FC gets a transposed copy, the Multiply keeps the original
TEST_F(TransformationTestsF, TransposeFCDecompressionParams_SharedScale) {
    auto make_model = [](bool transposed) {
        auto scale = make_param(ov::element::f16, {32, 4}, false);
        ov::OutputVector outputs;
        ov::ParameterVector params;
        for (int i = 0; i < 2; ++i) {
            auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(weights_shape[1])});
            auto weights = ov::op::v0::Constant::create(ov::element::u8, weights_shape, {1});
            auto fc_scale = transposed ? make_param(ov::element::f16, {32, 4}, true) : scale;
            auto no_bias = std::make_shared<op::Placeholder>();
            outputs.push_back(std::make_shared<op::FullyConnectedCompressed>(input, weights, no_bias, fc_scale, ov::element::f16, true, !transposed));
            params.push_back(input);
        }
        auto other = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 4});
        outputs.push_back(std::make_shared<ov::op::v1::Multiply>(other, scale));
        params.push_back(other);
        return std::make_shared<ov::Model>(outputs, params);
    };
    model = make_model(false);
    model_ref = make_model(true);
    manager.register_pass<TransposeFCDecompressionParams>();
}

TEST_F(TransformationTestsF, TransposeFCDecompressionParams_NonConstantScale) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, static_cast<int64_t>(weights_shape[1])});
    auto weights = ov::op::v0::Constant::create(ov::element::u8, weights_shape, {1});
    auto no_bias = std::make_shared<op::Placeholder>();
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{32, 4});
    auto fc = std::make_shared<op::FullyConnectedCompressed>(input, weights, no_bias, scale, ov::element::f16);
    model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input, scale});
    manager.register_pass<TransposeFCDecompressionParams>();
}

}  // namespace
