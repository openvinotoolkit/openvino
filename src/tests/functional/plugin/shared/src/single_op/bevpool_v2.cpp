// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/bevpool_v2.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/bevpool_v2.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace test {

std::string BevPoolV2LayerTest::getTestCaseName(const testing::TestParamInfo<BevPoolV2Params>& obj) {
    std::ostringstream result;
    const auto& [input_shapes, feature_type, index_type, dev] = obj.param;

    for (size_t s = 0; s < input_shapes.size(); ++s) {
        const auto& shape_item = input_shapes[s];
        result << "IS" << s << "=(" << shape_item.first << ")_TS=";
        for (const auto& ts : shape_item.second) {
            result << "{" << ov::test::utils::vec2str(ts) << "}_";
        }
    }

    result << "FeatureType=" << feature_type << "_";
    result << "IndexType=" << index_type << "_";
    result << "Device=" << dev;
    return result.str();
}

void BevPoolV2LayerTest::SetUp() {
    const auto& [input_shapes, feature_type, index_type, dev] = this->GetParam();
    targetDevice = dev;
    abs_threshold = 2e-3;
    rel_threshold = 2e-3;

    init_input_shapes(input_shapes);

    const auto cf = std::make_shared<ov::op::v0::Parameter>(feature_type, inputDynamicShapes[0]);
    const auto dw = std::make_shared<ov::op::v0::Parameter>(feature_type, inputDynamicShapes[1]);
    const auto idx = std::make_shared<ov::op::v0::Parameter>(index_type, inputDynamicShapes[2]);
    const auto itv = std::make_shared<ov::op::v0::Parameter>(index_type, inputDynamicShapes[3]);

    constexpr uint32_t input_channels = 4;
    constexpr uint32_t output_channels = 2;
    constexpr uint32_t image_width = 5;
    constexpr uint32_t image_height = 3;
    constexpr uint32_t feature_width = 7;
    constexpr uint32_t feature_height = 6;

    const ov::op::v15::Bound x_bound{-10.f, 10.f, 0.5f};
    const ov::op::v15::Bound y_bound{-10.f, 10.f, 0.5f};
    const ov::op::v15::Bound z_bound{-5.f, 3.f, 0.5f};
    const ov::op::v15::Bound d_bound{0.f, 2.f, 1.f};

    const auto bevpool = std::make_shared<ov::op::v15::BevPoolV2>(ov::OutputVector{cf, dw, idx, itv},
                                                                   input_channels,
                                                                   output_channels,
                                                                   image_width,
                                                                   image_height,
                                                                   feature_width,
                                                                   feature_height,
                                                                   x_bound,
                                                                   y_bound,
                                                                   z_bound,
                                                                   d_bound);

    function = std::make_shared<ov::Model>(bevpool->outputs(), ov::ParameterVector{cf, dw, idx, itv});
}

void BevPoolV2LayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& func_inputs = function->inputs();

    const auto cf_type = func_inputs[0].get_element_type();
    const auto idx_type = func_inputs[2].get_element_type();

    ov::test::utils::InputGenerateData feature_data;
    feature_data.start_from = 1;
    feature_data.range = 20;

    auto cf_tensor = ov::test::utils::create_and_fill_tensor(cf_type, targetInputStaticShapes[0], feature_data);
    auto dw_tensor = ov::test::utils::create_and_fill_tensor(cf_type, targetInputStaticShapes[1], feature_data);

    const auto m = targetInputStaticShapes[2].empty() ? 0 : targetInputStaticShapes[2][0];
    std::vector<int64_t> idx_values(m, 0);
    for (size_t i = 0; i < m; ++i) {
        idx_values[i] = static_cast<int64_t>(i % 8);
    }

    std::vector<int64_t> itv_values = {0, static_cast<int64_t>(std::min<size_t>(m, 2)), 0,
                                       static_cast<int64_t>(std::min<size_t>(m, 2)), static_cast<int64_t>(m), 1};

    ov::Tensor idx_tensor(idx_type, targetInputStaticShapes[2]);
    ov::Tensor itv_tensor(idx_type, targetInputStaticShapes[3]);

    if (idx_type == ov::element::i32) {
        auto* p_idx = idx_tensor.data<int32_t>();
        for (size_t i = 0; i < m; ++i) {
            p_idx[i] = static_cast<int32_t>(idx_values[i]);
        }

        auto* p_itv = itv_tensor.data<int32_t>();
        for (size_t i = 0; i < itv_values.size(); ++i) {
            p_itv[i] = static_cast<int32_t>(itv_values[i]);
        }
    } else {
        auto* p_idx = idx_tensor.data<int64_t>();
        for (size_t i = 0; i < m; ++i) {
            p_idx[i] = idx_values[i];
        }

        auto* p_itv = itv_tensor.data<int64_t>();
        for (size_t i = 0; i < itv_values.size(); ++i) {
            p_itv[i] = itv_values[i];
        }
    }

    inputs[func_inputs[0].get_node_shared_ptr()] = cf_tensor;
    inputs[func_inputs[1].get_node_shared_ptr()] = dw_tensor;
    inputs[func_inputs[2].get_node_shared_ptr()] = idx_tensor;
    inputs[func_inputs[3].get_node_shared_ptr()] = itv_tensor;
}

const BevPoolV2LayerTest::TGenData BevPoolV2LayerTest::GetTestDataForDevice(const char* deviceName) {
    const std::vector<std::vector<InputShape>> input_shapes = {
        {
            {{-1, 3, 5, 4}, {{1, 3, 5, 4}, {2, 3, 5, 4}}},
            {{-1, 2, 3, 5}, {{1, 2, 3, 5}, {2, 2, 3, 5}}},
            {{-1}, {{6}, {9}}},
            {{6}, {{6}, {6}}},
        },
        {
            {{1, 3, 5, 4}, {{1, 3, 5, 4}}},
            {{1, 2, 3, 5}, {{1, 2, 3, 5}}},
            {{6}, {{6}}},
            {{6}, {{6}}},
        },
    };

    const std::vector<ov::element::Type> feature_types = {ov::element::f32, ov::element::f16};
    const std::vector<ov::element::Type> index_types = {ov::element::i32, ov::element::i64};

    auto data = ::testing::Combine(::testing::ValuesIn(input_shapes),
                                   ::testing::ValuesIn(feature_types),
                                   ::testing::ValuesIn(index_types),
                                   ::testing::Values(deviceName));
    return data;
}

}  // namespace test
}  // namespace ov
