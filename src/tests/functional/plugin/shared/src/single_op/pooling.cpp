// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/pooling.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"

namespace ov {
namespace test {
using ov::test::utils::PoolingTypes;

std::string PoolingLayerTest::getTestCaseName(const testing::TestParamInfo<poolLayerTestParamsSet>& obj) {
    const auto& [pool_params, model_type, shapes, targetDevice] = obj.param;

    const auto& [pool_type, kernel, stride, pad_begin, pad_end, rounding_type, pad_type, excludePad] = pool_params;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    switch (pool_type) {
        case PoolingTypes::MAX:
            result << "MaxPool_";
            break;
        case PoolingTypes::AVG:
            result << "AvgPool_";
            result << "ExcludePad=" << excludePad << "_";
            break;
    }
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(pad_begin) << "_";
    result << "PE" << ov::test::utils::vec2str(pad_end) << "_";
    result << "Rounding=" << rounding_type << "_";
    result << "AutoPad=" << pad_type << "_";
    result << "modelType=" << model_type.get_type_name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void PoolingLayerTest::SetUp() {
    const auto& [pool_params, model_type, shapes, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    const auto& [pool_type, kernel, stride, pad_begin, pad_end, rounding_type, pad_type, excludePad] = pool_params;
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    std::shared_ptr<ov::Node> pooling;
    if (ov::test::utils::PoolingTypes::MAX == pool_type) {
        pooling = std::make_shared<ov::op::v1::MaxPool>(param, stride, pad_begin, pad_end, kernel, rounding_type, pad_type);
    } else {
        pooling = std::make_shared<ov::op::v1::AvgPool>(param, stride, pad_begin, pad_end, kernel, excludePad, rounding_type, pad_type);
    }

    auto result = std::make_shared<ov::op::v0::Result>(pooling);
    function = std::make_shared<ov::Model>(result, ov::ParameterVector{param}, "pooling");
}


std::string MaxPoolingV8LayerTest::getTestCaseName(const testing::TestParamInfo<maxPoolV8LayerTestParamsSet>& obj) {
    const auto& [pool_params, model_type, shapes, target_device] = obj.param;

    const auto& [kernel, stride, dilation, pad_begin, pad_end, index_element_type, axis, rounding_type, pad_type] =
        pool_params;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "D" << ov::test::utils::vec2str(dilation) << "_";
    result << "PB" << ov::test::utils::vec2str(pad_begin) << "_";
    result << "PE" << ov::test::utils::vec2str(pad_end) << "_";
    result << "IET" << index_element_type << "_";
    result << "A" << axis << "_";
    result << "Rounding=" << rounding_type << "_";
    result << "AutoPad=" << pad_type << "_";
    result << "modelType=" << model_type.get_type_name() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void MaxPoolingV8LayerTest::SetUp() {
    const auto& [pool_params, model_type, shapes, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;

    const auto& [kernel, stride, dilation, pad_begin, pad_end, index_element_type, axis, rounding_type, pad_type] =
        pool_params;
    init_input_shapes(shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());

    auto max_pool = std::make_shared<ov::op::v8::MaxPool>(param, stride, dilation, pad_begin, pad_end,
                                                          kernel, rounding_type, pad_type,
                                                          index_element_type, axis);

    const auto max_pool_v8_second_output_is_supported = targetDevice == ov::test::utils::DEVICE_GPU;

    ov::ResultVector results;
    if (max_pool_v8_second_output_is_supported) {
        results = {std::make_shared<ov::op::v0::Result>(max_pool->output(0)),
                   std::make_shared<ov::op::v0::Result>(max_pool->output(1))};
    } else {
        results = { std::make_shared<ov::op::v0::Result>(max_pool->output(0)) };
    }
    function = std::make_shared<ov::Model>(max_pool->outputs(), ov::ParameterVector{param}, "MaxPoolV8");
}

}  // namespace test
}  // namespace ov
