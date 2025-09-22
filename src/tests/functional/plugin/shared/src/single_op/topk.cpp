// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/topk.hpp"
#include <random>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "openvino/op/topk.hpp"

namespace ov {
namespace test {
std::string TopKLayerTest::getTestCaseName(const testing::TestParamInfo<TopKParams>& obj) {
    const auto& [keepK, axis, mode, sort, model_type, input_shapes, target_device] = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "k=" << keepK << "_";
    result << "axis=" << axis << "_";
    result << "mode=" << mode << "_";
    result << "sort=" << sort << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void TopKLayerTest::SetUp() {
    const auto& [keepK, axis, mode, sort, model_type, input_shapes, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;
    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, &keepK);
    auto topk = std::make_shared<ov::op::v1::TopK>(param, k, axis, mode, sort);
    function = std::make_shared<ov::Model>(topk->outputs(), ov::ParameterVector{param}, "TopK");
}

std::string TopK11LayerTest::getTestCaseName(const testing::TestParamInfo<TopK11Params>& obj) {
    const auto& [keepK, axis, mode, sort, model_type, input_shapes, stable, target_device] = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < input_shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({input_shapes[i].first})
               << (i < input_shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < input_shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < input_shapes.size(); j++) {
            result << ov::test::utils::vec2str(input_shapes[j].second[i]) << (j < input_shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "k=" << keepK << "_";
    result << "axis=" << axis << "_";
    result << "mode=" << mode << "_";
    result << "sort=" << sort << "_";
    result << "stable=" << stable << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev=" << target_device;
    return result.str();
}

void TopK11LayerTest::SetUp() {
    const auto& [keepK, axis, mode, sort, model_type, input_shapes, stable, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;
    init_input_shapes(input_shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, &keepK);
    auto topk = std::make_shared<ov::op::v11::TopK>(param, k, axis, mode, sort, ov::element::i32, stable);
    function = std::make_shared<ov::Model>(topk->outputs(), ov::ParameterVector{param}, "TopK");
}
}  // namespace test
}  // namespace ov
