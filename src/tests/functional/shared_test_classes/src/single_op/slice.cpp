// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/slice.hpp"

namespace ov {
namespace test {
std::string Slice8LayerTest::getTestCaseName(const testing::TestParamInfo<Slice8Params> &obj) {
    Slice8SpecificParams params;
    ov::element::Type model_type;
    std::string target_device;
    std::tie(params, model_type, target_device) = obj.param;
    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < params.shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({params.shapes[i].first})
               << (i < params.shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < params.shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < params.shapes.size(); j++) {
            result << ov::test::utils::vec2str(params.shapes[j].second[i]) << (j < params.shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "start="   << ov::test::utils::vec2str(params.start) << "_";
    result << "stop="    << ov::test::utils::vec2str(params.stop) << "_";
    result << "step="    << ov::test::utils::vec2str(params.step) << "_";
    result << "axes="    << ov::test::utils::vec2str(params.axes) << "_";
    result << "modelType=" << model_type.to_string() << "_";
    result << "trgDev="  << target_device;
    return result.str();
}

void Slice8LayerTest::SetUp() {
    Slice8SpecificParams test_params;
    ov::element::Type model_type;
    std::tie(test_params, model_type, targetDevice) = this->GetParam();

    init_input_shapes(test_params.shapes);

    auto param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes.front());
    ov::Shape const_shape = {test_params.start.size()};

    ASSERT_EQ(shape_size(const_shape), test_params.stop.size());
    ASSERT_EQ(shape_size(const_shape), test_params.step.size());

    auto begin_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, test_params.start.data());
    auto end_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, test_params.stop.data());
    auto stride_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, test_params.step.data());
    std::shared_ptr<ov::op::v8::Slice> slice;
    if (!test_params.axes.empty()) {
        ASSERT_EQ(shape_size(const_shape), test_params.axes.size());
        auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, test_params.axes.data());
        slice = std::make_shared<ov::op::v8::Slice>(param, begin_node, end_node, stride_node, axesNode);
    } else {
        slice = std::make_shared<ov::op::v8::Slice>(param, begin_node, end_node, stride_node);
    }
    function = std::make_shared<ov::Model>(slice->outputs(), ov::ParameterVector{param}, "Slice-8");
}
}  // namespace test
}  // namespace ov
