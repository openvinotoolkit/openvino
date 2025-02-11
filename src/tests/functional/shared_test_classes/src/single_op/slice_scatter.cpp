// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/slice_scatter.hpp"

namespace ov {
namespace test {
std::string SliceScatterLayerTest::getTestCaseName(const testing::TestParamInfo<SliceScatterParams> &obj) {
    SliceScatterSpecificParams params;
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

void SliceScatterLayerTest::SetUp() {
    SliceScatterSpecificParams test_params;
    ov::element::Type model_type;
    std::tie(test_params, model_type, targetDevice) = this->GetParam();

    init_input_shapes(test_params.shapes);

    auto data = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]);
    auto updates = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]);
    ov::Shape const_shape = {test_params.start.size()};

    ASSERT_EQ(shape_size(const_shape), test_params.stop.size());
    ASSERT_EQ(shape_size(const_shape), test_params.step.size());

    auto begin_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, test_params.start.data());
    auto end_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, test_params.stop.data());
    auto stride_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, test_params.step.data());
    std::shared_ptr<ov::op::v15::SliceScatter> slice_scatter;
    if (!test_params.axes.empty()) {
        ASSERT_EQ(shape_size(const_shape), test_params.axes.size());
        auto axesNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, const_shape, test_params.axes.data());
        slice_scatter = std::make_shared<ov::op::v15::SliceScatter>(data, updates, begin_node, end_node, stride_node, axesNode);
    } else {
        slice_scatter = std::make_shared<ov::op::v15::SliceScatter>(data, updates, begin_node, end_node, stride_node);
    }
    function = std::make_shared<ov::Model>(slice_scatter->outputs(), ov::ParameterVector{data, updates}, "SliceScatter-15");
}
}  // namespace test
}  // namespace ov
