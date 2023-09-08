// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/bucketize.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {

std::string BucketizeLayerTest::getTestCaseName(const testing::TestParamInfo<bucketizeParamsTuple>& obj) {
    std::vector<InputShape> shapes;
    bool with_right_bound;
    ov::element::Type in_data_type, in_buckets_type, model_type;
    std::string target_device;

    std::tie(shapes, with_right_bound, in_data_type, in_buckets_type, model_type, target_device) = obj.param;

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
    if (with_right_bound)
        result << "rightIntervalEdge_";
    else
        result << "leftIntervalEdge_";
    result << "in_data_type=" << in_data_type.get_type_name() << "_";
    result << "in_buckets_type=" << in_buckets_type.get_type_name() << "_";
    result << "model_type=" << model_type.get_type_name() << "_";
    result << "target_device=" << target_device;
    return result.str();
}

void BucketizeLayerTest::generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) {
    inputs.clear();
    auto params = function->get_parameters();
    OPENVINO_ASSERT(target_input_static_shapes.size() >= params.size());
    for (int i = 0; i < params.size(); i++) {
        auto name = params[i]->get_friendly_name();
        ov::Tensor tensor;
        if (name == "a_data") {
            auto shape_size = ov::shape_size(target_input_static_shapes[i]);
            tensor = ov::test::utils::create_and_fill_tensor(params[i]->get_element_type(), target_input_static_shapes[i], shape_size * 5, 0, 10, 7235346);
        } else if (name == "b_buckets") {
            tensor = ov::test::utils::create_and_fill_tensor_unique_sequence(params[i]->get_element_type(), target_input_static_shapes[i], 0, 10, 8234231);
        }
        inputs.insert({params[i], tensor});
    }
}

void BucketizeLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    bool with_right_bound;
    ov::element::Type in_data_type, in_buckets_type, model_type;

    std::tie(shapes, with_right_bound, in_data_type, in_buckets_type, model_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto data = std::make_shared<ov::op::v0::Parameter>(in_data_type, inputDynamicShapes[0]);
    data->set_friendly_name("a_data");
    auto buckets = std::make_shared<ov::op::v0::Parameter>(in_buckets_type, inputDynamicShapes[1]);
    buckets->set_friendly_name("b_buckets");
    auto bucketize = std::make_shared<ov::op::v3::Bucketize>(data, buckets, model_type, with_right_bound);
    auto result = std::make_shared<ov::op::v0::Result>(bucketize);
    function = std::make_shared<ov::Model>(result, ngraph::ParameterVector{data, buckets}, "Bucketize");
}
} // namespace test
} // namespace ov
