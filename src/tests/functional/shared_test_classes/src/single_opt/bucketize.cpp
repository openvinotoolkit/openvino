// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_opt/bucketize.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {

std::string BucketizeLayerTest::getTestCaseName(const testing::TestParamInfo<bucketizeParamsTuple>& obj) {
    std::vector<InputShape> shapes;
    bool with_right_bound;
    ov::element::Type inDataPrc, inBucketsPrc, netPrc;
    std::string targetDevice;

    std::tie(shapes, with_right_bound, inDataPrc, inBucketsPrc, netPrc, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (const auto& shape : shapes) {
        result << ov::test::utils::partialShape2str({shape.first}) << "_";
    }
    result << ")_TS=(";
    for (const auto& shape : shapes) {
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
    }
    if (with_right_bound)
        result << "rightIntervalEdge_";
    else
        result << "leftIntervalEdge_";
    result << "inDataPrc=" << inDataPrc.get_type_name() << "_";
    result << "inBucketsPrc=" << inBucketsPrc.get_type_name() << "_";
    result << "netPrc=" << netPrc.get_type_name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void BucketizeLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    auto itTargetShape = targetInputStaticShapes.begin();
    for (auto&& param : function->get_parameters()) {
        auto name = param->get_friendly_name();
        ov::Tensor tensor;
        if (name == "a_data") {
            auto shape_size = ov::shape_size(*itTargetShape);
            tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), *itTargetShape++, shape_size * 5, 0, 10, 7235346);
        } else if (name == "b_buckets") {
            tensor = ov::test::utils::create_and_fill_tensor_unique_sequence(param->get_element_type(), *itTargetShape++, 0, 10, 8234231);
        }
        inputs.insert({param, tensor});
    }
}

void BucketizeLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    bool with_right_bound;
    ov::element::Type in_data_type, in_buckets_type, net_type;

    std::tie(shapes, with_right_bound, in_data_type, in_buckets_type, net_type, targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    auto data = std::make_shared<ov::op::v0::Parameter>(in_data_type, inputDynamicShapes[0]);
    data->set_friendly_name("a_data");
    auto buckets = std::make_shared<ov::op::v0::Parameter>(in_buckets_type, inputDynamicShapes[1]);
    buckets->set_friendly_name("b_buckets");
    auto bucketize = std::make_shared<ov::op::v3::Bucketize>(data, buckets, net_type, with_right_bound);
    auto result = std::make_shared<ov::op::v0::Result>(bucketize);
    function = std::make_shared<ov::Model>(result, ngraph::ParameterVector{data, buckets}, "Bucketize");
}
} // namespace test
} // namespace ov
