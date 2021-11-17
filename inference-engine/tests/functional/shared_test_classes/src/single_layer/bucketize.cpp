// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/bucketize.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/ov_tensor_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace subgraph {

std::string BucketizeLayerTest::getTestCaseName(const testing::TestParamInfo<bucketizeParamsTuple>& obj) {
    InputShape dataShape;
    InputShape bucketsShape;
    bool with_right_bound;
    ElementType inDataPrc;
    ElementType inBucketsPrc;
    ElementType netPrc;
    TargetDevice targetDevice;

    std::tie(dataShape, bucketsShape, with_right_bound, inDataPrc, inBucketsPrc, netPrc, targetDevice) = obj.param;

    std::ostringstream result;
    result << "DS=" << CommonTestUtils::partialShape2str({dataShape.first}) << "_";
    for (const auto& item : dataShape.second) {
        result << CommonTestUtils::vec2str(item) << "_";
    }

    result << "BS=" << CommonTestUtils::partialShape2str({bucketsShape.first}) << "_";
    for (const auto& item : bucketsShape.second) {
        result << CommonTestUtils::vec2str(item) << "_";
    }

    if (with_right_bound)
        result << "rightIntervalEdge_";
    else
        result << "leftIntervalEdge_";
    result << "inDataPrc=" << inDataPrc << "_";
    result << "inBucketsPrc=" << inBucketsPrc << "_";
    result << "netPrc=" << netPrc << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void BucketizeLayerTest::generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();

    auto data_size = shape_size(targetInputStaticShapes[0]);
    ov::runtime::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                             targetInputStaticShapes[0],
                                                                             data_size * 5,
                                                                             0,
                                                                             10,
                                                                             7235346);

    ov::runtime::Tensor tensorBucket =
        ov::test::utils::create_and_fill_tensor_unique_sequence(funcInputs[1].get_element_type(),
                                                                targetInputStaticShapes[1],
                                                                0,
                                                                10,
                                                                8234231);

    inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    inputs.insert({funcInputs[1].get_node_shared_ptr(), tensorBucket});
}

void BucketizeLayerTest::SetUp() {
    InputShape dataShape;
    InputShape bucketsShape;
    bool with_right_bound;
    ElementType inDataPrc;
    ElementType inBucketsPrc;
    ElementType netPrc;

    std::tie(dataShape, bucketsShape, with_right_bound, inDataPrc, inBucketsPrc, netPrc, targetDevice) =
        this->GetParam();
    init_input_shapes({dataShape, bucketsShape});

    auto data = std::make_shared<ngraph::op::Parameter>(inDataPrc, inputDynamicShapes[0]);
    data->set_friendly_name("a_data");
    auto buckets = std::make_shared<ngraph::op::Parameter>(inBucketsPrc, inputDynamicShapes[1]);
    buckets->set_friendly_name("b_buckets");
    auto bucketize = std::make_shared<ngraph::op::v3::Bucketize>(data, buckets, netPrc, with_right_bound);
    function = std::make_shared<ngraph::Function>(std::make_shared<ngraph::opset1::Result>(bucketize),
                                                  ngraph::ParameterVector{data, buckets},
                                                  "Bucketize");
}
}  // namespace subgraph
}  // namespace test
}  // namespace ov
