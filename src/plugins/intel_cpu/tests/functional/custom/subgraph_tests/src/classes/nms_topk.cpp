// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nms_topk.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/topk.hpp"

namespace ov {
namespace test {

std::string NMSTopKTest::getTestCaseName(const testing::TestParamInfo<NMSTopKParams>& obj) {
    std::ostringstream results;
    ov::Shape inputShape;
    ov::element::Type inputPrecision;
    int64_t maxOutputBoxesPerClass;
    float iouThreshold;
    float scoreThreshold;
    std::string targetName;
    std::tie(inputShape, inputPrecision, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, targetName) = obj.param;

    results << "inputShape=" << ov::test::utils::vec2str(inputShape) << "_";
    results << "inputPrecision=" << inputPrecision << "_";
    results << "maxOutputBoxesPerClass=" << maxOutputBoxesPerClass << "_";
    results << "iouThreshold=" << iouThreshold << "_";
    results << "scoreThreshold=" << scoreThreshold << "_";
    results << "targetDevice=" << targetName;

    return results.str();
}

void NMSTopKTest::SetUp() {
    ov::Shape inputShape;
    ov::element::Type inputPrecision;
    int64_t maxOutputBoxesPerClass;
    float iouThreshold;
    float scoreThreshold;
    std::tie(inputShape, inputPrecision, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, targetDevice) =
        this->GetParam();

    OPENVINO_ASSERT(inputShape.size() >= 2,
                    "Unexpected NMS input shape dimension ",
                    inputShape.size(),
                    ", which should be at least 2.");
    const ov::Shape inputShape1 = inputShape;
    ov::Shape inputShape2 = inputShape;
    inputShape2[-1] = inputShape2[-2];
    inputShape2[-2] = 1;
    ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputShape1),
                                    std::make_shared<ov::op::v0::Parameter>(inputPrecision, inputShape2)};

    const auto nms = std::make_shared<ov::op::v9::NonMaxSuppression>(
        inputParams[0],
        inputParams[1],
        ov::op::v0::Constant::create(ov::element::i64, {}, {maxOutputBoxesPerClass}),
        ov::op::v0::Constant::create(inputPrecision, {}, {iouThreshold}),
        ov::op::v0::Constant::create(inputPrecision, {}, {scoreThreshold}),
        ov::op::v9::NonMaxSuppression::BoxEncodingType::CORNER,
        false);

    int k = static_cast<int>(maxOutputBoxesPerClass);
    const auto topK = std::make_shared<ov::op::v1::TopK>(nms,
                                                         ov::op::v0::Constant::create(ov::element::i32, {}, {k}),
                                                         0,
                                                         ov::op::TopKMode::MAX,
                                                         ov::op::TopKSortType::SORT_VALUES);

    ov::ResultVector results;
    for (size_t i = 0; i < topK->get_output_size(); i++) {
        results.push_back(std::make_shared<ov::op::v0::Result>(topK->output(i)));
    }

    function = std::make_shared<ov::Model>(results, inputParams, "nms_topk");
}

void NMSTopKTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    for (size_t i = 0; i < funcInputs.size(); ++i) {
        const auto& funcInput = funcInputs[i];
        ov::Tensor tensor;
        ov::test::utils::InputGenerateData in_data(0, 1, 100);
        tensor =
            ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);

        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }
}

TEST_P(NMSTopKTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
