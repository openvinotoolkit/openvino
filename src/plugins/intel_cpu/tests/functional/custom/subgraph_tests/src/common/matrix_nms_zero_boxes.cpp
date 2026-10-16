// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/matrix_nms.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// MatrixNms must not throw when none of the boxes pass the score/post threshold
// and the resulting output is legitimately empty (0 selected boxes).
class MatrixNmsZeroBoxes : public SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const ov::Shape boxesShape{1, 3, 4};
        const ov::Shape scoresShape{1, 2, 3};
        init_input_shapes({{{}, {boxesShape}}, {{}, {scoresShape}}});

        auto boxes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, boxesShape);
        auto scores = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, scoresShape);

        ov::op::v8::MatrixNms::Attributes attrs;
        // All generated scores are 0, so a positive score_threshold guarantees
        // every box is filtered out and the op must produce an empty (but valid) output.
        attrs.score_threshold = 0.5f;
        attrs.output_type = ov::element::i32;

        auto nms = std::make_shared<ov::op::v8::MatrixNms>(boxes, scores, attrs);

        ov::ResultVector results;
        for (const auto& output : nms->outputs()) {
            results.push_back(std::make_shared<ov::op::v0::Result>(output));
        }
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{boxes, scores}, "MatrixNmsZeroBoxes");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::Tensor tensor(funcInputs[i].get_element_type(), targetInputStaticShapes[i]);
            std::fill_n(tensor.data<float>(), tensor.get_size(), 0.0f);
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_F(MatrixNmsZeroBoxes, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
