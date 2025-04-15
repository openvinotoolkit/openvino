// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/nms_static_shape_ie.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace testing {
class ConvertMatrixNmsToMatrixNmsIEFixture : public ::testing::WithParamInterface<element::Type>,
                                             public TransformationTestsF {
public:
    static std::string getTestCaseName(testing::TestParamInfo<element::Type> obj) {
        std::ostringstream result;
        result << "ConvertMatrixNmsToMatrixNmsIE_" << obj.param.get_type_name();
        return result.str();
    }
    void Execute() {
        element::Type element_type = this->GetParam();
        {
            auto boxes = std::make_shared<opset1::Parameter>(element_type, Shape{1, 1000, 4});
            auto scores = std::make_shared<opset1::Parameter>(element_type, Shape{1, 1, 1000});

            auto nms = std::make_shared<opset8::MatrixNms>(boxes, scores, opset8::MatrixNms::Attributes());

            model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

            manager.register_pass<ov::pass::ConvertMatrixNmsToMatrixNmsIE>();
            manager.register_pass<pass::ConstantFolding>();
        }

        {
            auto boxes = std::make_shared<opset1::Parameter>(element_type, Shape{1, 1000, 4});
            auto scores = std::make_shared<opset1::Parameter>(element_type, Shape{1, 1, 1000});
            auto nms = std::make_shared<ov::op::internal::NmsStaticShapeIE<opset8::MatrixNms>>(
                boxes,
                scores,
                opset8::MatrixNms::Attributes());

            model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
        }
        ASSERT_EQ(model->get_output_element_type(0), model_ref->get_output_element_type(0))
            << "Output element type mismatch " << model->get_output_element_type(0).get_type_name() << " vs "
            << model_ref->get_output_element_type(0).get_type_name();
    }
};

TEST_P(ConvertMatrixNmsToMatrixNmsIEFixture, CompareFunctions) {
    Execute();
}

INSTANTIATE_TEST_SUITE_P(ConvertMatrixNmsToMatrixNmsIE,
                         ConvertMatrixNmsToMatrixNmsIEFixture,
                         ::testing::ValuesIn(std::vector<element::Type>{element::f32, element::f16}),
                         ConvertMatrixNmsToMatrixNmsIEFixture::getTestCaseName);
}  // namespace testing
