
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ov_models/pass/convert_prc.hpp"

#include "snippets/codegen_bert.hpp"
//  todo: Rewrite this test using Snippets test infrastructure. See add_convert or conv_eltwise for example

namespace ov {
namespace test {
namespace snippets {

    std::string CodegenBert::getTestCaseName(testing::TestParamInfo<ov::test::snippets::CodegenBertParams> obj) {
        ov::element::Type_t netPrecision;
        ov::Shape inputShapes0, inputShapes1, newInputShapes;
        std::string targetDevice;
        std::tie(netPrecision, inputShapes0, inputShapes1, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS[0]=" << ov::test::utils::vec2str(inputShapes0) << "_";
        result << "IS[1]=" << ov::test::utils::vec2str(inputShapes1) << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

    // the simplest possible eltwise operation with streaming access to the data
    void CodegenBert::SetUp() {
        ov::Shape inputShape0, inputShape1;
        ov::element::Type_t netPrecision;
        std::tie(netPrecision, inputShape0, inputShape1, targetDevice) = this->GetParam();

        auto shape = ngraph::Shape{inputShape0};
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, shape);
        auto input2 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, shape);

        auto shapeMM = ngraph::Shape{inputShape1};
        auto input3 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, shapeMM);

        auto add    = std::make_shared<ngraph::opset1::Add>(input1, input2);
        auto mm     = std::make_shared<ngraph::opset1::MatMul>(add, input3);

        std::vector<float> vals(ngraph::shape_size(shape));
        for (int i = 0; i < vals.size(); i++) {
            vals[i] = static_cast<float>(i)*vals.size();
        }

        auto c0 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, shape);
        auto add2    = std::make_shared<ngraph::opset1::Subtract>(mm, c0);

        auto add3    = std::make_shared<ngraph::opset1::Multiply>(add, add2);
        auto result = std::make_shared<ngraph::opset1::Result>(add3);

        function = std::make_shared<ngraph::Function>(
            ngraph::ResultVector{result},
            // it should be some topological order to pass parameters for reference code to be executed correctly
            ngraph::ParameterVector{input1, input2, c0, input3},
            "CodegenBert");
    }

TEST_P(CodegenBert, CompareWithRefImpl) {
    Run();
};


} // namespace snippets
} // namespace test
} // namespace ov
