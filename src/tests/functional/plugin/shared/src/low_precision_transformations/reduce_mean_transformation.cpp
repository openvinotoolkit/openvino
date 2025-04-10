// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reduce_mean_transformation.hpp"
#include <sstream>
#include <string>
#include <vector>

#include "ov_lpt_models/reduce.hpp"

namespace LayerTestsDefinitions {

ReduceMeanOperation::ReduceMeanOperation() : constantValues(), keepDims() { }

ReduceMeanOperation::ReduceMeanOperation(const std::vector<int64_t>& constantValues, const bool keepDims) {
    this->constantValues = constantValues;
    this->keepDims = keepDims;
}

std::string ReduceMeanTransformation::getTestCaseName(const testing::TestParamInfo<ReduceMeanTransformationParams>& obj) {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    std::string targetDevice;
    ov::pass::low_precision::LayerTransformation::Params params;
    ReduceMeanTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = obj.param;

    std::ostringstream result;
    result << get_test_case_name_by_params(netPrecision, inputShape, targetDevice, params) << "_" <<
           param.fakeQuantize <<
        param.convert <<
        param.dequantizationBefore <<
        (param.reduceMean.keepDims ? "_keepDims_" : "");

    result << "_reduce_axis_";
    for (const auto& elem : param.reduceMean.constantValues) {
        result << elem << "_";
    }

    return result.str();
}

void ReduceMeanTransformation::SetUp() {
    ov::element::Type netPrecision;
    ov::PartialShape inputShape;
    ov::pass::low_precision::LayerTransformation::Params params;
    ReduceMeanTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, param) = GetParam();

    init_input_shapes(inputShape);

    function = ov::builder::subgraph::ReduceFunction::get<ov::op::v1::ReduceMean>(
        netPrecision,
        inputShape,
        param.fakeQuantize,
        param.convert,
        param.dequantizationBefore,
        param.reduceMean.constantValues,
        param.reduceMean.keepDims,
        param.dequantizationAfter);
}

void ReduceMeanTransformation::run() {
    LayerTransformation::run();

    const auto params = std::get<4>(GetParam());
    const auto actualType = get_runtime_precision(params.layerName);
    EXPECT_EQ(actualType, params.expectedKernelType);
}

TEST_P(ReduceMeanTransformation, CompareWithRefImpl) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
};

} // namespace LayerTestsDefinitions
