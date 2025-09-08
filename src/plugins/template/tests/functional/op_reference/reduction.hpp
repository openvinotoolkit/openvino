// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"

using namespace ov;

namespace reference_tests {
namespace ReductionOpsRefTestDefinitions {

using ov::test::utils::ReductionType;

// Removes some values from a vector of axis values
template <typename AXIS_VALUES>
AXIS_VALUES reduce(const AXIS_VALUES& axis_values, const ov::AxisSet& deleted_axes, bool keep_dims) {
    AXIS_VALUES result;

    for (size_t i = 0; i < axis_values.size(); i++) {
        if (deleted_axes.find(i) == deleted_axes.end()) {
            result.push_back(axis_values[i]);
        } else {
            if (keep_dims)
                result.push_back(1);
        }
    }

    return result;
}

struct ReductionParams {
    ReductionParams(const ReductionType& reductType,
                    const bool keepDims,
                    const std::vector<int64_t>& axes,
                    const reference_tests::Tensor& dataTensor,
                    const reference_tests::Tensor& outputTensor)
        : reductionType(reductType),
          keepDimensions(keepDims),
          reductionAxes(axes),
          data(dataTensor),
          output(outputTensor) {}

    ReductionType reductionType;
    bool keepDimensions;
    std::vector<int64_t> reductionAxes;
    reference_tests::Tensor data;
    reference_tests::Tensor output;
};

class ReferenceReductionLayerTest : public testing::TestWithParam<ReductionParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.data.data};
        refOutData = {params.output.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReductionParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "reductionType=" << param.reductionType << "_";
        result << "dataType=" << param.data.type << "_";
        result << "dataShape=" << param.data.shape << "_";
        result << "axes=" << ov::test::utils::vec2str(param.reductionAxes);
        if (param.keepDimensions) {
            result << "_keepDims";
        }
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const ReductionParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.data.type, params.data.shape);
        const auto axes = std::make_shared<op::v0::Constant>(ov::element::i64,
                                                             ov::Shape{params.reductionAxes.size()},
                                                             params.reductionAxes);
        std::shared_ptr<ov::Node> reduction;
        switch (params.reductionType) {
        case ReductionType::Mean: {
            reduction = std::make_shared<ov::op::v1::ReduceMean>(data, axes, params.keepDimensions);
            break;
        }
        case ReductionType::Max: {
            reduction = std::make_shared<ov::op::v1::ReduceMax>(data, axes, params.keepDimensions);
            break;
        }
        case ReductionType::Min: {
            reduction = std::make_shared<ov::op::v1::ReduceMin>(data, axes, params.keepDimensions);
            break;
        }
        case ReductionType::Prod: {
            reduction = std::make_shared<ov::op::v1::ReduceProd>(data, axes, params.keepDimensions);
            break;
        }
        case ReductionType::Sum: {
            reduction = std::make_shared<ov::op::v1::ReduceSum>(data, axes, params.keepDimensions);
            break;
        }
        case ReductionType::LogicalOr: {
            reduction = std::make_shared<ov::op::v1::ReduceLogicalOr>(data, axes, params.keepDimensions);
            break;
        }
        case ReductionType::LogicalAnd: {
            reduction = std::make_shared<ov::op::v1::ReduceLogicalAnd>(data, axes, params.keepDimensions);
            break;
        }
        case ReductionType::L1: {
            reduction = std::make_shared<ov::op::v4::ReduceL1>(data, axes, params.keepDimensions);
            break;
        }
        case ReductionType::L2: {
            reduction = std::make_shared<ov::op::v4::ReduceL2>(data, axes, params.keepDimensions);
            break;
        }
        default:
            throw std::runtime_error("Can't create layer for this reduction type");
        }
        return std::make_shared<ov::Model>(reduction, ov::ParameterVector{data});
    }
};
}  // namespace ReductionOpsRefTestDefinitions
}  // namespace reference_tests
