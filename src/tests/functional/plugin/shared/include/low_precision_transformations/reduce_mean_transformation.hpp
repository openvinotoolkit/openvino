// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class ReduceMeanOperation {
public:
    std::vector<int64_t> constantValues;
    bool keepDims;

    ReduceMeanOperation();
    ReduceMeanOperation(const std::vector<int64_t>& constantValues, const bool keepDims);
};

class ReduceMeanTransformationParam {
public:
    ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    ov::builder::subgraph::DequantizationOperations::Convert convert;
    ov::builder::subgraph::DequantizationOperations dequantizationBefore;
    ReduceMeanOperation reduceMean;
    ov::builder::subgraph::DequantizationOperations dequantizationAfter;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ov::element::Type,
    ov::PartialShape,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    ReduceMeanTransformationParam
> ReduceMeanTransformationParams;

class ReduceMeanTransformation :
    public testing::WithParamInterface<ReduceMeanTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceMeanTransformationParams>& obj);

protected:
    void SetUp() override;
    void run() override;
};
}  // namespace LayerTestsDefinitions
