// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"

namespace LayerTestsDefinitions {

class GroupConvolutionTransformationParam {
public:
    GroupConvolutionTransformationParam() = default;
    GroupConvolutionTransformationParam(const size_t group,
                                        const int groupCalculationDimention,
                                        const ngraph::builder::subgraph::FakeQuantizeOnData& fakeQuantizeOnData,
                                        const ngraph::builder::subgraph::FakeQuantizeOnWeights& fakeQuantizeOnWeights,
                                        const bool addReshape = true,
                                        const std::string& layerName = "",
                                        const std::string& expectedKernelType = "")
        : group(group),
          groupCalculationDimention(groupCalculationDimention),
          fakeQuantizeOnData(fakeQuantizeOnData),
          fakeQuantizeOnWeights(fakeQuantizeOnWeights),
          addReshape(addReshape),
          layerName(layerName),
          expectedKernelType(expectedKernelType) {}

    size_t group;
    int groupCalculationDimention;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantizeOnData;
    ngraph::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
    bool addReshape;
    std::string layerName;
    std::string expectedKernelType;
};

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    ov::pass::low_precision::LayerTransformation::Params,
    std::pair<ngraph::PartialShape, ngraph::Shape>,
    GroupConvolutionTransformationParam,
    bool // add precision preserved operation
> GroupConvolutionTransformationParams;

class GroupConvolutionTransformation :
    public testing::WithParamInterface<GroupConvolutionTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupConvolutionTransformationParams>& obj);

protected:
    void SetUp() override;

    void Run() override;
};

}  // namespace LayerTestsDefinitions
