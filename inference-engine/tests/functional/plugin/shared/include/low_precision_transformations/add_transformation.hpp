// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>
#include "shared_test_classes/base/low_precision_transformations/layer_transformation.hpp"
#include "lpt_ngraph_functions/add_function.hpp"
#include "lpt_ngraph_functions/common/constant.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class Expected {
public:
    std::string name;
    std::string type;
    std::string expectedKernelType;
};

inline std::ostream& operator<<(std::ostream& out, const Expected& expected) {
    return out << "_" << expected.name << "_" << expected.type << "_" << expected.expectedKernelType;
}

class AddTestValues{
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize1;
    ngraph::builder::subgraph::AddOperation operation1;
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize2;
    ngraph::builder::subgraph::AddOperation operation2;
    int constInput;
    bool broadcast;
    std::vector<ngraph::element::Type> precisionOnActivations;
    std::vector<ngraph::element::Type> expectedPrecisions;
    std::vector<Expected> expected;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    std::string,
    AddTestValues
> AddTransformationParams;

class AddTransformation :
    public testing::WithParamInterface<AddTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AddTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
