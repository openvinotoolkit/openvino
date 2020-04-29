// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"

namespace LayerTestsDefinitions {

class NormalizeTransformation : public LayerTestsUtils::LayerTransformation {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj);

protected:
    void SetUp() override;

private:
    std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(const ngraph::Output<ngraph::Node>& output);
    void validate();
};

}  // namespace LayerTestsDefinitions
