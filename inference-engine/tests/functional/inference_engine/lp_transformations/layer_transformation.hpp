// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "transformations/low_precision/layer_transformation.hpp"

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    ngraph::pass::low_precision::LayerTransformation::Params> LayerTransformationParams;

class LayerTransformation : public CommonTestUtils::TestsCommon {
public:
    static ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsU8I8();
    static ngraph::pass::low_precision::LayerTransformation::Params LayerTransformation::createParamsI8I8();

    static std::string LayerTransformation::toString(const ngraph::pass::low_precision::LayerTransformation::Params& params);

    //static std::string getTestCaseName(testing::TestParamInfo<LayerTransformationParams> obj);
    static std::string getTestCaseNameByParams(
        const ngraph::element::Type& type,
        const ngraph::Shape& shape,
        const ngraph::pass::low_precision::LayerTransformation::Params& params);

protected:
    void transform(std::shared_ptr<ngraph::Function> function);

    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;
};
