// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <gtest/gtest.h>

#include "lpt_ngraph_functions/concat_function.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace ::testing;

class smoke_LPT_ConcatWithUnsupportedAxis : public Test {};

TEST_F(smoke_LPT_ConcatWithUnsupportedAxis, rtInfoCheck) {
    using namespace ngraph::builder::subgraph;

    const ngraph::element::Type precision = ngraph::element::f32;
    const ngraph::PartialShape inputPShape = PartialShape{ 1, 3, 16, 16 };
    const std::int64_t unsupportedAxis = 2;
    const auto fakeQuantize = FakeQuantizeOnData{ 256ul, ngraph::Shape({}), {0.f}, {2.55f}, {0.f}, {2.55f} };

    std::shared_ptr<ngraph::Function> function = ConcatFunction::getOriginalWithDifferentPrecisionOnChildren(
        precision,
        inputPShape,
        unsupportedAxis,
        fakeQuantize,
        fakeQuantize);

    SimpleLowPrecisionTransformer transformer;
    transformer.transform(function);

    const auto actualConcat = LayerTransformation::get<opset1::Concat>(function)[0];
    const auto& rtInfo = actualConcat->get_rt_info();
    ASSERT_TRUE(rtInfo.empty()) << "Unsupported concat mustn't contain LPT runtime attributes";
}
