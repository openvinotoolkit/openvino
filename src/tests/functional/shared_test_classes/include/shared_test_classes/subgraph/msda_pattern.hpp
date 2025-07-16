// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct MSDAPatternShapeParams {
    ov::Shape value_shape;
    ov::Shape offset_shape;
    ov::Shape weight_shape;
};

typedef std::tuple<MSDAPatternShapeParams>
    MSDAPatternParams;

class MSDAPattern : public testing::WithParamInterface<MSDAPatternShapeParams>,
                    public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MSDAPatternShapeParams> obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
