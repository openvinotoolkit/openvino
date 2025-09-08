// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::pair<float, float> QuantRange;

typedef std::tuple<
        uint64_t,
        QuantRange,
        QuantRange,
        ov::test::utils::QuantizationGranularity,
        ov::element::Type> QuantParams;

typedef std::tuple<
        QuantParams,
        QuantParams,
        ov::element::Type,
        ov::Shape,
        ov::Shape,
        std::string> QuantMatMulLayerTestParamsSet;

class QuantMatMulTest : public testing::WithParamInterface<QuantMatMulLayerTestParamsSet>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantMatMulLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
