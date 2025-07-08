// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<
        ov::Shape,
        ov::Shape,
        std::vector<ptrdiff_t>,
        std::vector<ptrdiff_t>,
        ov::Shape,
        size_t,
        ov::op::PadType,
        size_t,
        ov::test::utils::QuantizationGranularity> quantConvBackpropDataSpecificParams;
typedef std::tuple<
        quantConvBackpropDataSpecificParams,
        ov::element::Type,
        ov::Shape,
        std::string> quantConvBackpropDataLayerTestParamsSet;

class QuantConvBackpropDataLayerTest : public testing::WithParamInterface<quantConvBackpropDataLayerTestParamsSet>,
                                            virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<quantConvBackpropDataLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
