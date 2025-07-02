// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

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
        size_t,
        ov::op::PadType,
        size_t,
        ov::test::utils::QuantizationGranularity> quantGroupConvBackpropDataSpecificParams;
typedef std::tuple<
        quantGroupConvBackpropDataSpecificParams,
        ov::element::Type,
        ov::Shape,
        std::string> quantGroupConvBackpropDataLayerTestParamsSet;

class QuantGroupConvBackpropDataLayerTest : public testing::WithParamInterface<quantGroupConvBackpropDataLayerTestParamsSet>,
                                            virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<quantGroupConvBackpropDataLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
