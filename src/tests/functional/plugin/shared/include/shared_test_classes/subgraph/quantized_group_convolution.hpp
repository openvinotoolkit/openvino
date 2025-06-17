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
        size_t,
        ov::test::utils::QuantizationGranularity,
        bool> quantGroupConvSpecificParams;
typedef std::tuple<
        quantGroupConvSpecificParams,
        ov::element::Type,
        ov::Shape,
        std::string> quantGroupConvLayerTestParamsSet;

class QuantGroupConvLayerTest : public testing::WithParamInterface<quantGroupConvLayerTestParamsSet>,
                                            virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<quantGroupConvLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
