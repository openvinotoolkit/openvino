// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/one_hot.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        ov::element::Type,          // depth type (any integer type)
        int64_t,                    // depth value
        ov::element::Type,          // On & Off values type (any supported type)
        float,                      // OnValue
        float,                      // OffValue
        int64_t,                    // axis
        ov::element::Type,          // Model type
        std::vector<InputShape>,    // Input shapes
        std::string                 // Target device name
> oneHot1LayerTestParamsSet;

class OneHot1LayerTest : public testing::WithParamInterface<oneHot1LayerTestParamsSet>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<oneHot1LayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

typedef std::tuple<ov::element::Type,                     // depth type (any integer type)
                   int64_t,                               // depth value
                   ov::element::Type,                     // On & Off values type (any supported type)
                   float,                                 // OnValue
                   float,                                 // OffValue
                   int64_t,                               // axis
                   ov::element::Type,                     // Model type
                   std::vector<InputShape>,               // Input shapes
                   op::v16::OneHot::NegativeIndicesMode,  // negative indices mode
                   std::string                            // Target device name
                   >
    oneHot16LayerTestParamsSet;

class OneHot16LayerTest : public testing::WithParamInterface<oneHot16LayerTestParamsSet>,
                          virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<oneHot16LayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
