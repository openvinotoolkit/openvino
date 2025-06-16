// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

enum class ConstantSubgraphType { SINGLE_COMPONENT, SEVERAL_COMPONENT };

std::ostream& operator<<(std::ostream& os, ConstantSubgraphType type);

typedef std::tuple<ConstantSubgraphType,
                   ov::Shape,          // input shape
                   ov::element::Type,  // input element type
                   std::string         // Device name
    > constResultParams;

class ConstantResultSubgraphTest : public testing::WithParamInterface<constResultParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<constResultParams>& obj);
    void createGraph(const ConstantSubgraphType& type,
                     const ov::Shape& input_shape,
                     const ov::element::Type& input_type);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
