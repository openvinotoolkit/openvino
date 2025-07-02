// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
namespace ov {
namespace test {
enum class midOutputType {
    Sum,
    Sub,
    Mul,
};

typedef std::tuple<std::string,        // Target device name
                   ov::element::Type,  // Network precision
                   size_t,             // Input size
                   midOutputType,      // Type of layer that will be an output
                   ov::AnyMap          // Configuration
                   >
    outputBeforeActivationParams;

std::ostream& operator<<(std::ostream& os, const midOutputType& oType);

class OutputBeforeActivation : virtual public ov::test::SubgraphBaseStaticTest,
                               public testing::WithParamInterface<outputBeforeActivationParams> {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<outputBeforeActivationParams>& obj);
    // void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov
