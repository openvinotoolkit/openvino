// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using MatMulTransposeToReshapeParams = std::tuple<ov::element::Type,  // Network precision
                                                  std::string,        // Target device
                                                  ov::AnyMap>;        // Configuration

class MatMulTransposeToReshape : public testing::WithParamInterface<MatMulTransposeToReshapeParams>,
                                 virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulTransposeToReshapeParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
