// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct MatMulSplitDecomposeShapeParams {
    ov::Shape input_shape;
    ov::Shape weights_shape;
    bool trans_b;
    ov::Shape bias_shape;
    ov::Shape reshape_shape;
};

typedef std::tuple<MatMulSplitDecomposeShapeParams,
                   std::string  // Device name
                   >
    MatMulSplitDecomposeParams;

class MatMulSplitDecompose : public testing::WithParamInterface<MatMulSplitDecomposeParams>,
                             virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulSplitDecomposeParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
