// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

struct MatMulGatherDecomposeShapeParams {
    ov::Shape input_shape;
    ov::Shape weights_shape;
    bool trans_b;
    bool have_bias;
    ov::Shape bias_shape;
    ov::Shape reshape_shape;
};

typedef std::tuple<MatMulGatherDecomposeShapeParams,
                   std::string,  // Device name
                   bool          // Enable FakeQuantize
                   >
    MatMulGatherDecomposeParams;

class MatMulGatherDecompose : public testing::WithParamInterface<MatMulGatherDecomposeParams>,
                              virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulGatherDecomposeParams>& obj);

protected:
    void SetUp() override;
    void check_results();
};

}  // namespace test
}  // namespace ov
