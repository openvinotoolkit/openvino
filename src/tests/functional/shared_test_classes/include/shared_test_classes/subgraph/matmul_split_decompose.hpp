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
    ov::Shape bias_shape;
    ov::Shape reshape_shape;
};

typedef std::tuple<MatMulGatherDecomposeShapeParams,
                   std::string  // Device name
                   >
    MatMulGatherDecomposeParams;

class MatMulGatherDecompose : public testing::WithParamInterface<MatMulGatherDecomposeParams>,
                              virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulGatherDecomposeParams>& obj);

protected:
    void SetUp() override;
};

struct MatMulSplitDecomposeShapeParams {
    ov::Shape input_shape;    // [-1,-1,5120]
    ov::Shape weights_shape;  // [7680, 5120]
    bool trans_b;
    std::vector<int> split_lengths_data;  // split 7680 to 3 part[7680/3, 7680/3, 7680/3]
    ov::Shape reshape_shape1;  // [-1,-1,7680/3]
    ov::Shape reshape_shape2;  // [-1,-1,7680/3]
    ov::Shape reshape_shape3;  // [-1,-1,7680/3]
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
