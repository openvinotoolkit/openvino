// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace op_conformance {

enum ShapeMode {
    DYNAMIC,
    STATIC,
    BOTH
};

extern ShapeMode shapeMode;

using ReadIRParams = std::pair<std::string, std::string>; // { ir_path, ref_tensor_path}

class ReadIRTest : public testing::WithParamInterface<ReadIRParams>,
                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj);
    std::vector<ov::Tensor> calculate_refs() override;

protected:
    void SetUp() override;

private:
    std::string path_to_model, path_to_ref_tensor;
    std::vector<std::pair<std::string, size_t>> ocurance_in_models;
};
} // namespace op_conformance
} // namespace test
} // namespace ov
