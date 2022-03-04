// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

enum ShapeMode {
    DYNAMIC,
    STATIC,
    BOTH
};

extern ShapeMode shapeMode;

using ReadIRParams = std::tuple<
        std::string,                         // IR path
        std::string,                         // Target Device
        ov::AnyMap>;                         // Plugin Config

class ReadIRTest : public testing::WithParamInterface<ReadIRParams>,
                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadIRParams> &obj);
    void query_model() override;

protected:
    void SetUp() override;

private:
    std::string pathToModel;
    std::string sourceModel;
    std::vector<std::pair<std::string, size_t>> ocuranceInModels;
};
} // namespace subgraph
} // namespace test
} // namespace ov
