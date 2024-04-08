// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using FuseTransposeAndReorderParams = std::tuple<ov::Shape,         // Input shape
                                                 ov::element::Type  // Input precision
                                                 >;

class FuseTransposeAndReorderTest : public testing::WithParamInterface<FuseTransposeAndReorderParams>, public CPUTestsBase,
        virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseTransposeAndReorderParams> obj);

protected:
    void SetUp() override;
    virtual void create_model();
    void check_transpose_count(size_t expectedTransposeCount);

    ov::Shape input_shape;
    ov::element::Type in_prec;
};

class FuseTransposeAndReorderTest1 : public FuseTransposeAndReorderTest {
protected:
    void create_model() override;
};

class FuseTransposeAndReorderTest2 : public FuseTransposeAndReorderTest {
protected:
    void create_model() override;
};

class FuseTransposeAndReorderTest3 : public FuseTransposeAndReorderTest {
protected:
    void create_model() override;
};

class FuseTransposeAndReorderTest4 : public FuseTransposeAndReorderTest {
protected:
    void create_model() override;
};

class FuseTransposeAndReorderTest5 : public FuseTransposeAndReorderTest {
protected:
    void create_model() override;
};

} // namespace test
} // namespace ov
