// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<int>,         // Indices
        ov::Shape,                // Indices shape
        int,                      // Gather axis
        std::vector<InputShape>,  // Input shapes
        ov::element::Type,        // Model type
        std::string               // Device name
> gatherParamsTuple;

class GatherLayerTest : public testing::WithParamInterface<gatherParamsTuple>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gatherParamsTuple> &obj);
protected:
    void SetUp() override;
};

typedef std::tuple<
    std::vector<InputShape>,  // Input shapes
    ov::Shape,                // Indices shape
    std::tuple<int, int>,     // Gather axis and batch
    ov::element::Type,        // Model type
    std::string               // Device name
> gather7ParamsTuple;

class Gather7LayerTest : public testing::WithParamInterface<gather7ParamsTuple>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj);

protected:
    void SetUp() override;
};

class Gather8LayerTest : public testing::WithParamInterface<gather7ParamsTuple>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj);

protected:
    void SetUp() override;
};

class Gather8IndiceScalarLayerTest : public testing::WithParamInterface<gather7ParamsTuple>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather7ParamsTuple>& obj);

protected:
    void SetUp() override;
};

typedef std::tuple<
    gather7ParamsTuple,
    std::vector<int64_t> // indices data
> gather8withIndicesDataParamsTuple;

class Gather8withIndicesDataLayerTest : public testing::WithParamInterface<gather8withIndicesDataParamsTuple>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gather8withIndicesDataParamsTuple>& obj);

protected:
    void SetUp() override;
};

typedef std::tuple<std::vector<InputShape>,  // Input shapes
                   ov::Shape,                // Indices shape
                   std::tuple<int, int>,     // Gather axis and batch
                   ov::element::Type,        // Model type
                   std::string,              // Device name
                   std::vector<int64_t>,     // indices data
                   std::vector<std::string>  // String data
                   >
    GatherStringParamsTuple;

class GatherStringWithIndicesDataLayerTest : public testing::WithParamInterface<GatherStringParamsTuple>,
                                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherStringParamsTuple>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;

private:
    std::vector<std::string> string_data;
};

}  // namespace test
}  // namespace ov
