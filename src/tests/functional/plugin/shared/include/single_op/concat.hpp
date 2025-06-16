// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using concatParamsTuple = typename std::tuple<
        int,                               // Concat axis
        std::vector<InputShape>,           // Input shapes
        ov::element::Type,                 // Model type
        std::string>;                      // Device name

// Multichannel
class ConcatLayerTest : public testing::WithParamInterface<concatParamsTuple>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<concatParamsTuple> &obj);

protected:
    void SetUp() override;
};

using ConcatStringParamsTuple = typename std::tuple<int,                                     // Concat axis
                                                    std::vector<ov::Shape>,                 // Input shapes
                                                    ov::element::Type,                       // Model type
                                                    std::string,                             // Device name
                                                    std::vector<std::vector<std::string>>>;  // String data

class ConcatStringLayerTest : public testing::WithParamInterface<ConcatStringParamsTuple>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConcatStringParamsTuple>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;

private:
    std::vector<std::vector<std::string>> string_data;
};
}  // namespace test
}  // namespace ov
