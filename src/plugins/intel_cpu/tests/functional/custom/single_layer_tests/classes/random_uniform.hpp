// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

typedef std::tuple<
        ov::Shape,                        // Output shapes
        std::tuple<double, double>,       // Min and Max values
        ov::test::ElementType,            // Shape precision
        ov::test::ElementType,            // Output precision
        uint64_t,                         // Global seed
        uint64_t,                         // Operational seed
        ov::op::PhiloxAlignment,          // Alignment of generator
        bool,                             // Is 1st input constant
        bool,                             // Is 2nd input constant
        bool,                             // Is 3rd input constant
        CPUTestUtils::CPUSpecificParams,  // CPU specific params
        ov::AnyMap                        // Additional plugin configuration
> RandomUniformLayerTestCPUParamSet;

class RandomUniformLayerTestCPU : public testing::WithParamInterface<RandomUniformLayerTestCPUParamSet>,
                                  public ov::test::SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUniformLayerTestCPUParamSet>& obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

    template<typename T>
    void rndUCompare(const ov::Tensor& expected, const ov::Tensor& actual);

private:
    ov::Shape m_output_shape;
    uint64_t m_global_seed;
    uint64_t m_operational_seed;
    double m_min_val;
    double m_max_val;
    static constexpr double m_mean_threshold = 0.075;
    static constexpr double m_variance_threshold = 0.15;
};

}  // namespace test
}  // namespace ov
