// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "shared_test_classes/base/benchmark.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>

/*
The main purpose of the tests is to test cyclic inplace resolution in order to make sure that output edges are referenced whenever possible.
*/
// using namespace CPUTestUtils;
namespace ov {
namespace test {

using VectorShapes = std::vector<InputShape>;
using Parameters = std::tuple<std::pair<ov::Shape, ov::Shape>>;

class DuplicateLongConnections : public testing::WithParamInterface<Parameters>,
                                 virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<VectorShapes> obj) {
        (void) obj;
        return {};
    }

    void SetUp() override {
        const size_t B = std::stoi(std::getenv("CPU_TEST_B"));
        const size_t M = std::stoi(std::getenv("CPU_TEST_M"));
        const size_t K = std::stoi(std::getenv("CPU_TEST_K"));
        // const size_t K2 = std::stoi(std::getenv("CPU_TEST_K2"));
        const size_t N = std::stoi(std::getenv("CPU_TEST_N"));
        inType = ov::element::f32;
        static ov::AnyMap additionalConfig{ov::hint::inference_precision(ov::element::bf16)};
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::ParameterVector params;
        params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, PartialShape{-1, -1, static_cast<int64_t>(K)}));
        params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, PartialShape{-1, -1, static_cast<int64_t>(K)}));
        // std::pair<ov::PartialShape, std::vector<ov::Shape>>;
        const std::vector<InputShape> shapes{
            std::pair<ov::PartialShape, std::vector<ov::Shape>> {
                PartialShape{-1, -1, static_cast<int64_t>(K)},
                {ov::Shape{B, M, K}}
            },
            std::pair<ov::PartialShape, std::vector<ov::Shape>> {
                PartialShape{-1, -1, static_cast<int64_t>(K)},
                {ov::Shape{B, M, K}}
            }
        };
        init_input_shapes(shapes);

        const PartialShape add_const_shape{static_cast<int64_t>(B), static_cast<int64_t>(M), static_cast<int64_t>(K)};

        // auto add_tensor = ov::test::utils::create_and_fill_tensor(precision, add_const_shape.to_shape());
        // auto add_const_input = std::make_shared<ov::op::v0::Constant>(add_tensor);
        // auto add_parameter_input = std::make_shared<ov::op::v0::Parameter>(add_tensor);
        auto var = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{inputDynamicShapes[1], inType, "variable"});
        auto rv = std::make_shared<ov::op::v6::ReadValue>(params[1], var);
        auto add = std::make_shared<ov::op::v1::Add>(params[0], rv);
        auto assign = std::make_shared<ov::op::v6::Assign>(add, var);
        auto mamtmul_input = std::make_shared<ov::op::v0::Constant>(ov::test::utils::create_and_fill_tensor(precision, Shape{K, K}));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(add, mamtmul_input, false, true);
        auto swish = std::make_shared<ov::op::v4::Swish>(matmul);

        // auto var_k = std::make_shared<ov::op::util::Variable>(
        //     ov::op::util::VariableInfo{inputDynamicShapes[1], inType, "pastk"});
        // auto pastk = std::make_shared<ov::op::v6::ReadValue>(inputParams[3], var_k);
        // pastk->set_friendly_name("pastk_r");
        // auto sdp = std::make_shared<ov::op::v13::ScaledDotProductAttention>(swish, concatK, concatV, false);

        auto add_2 = std::make_shared<ov::op::v1::Add>(swish, add);
        auto mamtmul_2_input = std::make_shared<ov::op::v0::Constant>(ov::test::utils::create_and_fill_tensor(precision, Shape{N, K}));
        auto matmul_2 = std::make_shared<ov::op::v0::MatMul>(add_2, mamtmul_2_input, false, true);
        auto result_0 = std::make_shared<ov::op::v0::Result>(matmul_2);

        SinkVector sinks{assign};
        function = std::make_shared<ov::Model>(ov::ResultVector{result_0}, sinks, params, "Subgraph0");
        function->add_variables({var});
        rel_threshold = 1e-3f;
    }

protected:
    const ov::element::Type precision = ov::element::f32;
};

TEST_F(DuplicateLongConnections, smoke_CompareWithRefs) {
    run();
}

struct BenchmarkMatMulLayerCPUTest : BenchmarkLayerTest<DuplicateLongConnections> {};

TEST_F(BenchmarkMatMulLayerCPUTest, smoke_BenchmarkDuplicateLongConnections) {
    auto getNumIter = [](){
        static auto result = std::getenv("CPU_TEST_NUM_ITER") ? std::stoi(std::getenv("CPU_TEST_NUM_ITER")) : 50;
        return result;
    };

    auto getWarmUpTime = [](){
        static auto result = std::getenv("CPU_TEST_WARM_UP") ? std::stoi(std::getenv("CPU_TEST_WARM_UP")) : 2000;
        return result;
    };

    run_benchmark("FullyConnected", std::chrono::milliseconds(getWarmUpTime()), getNumIter());
}

}  // namespace test
}  // namespace ov
