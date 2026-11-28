// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Integer ReduceProd must be computed in i64 (PyTorch promotes integer prod to int64), so products that do not fit
// into i32 stay exact on CPU (openvinotoolkit/openvino#38719):
//
//   Parameter(i32) -> Convert(i64) -> ReduceProd -> Result

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using ReduceProdI64Params = std::tuple<ov::Shape,             // input shape
                                       std::vector<int64_t>,  // axes
                                       bool>;                 // keep dims

class ReduceProdI64CPUTest : public testing::WithParamInterface<ReduceProdI64Params>, public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceProdI64Params>& obj) {
        const auto& [shape, axes, keep_dims] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(shape) << "_axes=" << ov::test::utils::vec2str(axes)
               << "_keepDims=" << keep_dims;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        const auto& [shape, axes, keep_dims] = GetParam();

        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape);
        auto convert = std::make_shared<ov::op::v0::Convert>(param, ov::element::i64);
        auto axes_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto prod = std::make_shared<ov::op::v1::ReduceProd>(convert, axes_const, keep_dims);
        function = std::make_shared<ov::Model>(ov::OutputVector{prod}, ov::ParameterVector{param}, "ReduceProdI64");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInput = function->inputs()[0];
        // |x| in [-2000, 2000): a product of 3+ elements usually overflows i32 but fits i64
        auto tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                              targetInputStaticShapes[0],
                                                              ov::test::utils::InputGenerateData(-2000, 4000));
        inputs.insert({funcInput.get_node_shared_ptr(), tensor});
    }

    void check_reduce_runs_in_i64() {
        for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
            const auto& rt_info = node->get_rt_info();
            if (rt_info.at(ov::exec_model_info::LAYER_TYPE).as<std::string>() == "Reduce") {
                ASSERT_EQ(rt_info.at(ov::exec_model_info::RUNTIME_PRECISION).as<std::string>(), "i64");
            }
        }
    }
};

TEST_P(ReduceProdI64CPUTest, CompareWithRefs) {
    run();
    check_reduce_runs_in_i64();
}

INSTANTIATE_TEST_SUITE_P(smoke_ReduceProdI64,
                         ReduceProdI64CPUTest,
                         ::testing::Values(ReduceProdI64Params{{8}, {0}, false},
                                           ReduceProdI64Params{{2, 5}, {1}, false},
                                           ReduceProdI64Params{{3, 4, 5}, {0, 2}, true},
                                           ReduceProdI64Params{{2, 3, 4, 5}, {1}, false},
                                           ReduceProdI64Params{{2, 0, 4}, {1}, false}),  // empty reduction -> 1
                         ReduceProdI64CPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
