// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "internal_properties.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/parameter.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

namespace {
enum class RemainderType { MOD, FLOOR_MOD };

// Model: Mod or FloorMod, second input as Parameter or Constant, enforce Snippets, follow with a fused i32 Add
using IntegerRemainderParams = std::tuple<RemainderType, utils::InputLayerType, bool, bool>;

// Every pair of these dividends and divisors is computed. Most of the values are not exactly representable in f32,
// and the 2^24 neighbours give quotients that f32 rounds to the wrong integer.
std::vector<int32_t> dividends() {
    constexpr auto i32_min = std::numeric_limits<int32_t>::min();
    constexpr auto i32_max = std::numeric_limits<int32_t>::max();
    return {i32_min,
            i32_min + 1,
            -(1 << 24) - 1,
            -(1 << 24) + 1,
            -7,
            0,
            5,
            (1 << 24) + 1,
            (1 << 25) + 1,
            1000000007,
            i32_max - 1,
            i32_max};
}

std::vector<int32_t> divisors(RemainderType type) {
    constexpr auto i32_min = std::numeric_limits<int32_t>::min();
    constexpr auto i32_max = std::numeric_limits<int32_t>::max();
    std::vector<int32_t>
        values{1, 2, -2, 3, -3, 7, 65536, -(1 << 24) + 1, 1 << 24, (1 << 24) + 1, 1000000000, i32_max, i32_min};
    // The reference Mod computes x % y, which is undefined for INT_MIN % -1, so -1 is only used for FloorMod
    if (type == RemainderType::FLOOR_MOD) {
        values.push_back(-1);
    }
    return values;
}

// Returns the dividend and divisor of every pair, flattened into two inputs of the same length
std::pair<std::vector<int32_t>, std::vector<int32_t>> remainder_inputs(RemainderType type) {
    std::vector<int32_t> lhs;
    std::vector<int32_t> rhs;
    for (const auto a : dividends()) {
        for (const auto b : divisors(type)) {
            lhs.push_back(a);
            rhs.push_back(b);
        }
    }
    return {lhs, rhs};
}
}  // namespace

class IntegerRemainderCPUTest : public testing::WithParamInterface<IntegerRemainderParams>,
                                virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<IntegerRemainderParams>& obj) {
        const auto& [type, secondInputType, enforceSnippets, fused] = obj.param;
        std::ostringstream result;
        result << (type == RemainderType::MOD ? "Mod" : "FloorMod");
        result << "_secondInput=" << secondInputType;
        result << "_enforceSnippets=" << enforceSnippets;
        result << "_fused=" << fused;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        // Integer results must match exactly; the default relative threshold allows large errors near the i32 limits
        abs_threshold = 0;
        rel_threshold = 0;
        const auto& [type, secondInputType, enforceSnippets, fused] = GetParam();
        std::tie(lhs, rhs) = remainder_inputs(type);
        const Shape shape{lhs.size()};

        configuration.insert(intel_cpu::snippets_mode(enforceSnippets ? intel_cpu::SnippetsMode::IGNORE_CALLBACK
                                                                      : intel_cpu::SnippetsMode::DISABLE));

        ParameterVector parameters{std::make_shared<op::v0::Parameter>(element::i32, shape)};
        std::shared_ptr<Node> divisor;
        if (secondInputType == utils::InputLayerType::PARAMETER) {
            init_input_shapes({InputShape{{shape}, {shape}}, InputShape{{shape}, {shape}}});
            parameters.push_back(std::make_shared<op::v0::Parameter>(element::i32, shape));
            divisor = parameters.back();
        } else {
            init_input_shapes({InputShape{{shape}, {shape}}});
            divisor = op::v0::Constant::create(element::i32, shape, rhs);
        }

        std::shared_ptr<Node> result;
        if (type == RemainderType::MOD) {
            result = std::make_shared<op::v1::Mod>(parameters[0], divisor);
        } else {
            result = std::make_shared<op::v1::FloorMod>(parameters[0], divisor);
        }
        if (fused) {
            result = std::make_shared<op::v1::Add>(result, op::v0::Constant::create(element::i32, {1}, {3}));
        }
        function = std::make_shared<Model>(OutputVector{result}, parameters, "IntegerRemainder");
    }

    void generate_inputs(const std::vector<Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        for (size_t i = 0; i < modelInputs.size(); ++i) {
            const auto& values = i == 0 ? lhs : rhs;
            Tensor tensor(element::i32, targetInputStaticShapes[i]);
            std::copy(values.begin(), values.end(), tensor.data<int32_t>());
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }
    }

private:
    std::vector<int32_t> lhs;
    std::vector<int32_t> rhs;
};

TEST_P(IntegerRemainderCPUTest, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_IntegerRemainder,
                         IntegerRemainderCPUTest,
                         ::testing::Combine(::testing::Values(RemainderType::MOD, RemainderType::FLOOR_MOD),
                                            ::testing::Values(utils::InputLayerType::PARAMETER,
                                                              utils::InputLayerType::CONSTANT),
                                            ::testing::Values(false, true),
                                            ::testing::Values(false, true)),
                         IntegerRemainderCPUTest::getTestCaseName);

}  // namespace ov::test
