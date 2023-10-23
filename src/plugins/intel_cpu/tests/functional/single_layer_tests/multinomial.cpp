// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>
#include <string>
#include <vector>

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

using MultinomialCPUTestParams = typename std::tuple<InputShape,             // probs_shape
                                                     InputShape,             // num_samples_shape
                                                     ov::test::ElementType,  // convert_type
                                                     bool,                   // with_replacement
                                                     bool,                   // log_probs
                                                     uint64_t,               // global_seed
                                                     uint64_t,               // op_seed
                                                     >;

class MultinomialLayerCPUTest : public testing::WithParamInterface<MultinomialCPUTestParams>,
                                virtual public SubgraphBaseTest,
                                public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultinomialCPUTestParams>& obj) {
        InputShape probs_shape;
        InputShape num_samples_shape;
        ov::test::ElementType convert_type;
        bool with_replacement;
        bool log_probs;
        uint64_t global_seed;
        uint64_t op_seed;
        ov::test::ElementType network_precision;

        std::tie(probs_shape, num_samples_shape, convert_type, with_replacement, log_probs, global_seed, op_seed) =
            obj.param;

        std::ostringstream result;
        const char separator = '_';

        result << "probs_shape=" << ov::test::utils::partialShape2str({probs_shape.first}) << separator;
        result << "num_shape=" << ov::test::utils::partialShape2str({num_samples_shape.first}) << separator;
        result << "conv_type=" << convert_type << separator;
        result << "repl=" <<  ov::test::utils::bool2str(with_replacement) << separator;
        result << "log_p=" << ov::test::utils::bool2str(log_probs) << separator;
        result << "seed_g=" << global_seed << separator;
        result << "seed_o=" << op_seed << separator;
        return result.str();
    }

protected:
    void SetUp() override {
        MultinomialCPUTestParams test_params;

        InputShape probs_shape;
        InputShape num_samples_shape;
        ov::test::ElementType convert_type;
        bool with_replacement;
        bool log_probs;
        uint64_t global_seed;
        uint64_t op_seed;

        std::tie(probs_shape, num_samples_shape, convert_type, with_replacement, log_probs, global_seed, op_seed) =
            GetParam();

        selectedType = makeSelectedTypeStr("ref_any", ov::test::ElementType::f32);
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes({probs_shape, num_samples_shape});

        ov::ParameterVector params;
        auto probs_param = std::make_shared<ov::op::v0::Parameter>(ov::test::ElementType::f32, probs_shape);
        auto num_samples_param = std::make_shared<ov::op::v0::Parameter>(ov::test::ElementType::i32, probs_shape);
        auto multinomial = std::make_shared<ov::op::v13::Multinomial>(probs_param,
                                                                      num_samples_param,
                                                                      convert_type,
                                                                      with_replacement,
                                                                      log_probs,
                                                                      global_seed,
                                                                      op_seed);

        ov::ResultVector results{std::make_shared<ov::opset10::Result>(multinomial)};
        function = std::make_shared<ov::Model>(results, params, "MultinomialCPU");
    }
};

TEST_P(MultinomialLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Multinomial");
}

namespace {

const std::vector<ov::test::ElementType> convert_types = {
    ov::test::ElementType::i32,
    ov::test::ElementType::i64,
};

const std::vector<bool> with_replacements = {true, false};

const std::vector<bool> log_probs = {true, false};

const std::vector<InputShape> probs_static = {
    {{4, 4}, {{4, 4}}},
    {{2, 7}, {{2, 7}}},
};

const std::vector<InputShape> probs_dynamic = {
    {{-1, -1}, {{4, 4}}},
    {{-1, -1}, {{2, 7}}},
};

const std::vector<InputShape> num_samples_static = {
    {{1}, {{1}}},
    {{1}, {{1}}},
};

const std::vector<InputShape> num_samples_dynamic = {
    {{-1}, {{1}}},
    {{-1}, {{1}}},
};

const auto params_static = ::testing::Combine(::testing::ValuesIn(probs_static),
                                              ::testing::ValuesIn(num_samples_static),
                                              ::testing::ValuesIn(convert_types),
                                              ::testing::ValuesIn(with_replacements),
                                              ::testing::ValuesIn(log_probs),
                                              ::testing::Values(1),
                                              ::testing::Values(1));

const auto params_dynamic = ::testing::Combine(::testing::ValuesIn(probs_dynamic),
                                               ::testing::ValuesIn(num_samples_dynamic),
                                               ::testing::ValuesIn(convert_types),
                                               ::testing::ValuesIn(with_replacements),
                                               ::testing::ValuesIn(log_probs),
                                               ::testing::Values(1),
                                               ::testing::Values(1));

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialStatic,
                         MultinomialLayerCPUTest,
                         params_static,
                         MultinomialLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultinomialDynamic,
                         MultinomialLayerCPUTest,
                         params_dynamic,
                         MultinomialLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace CPULayerTestsDefinitions
