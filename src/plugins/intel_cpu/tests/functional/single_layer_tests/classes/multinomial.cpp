// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multinomial.hpp"
#include "ov_models/builders.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

class MultinomialLayerCPUTest : public testing::WithParamInterface<MultinomialTestCPUParams>,,
                                public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultinomialTestCPUParams>& obj) {
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
        MultinomialTestCPUParams test_params;

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

TEST_P(MultinomialLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Multinomial");
}
}  // namespace CPULayerTestsDefinitions
