// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace DiagonalInsertionTestNs {

using namespace ngraph;
using namespace ngraph::builder;
using namespace ngraph::element;
using namespace ngraph::op;
using namespace ngraph::opset9;
using namespace std;

using DiagonalInsertionTestParams = tuple<map<std::string, std::string>,   // Configuration
                                          vector<vector<float>>            // FakeQuantize min/max params
                                          >;

constexpr uint16_t fq_levels = numeric_limits<uint16_t>::max();

// This class performs tests on the following network:
//                              Params
//                     Const      |
//                       |    FakeQuantize
//                  FakeQuantize  |
//                       |      Reshape
//                        \     /
//                        MatMul
//                          |
//          Const       Reshape
//            |          /
//       FakeQuantize   /
//               \     /
//                 Add
//                  |
//             FakeQuantize
//                  |
//                ReLU
//                  |
//                Result
// The above network should cause the FuseFullyConnectedWithEltwisePass to be fired
// The final network should have only one functional layer - FullyConnected

class DiagonalInsertionTest : public testing::WithParamInterface<DiagonalInsertionTestParams>,
                              public LayerTestsUtils::LayerTestsCommon {
    const int32_t seed = 7235346;

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        return FuncTestUtils::createAndFillBlobFloatNormalDistribution(info.getTensorDesc(), 0.0f, 0.2f, seed);
    }

    ParameterVector CreateInputVector(const Type& type, const vector<std::size_t>& shapes) {
        return ov::ParameterVector{std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(shapes))};
    }

    shared_ptr<FakeQuantize> CreateFQNode(const Type& type,
                                          const shared_ptr<ov::Node>& node,
                                          float fq_min,
                                          float fq_max,
                                          std::size_t levels) {
        //
        auto fq_inp_min = makeConstant<float>(type, {1}, {fq_min});
        auto fq_inp_max = makeConstant<float>(type, {1}, {fq_max});
        auto fq_out_min = makeConstant<float>(type, {1}, {fq_min});
        auto fq_out_max = makeConstant<float>(type, {1}, {fq_max});
        return make_shared<FakeQuantize>(node, fq_inp_min, fq_inp_max, fq_out_min, fq_out_max, levels);
    }

    std::shared_ptr<Reshape> CreateReshapeNode(element::Type in_type,
                                               shared_ptr<Node> input_node,
                                               std::vector<size_t> target_shape_vect) {
        //
        const auto target_shape_const = Constant::create(in_type, Shape{target_shape_vect.size()}, target_shape_vect);
        return std::make_shared<Reshape>(input_node, target_shape_const, false);
    }

    bool IsDebugEnabled(map<std::string, std::string>& configuration) {
        return configuration.find("LOG_LEVEL") != configuration.end() && configuration["LOG_LEVEL"] == "LOG_DEBUG";
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<DiagonalInsertionTestParams> obj) {
        map<std::string, std::string> configuration;
        vector<vector<float>> fq_min_max;

        tie(configuration, fq_min_max) = obj.param;

        ostringstream result;
        for (auto const& config_item : configuration) {
            result << "_configItem=" << config_item.first << ":" << config_item.second;
        }
        for (auto const& fq : fq_min_max) {
            result << "_fqMin=" << fq[0] << "_fqMax=" << fq[1];
        }

        return result.str();
    }

protected:
    void SetUp() override {
        // Loosen threshold because of precision decrease during test
        threshold = 0.1;
        targetDevice = ov::test::utils::DEVICE_GNA;

        const size_t height = 512;
        const size_t width = 1024;
        const auto precision = ::ngraph::element::Type_t::f32;
        const vector<std::size_t> input_shape = {width};

        // Receive test params
        vector<vector<float>> fq_min_max;
        tie(configuration, fq_min_max) = this->GetParam();

        // Create network

        ov::ParameterVector input_vect{std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape(input_shape))};
        auto input_fq = CreateFQNode(precision, input_vect[0], fq_min_max[0][0], fq_min_max[0][1], fq_levels);

        auto reshape = CreateReshapeNode(ngraph::element::Type_t::i32, input_fq, {width, 1});

        auto mm_const = makeConstant<float>(precision, {height, width}, {}, true);
        auto mm_const_fq = CreateFQNode(precision, mm_const, fq_min_max[1][0], fq_min_max[1][1], fq_levels);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(mm_const_fq, reshape);
        auto matmul_fq = CreateFQNode(precision, matmul, fq_min_max[2][0], fq_min_max[2][1], fq_levels);
        auto add_mm_reshape = CreateReshapeNode(ngraph::element::Type_t::i32, matmul, {height});

        auto add_const = makeConstant<float>(precision, {height}, {}, true);
        auto add_const_fq = CreateFQNode(precision, add_const, fq_min_max[3][0], fq_min_max[3][1], fq_levels);

        auto add = make_shared<Add>(add_const_fq, add_mm_reshape);
        auto add_fq = CreateFQNode(precision, add, fq_min_max[4][0], fq_min_max[4][1], fq_levels);

        auto relu = make_shared<Relu>(add_fq);

        function = make_shared<ngraph::Function>(relu, input_vect, "DiagonalInsertion");
    }
};

TEST_P(DiagonalInsertionTest, CompareWithRefs) {
    Run();
};

const vector<map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_PRECISION", "I16"},
        {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"},
    },
};

vector<vector<float>> fq_mm1 = {{-19.38653564453125, 19.38653564453125},
                                {-4.872922897338867, 4.872922897338867},
                                {-633.115478515625, 633.115478515625},
                                {-3.2157254219055176, 3.2157254219055176},
                                {-633.0288696289062, 633.0288696289062}};

vector<vector<float>> fq_mm2 = {{-1.38653564453125, 1.38653564453125},
                                {-0.872922897338867, 0.872922897338867},
                                {-63.115478515625, 63.115478515625},
                                {-0.2157254219055176, 0.2157254219055176},
                                {-63.0288696289062, 63.0288696289062}};

vector<vector<float>> fq_mm3 = {{-0.1938653564453125, 0.1938653564453125},
                                {-0.04872922897338867, 0.04872922897338867},
                                {-6.33115478515625, 6.33115478515625},
                                {-0.032157254219055176, 0.032157254219055176},
                                {-6.330288696289062, 6.330288696289062}};

vector<vector<float>> fq_mm4 = {{-4.38653564453125, 4.38653564453125},
                                {-48.72922897338867, 48.72922897338867},
                                {-3.115478515625, 3.115478515625},
                                {-32.157254219055176, 32.157254219055176},
                                {-30.0288696289062, 30.0288696289062}};

vector<vector<float>> fq_mm5 = {{-390.38653564453125, 390.38653564453125},
                                {-400.872922897338867, 400.872922897338867},
                                {-633.115478515625, 633.115478515625},
                                {-399.2157254219055176, 399.2157254219055176},
                                {-633.0288696289062, 633.0288696289062}};

vector<vector<vector<float>>> fq_min_max = {fq_mm1, fq_mm2, fq_mm3, fq_mm4, fq_mm5};

INSTANTIATE_TEST_SUITE_P(smoke_DiagonalInsertion,
                         DiagonalInsertionTest,
                         ::testing::Combine(::testing::ValuesIn(configs), ::testing::ValuesIn(fq_min_max)),
                         DiagonalInsertionTest::getTestCaseName);

}  // namespace DiagonalInsertionTestNs
