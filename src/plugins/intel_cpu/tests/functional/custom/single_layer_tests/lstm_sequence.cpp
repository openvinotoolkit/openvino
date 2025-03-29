// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>

#include "common_test_utils/node_builders/lstm_cell.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/pass/manager.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using LSTMSequenceCpuSpecificParams =
    typename std::tuple<std::vector<InputShape>,             // Shapes
                        ov::test::utils::SequenceTestsMode,  // Pure Sequence or TensorIterator
                        std::vector<std::string>,            // Activations
                        float,                               // Clip
                        ov::op::RecurrentSequenceDirection,  // Direction
                        ElementType,                         // Network precision
                        CPUSpecificParams,                   // CPU specific params
                        ov::AnyMap                           // Additional config
                        >;

class LSTMSequenceCPUTest : public testing::WithParamInterface<LSTMSequenceCpuSpecificParams>,
                            virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSequenceCpuSpecificParams> &obj) {
        std::vector<InputShape> inputShapes;
        ov::test::utils::SequenceTestsMode seqMode;
        std::vector<std::string> activations;
        float clip;
        ov::op::RecurrentSequenceDirection direction;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;

        std::tie(inputShapes, seqMode, activations, clip, direction, netPrecision, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=(";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=";
        for (size_t i = 0lu; i < inputShapes.front().second.size(); i++) {
            result << "{";
            for (size_t j = 0lu; j < inputShapes.size(); j++) {
                result << ov::test::utils::vec2str(inputShapes[j].second[i]) << (j < inputShapes.size() - 1 ? "_" : "");
            }
            result << "}_";
        }
        result << "seqMode=" << seqMode << "_";
        result << "activations=" << ov::test::utils::vec2str(activations)  << "_";
        result << "clip=" << clip << "_";
        result << "direction=" << direction << "_";
        result << "netPrec=" << netPrecision << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<InputShape> inputShapes;
        ov::test::utils::SequenceTestsMode seqMode;
        std::vector<std::string> activations;
        float clip;
        ov::op::RecurrentSequenceDirection direction;
        ElementType netPrecision;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;

        std::tie(inputShapes, seqMode, activations, clip, direction, netPrecision, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes(inputShapes);
        if (inputDynamicShapes.size() == 3 && inputDynamicShapes[0][0].is_dynamic() &&
                inputDynamicShapes[1][0].is_dynamic() && inputDynamicShapes[2][0].is_dynamic())
            throw std::runtime_error("Invalid test case. If 4th input is constant, batch dimension must be static.");

        const size_t inputSize = targetStaticShapes.front()[0][2];
        const size_t hiddenSize = targetStaticShapes.front()[1][2];
        const size_t numDirections = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;

        float WRB_range = 0;
        auto it_dynamic_batch = additionalConfig.find("_dynamic_batch_test");
        if (it_dynamic_batch != additionalConfig.end() && it_dynamic_batch->second == "yes") {
            additionalConfig.erase(it_dynamic_batch);
            // special config for _dynamic_batch_test
            abs_threshold = 0.001f;
            WRB_range = 1.0f;
        }

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        auto it = additionalConfig.find(ov::hint::inference_precision.name());
        if (it != additionalConfig.end() && it->second.as<ov::element::Type>() == ov::element::bf16) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            selectedType = makeSelectedTypeStr(selectedType, netPrecision);
        }

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
        }
        const size_t batchSize = inputDynamicShapes[0][0].is_static() ? inputDynamicShapes[0][0].get_length() :
            inputDynamicShapes[1][0].is_static() ? inputDynamicShapes[1][0].get_length() :
            inputDynamicShapes[2][0].is_static() ? inputDynamicShapes[2][0].get_length() :
            inputDynamicShapes.size() > 3 && inputDynamicShapes[3][0].is_static() ? inputDynamicShapes[3][0].get_length() :
            1lu;
        if (inputDynamicShapes.size() > 3) {
            if (!inputDynamicShapes[3].is_dynamic() &&
                    seqMode != ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM &&
                    seqMode != ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM) {
                params.pop_back();
            } else {
                params[3]->set_element_type(ElementType::i64);
            }
        }

        ov::OutputVector paramsOuts;
        for (const auto& param : params)
          paramsOuts.push_back(param);

        std::vector<ov::Shape> WRB = {{numDirections, 4 * hiddenSize, inputSize}, {numDirections, 4 * hiddenSize, hiddenSize},
                {numDirections, 4 * hiddenSize}, {batchSize}};
        auto lstmSequenceOp = utils::make_lstm(paramsOuts,
                                               WRB,
                                               hiddenSize,
                                               activations,
                                               {},
                                               {},
                                               clip,
                                               true,
                                               direction,
                                               seqMode,
                                               WRB_range);

        function = makeNgraphFunction(netPrecision, params, lstmSequenceOp, "lstmSequenceOp");

        if (seqMode != ov::test::utils::SequenceTestsMode::PURE_SEQ) {
            ov::pass::Manager manager;
            if (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
            manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
            manager.run_passes(function);
            bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, true);
        } else {
            bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, false);
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);

        const size_t batchSize = targetInputStaticShapes[0][0];
        const int64_t maxSeqLen = targetInputStaticShapes[0][1];
        const auto& funcInputs = function->inputs();
        if (funcInputs.size() > 3) {
            const auto& seqLenInput = inputs.find(funcInputs[3].get_node_shared_ptr());
            if (seqLenInput == inputs.end())
                throw std::runtime_error("Could not find Sequence length input.");

            auto lenData = seqLenInput->second.data<ov::element_type_traits<ElementType::i64>::value_type>();
            std::fill(lenData, lenData + batchSize, maxSeqLen);
        }
    }
};

TEST_P(LSTMSequenceCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RNNSeq");
}

namespace {
/* CPU PARAMS */
std::vector<ov::AnyMap> additionalConfig = {{{ov::hint::inference_precision(ov::element::f32)}},
                                            {{ov::hint::inference_precision(ov::element::bf16)}}};

CPUSpecificParams cpuParams{{ntc, tnc, tnc}, {ntc, tnc, tnc}, {"ref_any"}, "ref_any"};
// CPUSpecificParams cpuParamsBatchSizeOne{{tnc, ntc, ntc}, {tnc, ntc, ntc}, {"ref_any"}, "ref_any"};
CPUSpecificParams cpuParamsBatchSizeOne{{tnc, tnc, tnc}, {tnc, tnc, tnc}, {"ref_any"}, "ref_any"};

std::vector<ov::test::utils::SequenceTestsMode> mode{ov::test::utils::SequenceTestsMode::PURE_SEQ};
// oneDNN supports only sigmoid-tanh-tanh
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
// oneDNN supports only zero clip
std::vector<float> clip{0.f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD};

std::vector<ElementType> netPrecisions = { ElementType::f32 };

const std::vector<std::vector<InputShape>> staticShapes = {
    { { {}, { {10, 2, 10} } }, // Static shapes
      { {}, { {10, 1, 1} } },
      { {}, { {10, 1, 1} } },
      { {}, { {10} } } },
    { { {}, { {10, 2, 10} } }, // Static shapes
      { {}, { {10, 1, 10} } },
      { {}, { {10, 1, 10} } },
      { {}, { {10} } } },
    { { {}, { {1, 2, 10} } },  // Static shapes
      { {}, { {1, 1, 1} } },
      { {}, { {1, 1, 1} } },
      { {}, { {1} } } },
    { { {}, { {1, 2, 10} } },  // Static shapes
      { {}, { {1, 1, 10} } },
      { {}, { {1, 1, 10} } },
      { {}, { {1} } } },
    { { {}, { {1, 2, 10} } },  // Static shapes
      { {}, { {1, 1, 10} } },
      { {}, { {1, 1, 10} } },
      { {}, { {1} } } },
};

INSTANTIATE_TEST_SUITE_P(smoke_static, LSTMSequenceCPUTest,
                ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<InputShape>>{staticShapes[0], staticShapes[1]}),
                                   ::testing::ValuesIn(mode),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(direction),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParams),
                                   ::testing::Values(ov::AnyMap{})),
                LSTMSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_BatchSizeOne, LSTMSequenceCPUTest,
                ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<InputShape>>{staticShapes[3]}),
                                   ::testing::ValuesIn(mode),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(direction),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParamsBatchSizeOne),
                                   ::testing::Values(ov::AnyMap{})),
                LSTMSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_static_bf16, LSTMSequenceCPUTest,
                ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<InputShape>>{staticShapes[0]}),
                                   ::testing::ValuesIn(mode),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(direction),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParams),
                                   ::testing::ValuesIn(additionalConfig)),
                LSTMSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_static_bf16_BatchSizeOne, LSTMSequenceCPUTest,
                ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<InputShape>>{staticShapes[4]}),
                                   ::testing::ValuesIn(mode),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(direction),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParamsBatchSizeOne),
                                   ::testing::ValuesIn(additionalConfig)),
                LSTMSequenceCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicShapes = {
    { { {-1, {1, 5}, 10},                           // #0. Dynamic shape 0
        { {10, 2, 10}, {8, 3, 10}, {5, 4, 10} } },  // Target shapes
      { {{0, 15}, 1, 1},                            // Dynamic shape 1
        { {10, 1, 1}, {8, 1, 1}, {5, 1, 1} } },     // Target shapes
      { {{0, 15}, 1, 1},                            // Dynamic shape 2
        { {10, 1, 1}, {8, 1, 1}, {5, 1, 1} } },     // Target shapes
      { {{0, 12}},                                  // Dynamic shape 3
        { {10}, {8}, {5} } } },                     // Target shapes
    { { {{0, 11}, -1, 10},                          // #1. Dynamic shape 0
        { {10, 2, 10}, {3, 4, 10}, {5, 5, 10} } },  // Target shapes
      { {-1, 1, 10},                                // Dynamic shape 1
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10} } },  // Target shapes
      { {-1, 1, 10},                                // Dynamic shape 2
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10} } },  // Target shapes
      { {-1},                                       // Dynamic shape 3
        { {10}, {3}, {5} } } },                     // Target shapes
    { { {{0, 11}, -1, {5, 15}},                     // #2. Dynamic shape 0
        { {10, 2, 10}, {3, 4, 10}, {5, 5, 10} } },  // Target shapes
      { {-1, 1, -1},                                // Dynamic shape 1
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10} } },  // Target shapes
      { {-1, 1, -1},                                // Dynamic shape 2
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10} } },  // Target shapes
      { {-1},                                       // Dynamic shape 3
        { {10}, {3}, {5} } } },                     // Target shapes
    { { {-1, {0, 7}, 10},                           // #3. Dynamic shape 0
        { {1, 2, 10}, {1, 3, 10}, {1, 6, 10} } },   // Target shapes
      { {-1, 1, 1},                                 // Dynamic shape 1
        { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} } },      // Target shapes
      { {-1, 1, 1},                                 // Dynamic shape 2
        { {1, 1, 1}, {1, 1, 1}, {1, 1, 1} } },      // Target shapes
      { {-1},                                       // Dynamic shape 3
        { {1}, {1}, {1} } } },                      // Target shapes
    { { {1, -1, 10},                                // #4. Dynamic shape 0
        { {1, 2, 10}, {1, 4, 10}, {1, 8, 10} } },   // Target shapes
      { {1, 1, 10},                                 // Dynamic shape 1
        { {1, 1, 10}, {1, 1, 10}, {1, 1, 10} } },   // Target shapes
      { {1, 1, 10},                                 // Dynamic shape 2
        { {1, 1, 10}, {1, 1, 10}, {1, 1, 10} } },   // Target shapes
      { {-1},                                       // Dynamic shape 3
        { {1}, {1}, {1} } } },                      // Target shapes
    { { {-1, -1, -1},                               // #5. Dynamic shape 0
        { {1, 2, 10}, {1, 4, 10}, {1, 8, 10} } },   // Target shapes
      { {-1, -1, -1},                               // Dynamic shape 1
        { {1, 1, 10}, {1, 1, 10}, {1, 1, 10} } },   // Target shapes
      { {-1, -1, -1},                               // Dynamic shape 2
        { {1, 1, 10}, {1, 1, 10}, {1, 1, 10} } },   // Target shapes
      { {-1},                                       // Dynamic shape 3
        { {1}, {1}, {1} } } },                      // Target shapes
    { { {3, -1, {0, 12}},                           // #6. Dynamic shape 0
        { {3, 2, 10}, {3, 4, 10}, {3, 5, 10} } },   // Target shapes
      { {3, -1, {0, 12}},                           // Dynamic shape 1
        { {3, 1, 10}, {3, 1, 10}, {3, 1, 10} } },   // Target shapes
      { {3, -1, {0, 12}},                           // Dynamic shape 2
        { {3, 1, 10}, {3, 1, 10}, {3, 1, 10} } },   // Target shapes
      { {-1},                                       // Dynamic shape 3
        { {3}, {3}, {3} } }},                       // Target shapes
    { { {{0, 11}, -1, {5, 15}},                     // #7. Dynamic shape 0
        { {10, 2, 10}, {3, 4, 10}, {5, 5, 10}, {10, 2, 10}, {5, 5, 10} } },  // Target shapes
      { {-1, 1, -1},                                // Dynamic shape 1
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10}, {10, 1, 10}, {5, 1, 10} } },  // Target shapes
      { {-1, 1, -1},                                // Dynamic shape 2
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10}, {10, 1, 10}, {5, 1, 10} } },  // Target shapes
      { {-1},                                       // Dynamic shape 3
        { {10}, {3}, {5}, {10}, {5} } } },          // Target shapes
};

namespace dynamicShapesBatchSwitch {
  const int input_size = 240;
  const int seq_length = 1;
  const int hidden_size = 1024;
  const int num_directions = 1;
  const ov::test::utils::SequenceTestsMode mode = ov::test::utils::SequenceTestsMode::PURE_SEQ;
  CPUSpecificParams cpuParams{{ntc, tnc, tnc}, {ntc, tnc, tnc}, {"ref_any"}, "ref_any"};

  const std::vector<InputShape> shapes = {
    {
      // X: [batch_size, seq_length, input_size]
      {-1, seq_length, input_size},
      {
        {1, seq_length, input_size},
        {20, seq_length, input_size},
        {1, seq_length, input_size},
      }
    },
    {
      // initial_hidden_state: [batch_size, num_directions, hidden_size]
      {-1, num_directions, hidden_size},
      {
        {1, num_directions, hidden_size},
        {20, num_directions, hidden_size},
        {1, num_directions, hidden_size},
      }
    },
    {
      // initial_cell_state: [batch_size, num_directions, hidden_size]
      {-1, num_directions, hidden_size},
      {
        {1, num_directions, hidden_size},
        {20, num_directions, hidden_size},
        {1, num_directions, hidden_size},
      }
    },
    {
      // sequence_lengths: [batch_size]
      {-1},
      {
        {1},
        {20},
        {1}
      }
    },
  };
}; // namespace dynamicShapesBatchSwitch

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_batch, LSTMSequenceCPUTest,
            ::testing::Combine(::testing::Values(dynamicShapesBatchSwitch::shapes),
                               ::testing::Values(dynamicShapesBatchSwitch::mode),
                               ::testing::ValuesIn(activations),
                               ::testing::Values(0.0f),
                               ::testing::Values(ov::op::RecurrentSequenceDirection::FORWARD),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(dynamicShapesBatchSwitch::cpuParams),
                               ::testing::Values(ov::AnyMap{{"_dynamic_batch_test", "yes"}})),
            LSTMSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, LSTMSequenceCPUTest,
            ::testing::Combine(::testing::ValuesIn({dynamicShapes[0], dynamicShapes[1], dynamicShapes[2]}),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParams),
                               ::testing::Values(ov::AnyMap{})),
            LSTMSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_BatchSizeOne, LSTMSequenceCPUTest,
            ::testing::Combine(::testing::ValuesIn({dynamicShapes[4]}),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParamsBatchSizeOne),
                               ::testing::Values(ov::AnyMap{})),
            LSTMSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_dynamic, LSTMSequenceCPUTest,
            ::testing::Combine(::testing::ValuesIn({dynamicShapes[5], dynamicShapes[7]}),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParams),
                               ::testing::Values(ov::AnyMap{})),
            LSTMSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_dynamic_bf16, LSTMSequenceCPUTest,
            ::testing::Combine(::testing::ValuesIn({dynamicShapes[6]}),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParams),
                               ::testing::Values(additionalConfig[1])),
            LSTMSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_dynamic_bf16_BatchSizeOne, LSTMSequenceCPUTest,
            ::testing::Combine(::testing::ValuesIn({dynamicShapes[4]}),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParamsBatchSizeOne),
                               ::testing::Values(additionalConfig[1])),
            LSTMSequenceCPUTest::getTestCaseName);

// Odd but valid use case
std::vector<InputShape> mixedDynamicStaticBatch {
    {{ {2, 3}, 5, 10},                         // Dynamic shape 0
     { {2, 5, 10}, {2, 5, 10}, {2, 5, 10} } }, // Target shapes
    {{ {2, 3}, 1, 1},                          // Dynamic shape 1
      { {2, 1, 1}, {2, 1, 1}, {2, 1, 1} } },   // Target shapes
    {{ {2, 3}, 1, 1},                          // Dynamic shape 2
      { {2, 1, 1}, {2, 1, 1}, {2, 1, 1} } },   // Target shapes
    { {2},                                     // Static shape 3
      { {2}, {2}, {2} } }                      // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_mixedDynamicStaticBatch, LSTMSequenceCPUTest,
            ::testing::Combine(::testing::Values(mixedDynamicStaticBatch),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParams),
                               ::testing::Values(ov::AnyMap{})),
            LSTMSequenceCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
