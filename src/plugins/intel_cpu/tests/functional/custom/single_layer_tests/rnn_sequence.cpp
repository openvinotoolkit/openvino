// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/rnn_cell.hpp"
#include "utils/cpu_test_utils.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "common_test_utils/ov_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using RNNSequenceCpuSpecificParams = typename std::tuple<
        std::vector<InputShape>,                  // Shapes
        ov::test::utils::SequenceTestsMode,       // Pure Sequence or TensorIterator
        std::vector<std::string>,                 // Activations
        float,                                    // Clip
        ov::op::RecurrentSequenceDirection,       // Direction
        ElementType,                              // Network precision
        CPUSpecificParams,                        // CPU specific params
        ov::AnyMap        // Additional config
>;

class RNNSequenceCPUTest : public testing::WithParamInterface<RNNSequenceCpuSpecificParams>,
                           virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RNNSequenceCpuSpecificParams> &obj) {
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
        if (inputDynamicShapes.size() == 2 && inputDynamicShapes[0][0].is_dynamic() && inputDynamicShapes[1][0].is_dynamic())
            throw std::runtime_error("Invalid test case. If 3rd input is constant, batch dimension must be static.");

        const size_t hiddenSize = targetStaticShapes.front()[1][2];
        const size_t inputSize = targetStaticShapes.front()[0][2];
        const size_t numDirections = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        auto it = additionalConfig.find(ov::hint::inference_precision.name());
        if (it != additionalConfig.end() && it->second.as<ov::element::Type>() == ov::element::bf16) {
            selectedType = makeSelectedTypeStr(selectedType, ElementType::bf16);
        } else {
            selectedType = makeSelectedTypeStr(selectedType, netPrecision);
        }

        if (selectedType.find("bf16") != std::string::npos) {
            rel_threshold = 5e-2;
        } else if (selectedType.find("f32") != std::string::npos) {
            rel_threshold = 1e-4;
        }

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrecision, shape));
        }
        const size_t batchSize = inputDynamicShapes[0][0].is_static() ? inputDynamicShapes[0][0].get_length() :
            inputDynamicShapes[1][0].is_static() ? inputDynamicShapes[1][0].get_length() :
            inputDynamicShapes.size() > 2 && inputDynamicShapes[2][0].is_static() ? inputDynamicShapes[2][0].get_length() :
            1lu;
        if (inputDynamicShapes.size() > 2) {
            if (!inputDynamicShapes[2].is_dynamic() &&
                    seqMode != ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM &&
                    seqMode != ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM) {
                params.pop_back();
            } else {
                params[2]->set_element_type(ElementType::i64);
            }
        }

        ov::OutputVector paramsOuts;
        for (const auto& param : params)
          paramsOuts.push_back(param);

        std::vector<ov::Shape> WRB = {{numDirections, hiddenSize, inputSize}, {numDirections, hiddenSize, hiddenSize}, {numDirections, hiddenSize},
                                        {batchSize}};
        auto rnn_sequence =
            utils::make_rnn(paramsOuts, WRB, hiddenSize, activations, {}, {}, clip, true, direction, seqMode);
        function = makeNgraphFunction(netPrecision, params, rnn_sequence, "rnnSequence");

        if (seqMode != ov::test::utils::SequenceTestsMode::PURE_SEQ) {
            ov::pass::Manager manager;
            if (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                manager.register_pass<ov::pass::BidirectionalRNNSequenceDecomposition>();
            manager.register_pass<ov::pass::ConvertRNNSequenceToTensorIterator>();
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
        if (funcInputs.size() > 2) {
            const auto& seqLenInput = inputs.find(funcInputs[2].get_node_shared_ptr());
            if (seqLenInput == inputs.end())
                throw std::runtime_error("Could not find Sequence length input.");

            auto lenData = seqLenInput->second.data<ov::element_type_traits<ElementType::i64>::value_type>();
            std::fill(lenData, lenData + batchSize, maxSeqLen);
        }
    }
};

TEST_P(RNNSequenceCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RNNSeq");
}

namespace {
/* CPU PARAMS */
std::vector<ov::AnyMap> additionalConfig = {{ov::hint::inference_precision(ov::element::f32)},
                                            {ov::hint::inference_precision(ov::element::bf16)}};

CPUSpecificParams cpuParams{{ntc, tnc}, {ntc, tnc}, {"ref_any"}, "ref_any"};
CPUSpecificParams cpuParamsBatchSizeOne{{tnc, ntc}, {tnc, tnc}, {"ref_any"}, "ref_any"};

std::vector<ov::test::utils::SequenceTestsMode> mode{ov::test::utils::SequenceTestsMode::PURE_SEQ};
// output values increase rapidly without clip, so use only seq_lengths = 2
std::vector<size_t> seq_lengths_zero_clip{ 2 };
std::vector<std::vector<std::string>> activations = {{"relu"}, {"sigmoid"}, {"tanh"}};
// oneDNN supports only zero clip
std::vector<float> clip{0.f};

std::vector<ov::op::RecurrentSequenceDirection> direction{ov::op::RecurrentSequenceDirection::FORWARD};

std::vector<ElementType> netPrecisions = { ElementType::f32 };

const std::vector<std::vector<InputShape>> staticShapes = {
    { { {}, { {10, 2, 10} } }, // Static shapes
      { {}, { {10, 1, 1} } },
      { {}, { {10} } } },
    { { {}, { {10, 2, 10} } }, // Static shapes
      { {}, { {10, 1, 10} } },
      { {}, { {10} } } },
    { { {}, { {1, 2, 10} } }, // Static shapes
      { {}, { {1, 1, 1} } },
      { {}, { {1} } } },
    { { {}, { {1, 2, 10} } }, // Static shapes
      { {}, { {1, 1, 10} } },
      { {}, { {1} } } }
};

INSTANTIATE_TEST_SUITE_P(smoke_static, RNNSequenceCPUTest,
                ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<InputShape>>{staticShapes[0], staticShapes[1]}),
                                   ::testing::ValuesIn(mode),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(direction),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParams),
                                   ::testing::Values(ov::AnyMap{})),
                RNNSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_static_BatchSizeOne, RNNSequenceCPUTest,
                ::testing::Combine(::testing::Values(staticShapes[3]),
                                   ::testing::ValuesIn(mode),
                                   ::testing::ValuesIn(activations),
                                   ::testing::ValuesIn(clip),
                                   ::testing::ValuesIn(direction),
                                   ::testing::ValuesIn(netPrecisions),
                                   ::testing::Values(cpuParamsBatchSizeOne),
                                   ::testing::Values(ov::AnyMap{})),
                RNNSequenceCPUTest::getTestCaseName);

const std::vector<std::vector<InputShape>> dynamicShapes = {
    { { {-1, {1, 5}, 10},                                // #0. Dynamic shape 0
        { {10, 2, 10}, {8, 3, 10}, {5, 4, 10} } },       // Target shapes
      { {{0, 15}, 1, 1},                                 // Dynamic shape 1
        { {10, 1, 1}, {8, 1, 1}, {5, 1, 1} } },          // Target shapes
      { {{0, 12}},                                       // Dynamic shape 2
        { {10}, {8}, {5} } } },                          // Target shapes
    { { {{0, 11}, -1, 10},                               // #1. Dynamic shape 0
        { {10, 2, 10}, {3, 4, 10}, {5, 5, 10} } },       // Target shapes
      { {-1, 1, 10},                                     // Dynamic shape 1
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10} } },       // Target shapes
      { {-1},                                            // Dynamic shape 2
        { {10}, {3}, {5} } } },                          // Target shapes
    { { {{0, 11}, -1, {5, 15}},                          // #2. Dynamic shape 0
        { {10, 2, 10}, {3, 4, 10}, {5, 5, 10} } },       // Target shapes
      { {-1, -1, {8, 11}},                               // Dynamic shape 1
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10} } },       // Target shapes
      { {-1},                                            // Dynamic shape 2
        { {10}, {3}, {5} } } },                          // Target shapes
    { { {-1, {0, 7}, 10},                                // #3. Dynamic shape 0
        { {1, 2, 10}, {2, 3, 10}, {1, 6, 10} } },        // Target shapes
      { {-1, 1, 1},                                      // Dynamic shape 1
        { {1, 1, 1}, {2, 1, 1}, {1, 1, 1} } },           // Target shapes
      { {-1},                                            // Dynamic shape 2
        { {1}, {2}, {1} } } },                           // Target shapes
    { { {1, -1, 10},                                     // #4. Dynamic shape 0
        { {1, 2, 10}, {1, 4, 10}, {1, 8, 10} } },        // Target shapes
      { {1, 1, 10},                                      // Dynamic shape 1
        { {1, 1, 10}, {1, 1, 10}, {1, 1, 10} } },        // Target shapes
      { {-1},                                            // Dynamic shape 2
        { {1}, {1}, {1} } } },                           // Target shapes
    { { {-1, -1, -1},                                    // #5. Dynamic shape 0
        { {1, 2, 10}, {1, 4, 10}, {1, 8, 10} } },        // Target shapes
      { {-1, -1, -1},                                    // Dynamic shape 1
        { {1, 1, 10}, {1, 1, 10}, {1, 1, 10} } },        // Target shapes
      { {-1},                                            // Dynamic shape 2
        { {1}, {1}, {1} } } },                           // Target shapes
    { { {7, {1, 5}, 10},                                 // #6. Dynamic shape 0
        { {7, 2, 10}, {7, 3, 10}, {7, 5, 10}} },         // Target shapes
      { {7, 1, 1},                                       // Dynamic shape 1
        { {7, 1, 1}, {7, 1, 1}, {7, 1, 1} } },           // Target shapes
      { {-1},                                            // Dynamic shape 2
        { {7}, {7}, {7}} }},                             // Target shapes
    { { {5, -1, 10},                                     // #7. Dynamic shape 0
        { {5, 2, 10}, {5, 4, 10}, {5, 5, 10} } },        // Target shapes
      { {5, 1, 10},                                      // Dynamic shape 1
        { {5, 1, 10}, {5, 1, 10}, {5, 1, 10} } },        // Target shapes
      { {-1},                                            // Dynamic shape 2
        { {5}, {5}, {5} } }},                            // Target shapes
    { { {{0, 11}, -1, 10},                               // #8. Dynamic shape 0
        { {10, 2, 10}, {3, 4, 10}, {5, 5, 10}, {10, 2, 10}, {5, 5, 10} } },   // Target shapes
      { {-1, 1, 10},                                     // Dynamic shape 1
        { {10, 1, 10}, {3, 1, 10}, {5, 1, 10}, {10, 1, 10}, {5, 1, 10} } }, // Target shapes
      { {-1},                                            // Dynamic shape 2
        { {10}, {3}, {5}, {10}, {5} } } }                // Target shapes
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic, RNNSequenceCPUTest,
            ::testing::Combine(::testing::ValuesIn({dynamicShapes[0], dynamicShapes[1], dynamicShapes[2], dynamicShapes[3]}),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParams),
                               ::testing::Values(ov::AnyMap{})),
            RNNSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_BatchSizeOne, RNNSequenceCPUTest,
            ::testing::Combine(::testing::Values(dynamicShapes[4]),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParamsBatchSizeOne),
                               ::testing::Values(ov::AnyMap{})),
            RNNSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_dynamic, RNNSequenceCPUTest,
            ::testing::Combine(::testing::ValuesIn({dynamicShapes[5], dynamicShapes[8]}),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParams),
                               ::testing::Values(ov::AnyMap{})),
            RNNSequenceCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_dynamic_bf16, RNNSequenceCPUTest,
            ::testing::Combine(::testing::ValuesIn({dynamicShapes[6], dynamicShapes[7]}),
                               ::testing::ValuesIn(mode),
                               ::testing::ValuesIn(activations),
                               ::testing::ValuesIn(clip),
                               ::testing::ValuesIn(direction),
                               ::testing::ValuesIn(netPrecisions),
                               ::testing::Values(cpuParams),
                               ::testing::Values(additionalConfig[1])),
            RNNSequenceCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov
