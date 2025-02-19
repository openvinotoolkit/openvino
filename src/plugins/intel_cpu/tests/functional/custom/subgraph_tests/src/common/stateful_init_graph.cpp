// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;
using InitGraphStatefulModelTestParams = std::tuple<std::vector<InputShape>,  // input shapes
                                                    bool                      // ReadValue Assgin Direct pair or not
                                                    >;
class InitGraphStatefulModelBase : virtual public ov::test::SubgraphBaseTest,
                                   public testing::WithParamInterface<InitGraphStatefulModelTestParams>,
                                   public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InitGraphStatefulModelTestParams>& obj) {
        std::ostringstream result;

        std::vector<InputShape> inputShapes;
        bool directPair;
        std::tie(inputShapes, directPair) = obj.param;

        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")";
        }
        result << "_DirectAssign=" << ov::test::utils::bool2str(directPair);
        result << ")";

        return result.str();
    }

    std::vector<ov::Tensor> calculate_refs() override {
        for (const auto& param : functionRefs->get_parameters()) {
            inferRequestRef.set_tensor(param->get_default_output(), inputs.at(matched_parameters[param]));
        }
        inferRequestRef.infer();

        auto outputs = std::vector<ov::Tensor>{};
        for (const auto& output : functionRefs->outputs()) {
            outputs.push_back(inferRequestRef.get_tensor(output));
        }

        return outputs;
    }

    std::vector<ov::Tensor> get_plugin_outputs() override {
        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        auto outputs = std::vector<ov::Tensor>{};
        for (const auto& output : function->outputs()) {
            outputs.push_back(inferRequest.get_tensor(output));
        }
        return outputs;
    }

    void run() override {
        prepare();

        auto&& states = inferRequest.query_state();
        auto&& refStates = inferRequestRef.query_state();

        for (size_t i = 0; i < targetStaticShapes.size(); i++) {
            for (auto iters = 0; iters < 5; iters++) {
                generate_inputs(targetStaticShapes[i]);

                if (iters & 0x1) {
                    states.front().reset();
                    refStates.front().reset();
                } else {
                    // generate and set state tensors every even iteration
                    using ov::test::utils::InputGenerateData;

                    auto stateShape = get_state_shape(i);
                    auto tensor = utils::create_and_fill_tensor(statePrc,
                                                                stateShape,
                                                                InputGenerateData{0, 1, 1, iters});
                    states.front().set_state(tensor);
                    refStates.front().set_state(tensor);
                }

                validate();
            }
        }
    }

protected:
    virtual void check_init_graph_node() = 0;

    virtual ov::Shape get_state_shape(size_t i) = 0;

    void prepare() {
        compile_model();

        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);

        check_init_graph_node();

        // ref
        functionRefs = function->clone();

        matched_parameters.clear();
        const auto& ref_params = functionRefs->get_parameters();
        const auto& params = function->get_parameters();
        for (size_t in_idx = 0; in_idx < params.size(); ++in_idx) {
            matched_parameters.insert({ref_params[in_idx], params[in_idx]});
        }

        auto compiledModelRef = core->compile_model(functionRefs, ov::test::utils::DEVICE_TEMPLATE);
        inferRequestRef = compiledModelRef.create_infer_request();
    }

    std::vector<InputShape> inputShapes;
    const ov::element::Type netPrc = ElementType::f32;
    ov::InferRequest inferRequestRef;
    ov::element::Type statePrc;
};

// ReadValue Assign direct pair
//
//             input_1   input_2
//                |        |
//              Add_1     /
//                \      /
//                 MatMul
//                   |
//   input_0     ReadValue ..........
//       \      /       \           .
//         Add_0      Assign ........
//          |
//        Result

class InitGraphStatefulModel : public InitGraphStatefulModelBase {
public:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;

        bool directPair;
        std::tie(inputShapes, directPair) = this->GetParam();

        init_input_shapes(inputShapes);
        ov::ParameterVector input_params;
        for (auto&& shape : inputDynamicShapes) {
            input_params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrc, shape));
        }

        input_params[0]->set_friendly_name("input_0");
        input_params[1]->set_friendly_name("input_1");
        input_params[2]->set_friendly_name("input_2");

        // init_graph
        auto add_1 =
            std::make_shared<ov::op::v1::Add>(input_params[1], ov::op::v0::Constant::create(netPrc, {1}, {1.0f}));
        add_1->set_friendly_name("init_graph/add_1");
        auto mm_0 = std::make_shared<ov::op::v0::MatMul>(add_1, input_params[2]);
        mm_0->set_friendly_name("init_graph/mm_0");

        const std::string variable_name("var_direct_pair");
        statePrc = netPrc;
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{{inputDynamicShapes[1][0], inputDynamicShapes[2][1]}, statePrc, variable_name});

        auto read = std::make_shared<ov::op::v6::ReadValue>(mm_0, variable);
        std::shared_ptr<ov::Node> add_0 = std::make_shared<ov::op::v1::Add>(input_params[0], read);
        add_0->set_friendly_name("add_0");
        auto assign = std::make_shared<ov::op::v6::Assign>(directPair ? read : add_0, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add_0);
        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), input_params);
    }

    void check_init_graph_node() override {
        // Node with friendly name "init_graph/add_1" and init_graph/mm_0 should be moved into subgraph.
        CheckNumberOfNodesWithType(compiledModel, "Add", 0);
        CheckNumberOfNodesWithType(compiledModel, "MatMul", 0);
    }

    ov::Shape get_state_shape(size_t i) override {
        return ov::Shape({inputShapes[1].second[i][0], inputShapes[2].second[i][1]});
    }
};

TEST_P(InitGraphStatefulModel, CompareWithRefs) {
    run();
}

// ReadValueWithSubgraph have different precision.
//
//         input[fp32]
//            |
//       Convert[fp32->fp16]
//            |
//        ReadValue ..........
//       /       \           .
//     Add      Assign .......
//      |
//    Result
class InitGraphStatefulDiffPrimitiveModel : public InitGraphStatefulModelBase {
public:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;

        configuration.insert({"SNIPPETS_MODE", "DISABLE"});

        std::tie(inputShapes, directPair) = this->GetParam();

        init_input_shapes(inputShapes);
        ov::ParameterVector input_params;
        for (auto&& shape : inputDynamicShapes) {
            input_params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrc, shape));
        }

        input_params[0]->set_friendly_name("input");

        // init_graph
        auto convert = std::make_shared<ov::op::v0::Convert>(input_params[0], ov::element::f16);
        convert->set_friendly_name("init_graph/convert");

        const std::string variable_name("var_diff_precison");
        statePrc = ov::element::f16;
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{{inputDynamicShapes[0]}, statePrc, variable_name});

        auto readvalue = std::make_shared<ov::op::v6::ReadValue>(convert, variable);

        std::shared_ptr<ov::Node> add =
            std::make_shared<ov::op::v1::Add>(readvalue, ov::op::v0::Constant::create(ov::element::f16, {1}, {1.0f}));

        auto assign = std::make_shared<ov::op::v6::Assign>(directPair ? readvalue : add, variable);

        auto res = std::make_shared<ov::op::v0::Result>(add);

        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), input_params);
    }

    void check_init_graph_node() override {
#if defined(OPENVINO_ARCH_ARM64)
        // Convert node is fused into Eltwise on arm platforms
        if (directPair) {
            CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
        } else {
            CheckNumberOfNodesWithType(compiledModel, "Convert", 1);
        }
#else
        CheckNumberOfNodesWithType(compiledModel, "Convert", 1);
#endif
    }

    ov::Shape get_state_shape(size_t i) override {
        return inputShapes[0].second[i];
    }

private:
    bool directPair;
};

TEST_P(InitGraphStatefulDiffPrimitiveModel, CompareWithRefs) {
    run();
}

namespace {
const std::vector<std::vector<InputShape>> inputShapes = {
    {
        // Dynamic shape.
        {{1, -1}, {{1, 2}, {1, 2}, {1, 1}}},
        {{2, -1}, {{2, 3}, {2, 10}, {2, 1}}},
        {{-1, 2}, {{3, 2}, {10, 2}, {1, 2}}},
    },
    {
        // Static shape.
        {{1, 1}, {{1, 1}}},
        {{4, 2}, {{4, 2}}},
        {{2, 10}, {{2, 10}}},
    }
};

const std::vector<bool> readValueAssginDirectPair = {true, false};

const auto testParams_smoke = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::ValuesIn(readValueAssginDirectPair));

INSTANTIATE_TEST_SUITE_P(smoke_StatefulInitGraph,
                         InitGraphStatefulModel,
                         testParams_smoke,
                         InitGraphStatefulModel::getTestCaseName);


const std::vector<std::vector<InputShape>> inputShapesDiffPrecision = {
    {
        // Dynamic shape.
        {{1, -1}, {{1, 10}, {1, 1}}},
    },
    {
        // Static shape.
        {{1, 1}, {{1, 1}}},
    }
};

const auto testParamsDiffPrecision_smoke = ::testing::Combine(
    ::testing::ValuesIn(inputShapesDiffPrecision),
    ::testing::ValuesIn(readValueAssginDirectPair));

INSTANTIATE_TEST_SUITE_P(smoke_StatefulInitGraph,
                         InitGraphStatefulDiffPrimitiveModel,
                         testParamsDiffPrecision_smoke,
                         InitGraphStatefulDiffPrimitiveModel::getTestCaseName);

}  // namespace

