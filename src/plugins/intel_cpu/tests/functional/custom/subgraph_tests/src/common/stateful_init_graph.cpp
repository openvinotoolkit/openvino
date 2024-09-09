// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>

#include "utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

// This test case is for stateful model with an init_state graph,
// and Assign is straight from ReadValue.
// What's more, during each iteration (except the first iteration),
// the state should keep unchanged untill it's resetted.

using InitGraphStatefulModelTestParams = std::vector<InputShape>;

class InitGraphStatefulModelBase : virtual public ov::test::SubgraphBaseTest,
                                   public testing::WithParamInterface<InitGraphStatefulModelTestParams>,
                                   public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InitGraphStatefulModelTestParams>& obj) {
        std::ostringstream result;
        std::vector<InputShape> inputShapes = obj.param;

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

        // iterating with state reset
        for (auto iters = 0; iters < 2; iters++) {
            // std::cout << "========= iters" << iters << std::endl;
            for (const auto& targetStaticShapeVec : targetStaticShapes) {
                // std::cout << "========= targetStaticShapeVec" << targetStaticShapeVec.front() << std::endl;
                generate_inputs(targetStaticShapeVec);
                validate();
            }

            // std::cout << "========= reset =============" << std::endl << std::endl;
            reset();
        }
    }

protected:
    void reset() {
        for (auto&& state : inferRequest.query_state()) {
            state.reset();
        }

        for (auto&& state : inferRequestRef.query_state()) {
            state.reset();
        }
    }

    void prepare() {
        compile_model();

        inferRequest = compiledModel.create_infer_request();
        ASSERT_TRUE(inferRequest);

        // Node with friendly name "init_graph/add" should be moved into subgraph.
        bool found_init_graph_add = false;
        for (auto node : compiledModel.get_runtime_model()->get_ops()) {
            if (node->get_friendly_name() == "init_graph/add") {
                found_init_graph_add = true;
                break;
            }
        }
        EXPECT_FALSE(found_init_graph_add);

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

    const ov::element::Type netPrc = ElementType::f32;
    ov::InferRequest inferRequestRef;
};

//
//                      ┌─────────┐
//                      │ Param0  │
//                      └────┬────┘
//                           |
//                        ┌──┴──┐
//                        │ Add │
//                        └──┬──┘
//                           |
//      ┌─────────┐    ┌─────┴─────┐
//      │ Param1  │    | ReadValue │...........
//      └────┬────┘    └─────┬─────┘          .
//           |            /     \             .
//           \           /       \            .
//            \         /         \           .
//             \       /           \          .
//             ┌───┴───┐       ┌────┴───┐     .
//             │  Add  │       │ Assign │......
//             └───┬───┘       └────────┘
//                 |
//            ┌────┴───┐
//            │ Result │
//            └────────┘
class InitGraphStatefulModelImmediatePair : public InitGraphStatefulModelBase {
public:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        auto& InputShapes = this->GetParam();
        init_input_shapes(InputShapes);
        ov::ParameterVector input_params;
        for (auto&& shape : inputDynamicShapes) {
            input_params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrc, shape));
        }

        input_params[0]->set_friendly_name("xa");
        input_params[1]->set_friendly_name("input_ids");

        // init_graph
        auto add_0 =
            std::make_shared<ov::op::v1::Add>(input_params[0], ov::op::v0::Constant::create(netPrc, {1}, {1.0f}));
        add_0->set_friendly_name("init_graph/add");

        // The ReadValue/Assign operations must be used in pairs in the model.
        // For each such a pair, its own variable object must be created.
        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes.front(), netPrc, variable_name});

        // Creating ov::Model
        auto read = std::make_shared<ov::op::v6::ReadValue>(add_0, variable);
        auto add_1 = std::make_shared<ov::op::v1::Add>(input_params[1], read);
        add_1->set_friendly_name("add_1");
        auto assign = std::make_shared<ov::op::v6::Assign>(read, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add_1);
        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), input_params);

        // ov::pass::Serialize serializer("InitGraphStatefulModelImmediatePair.xml",
        //                                "InitGraphStatefulModelImmediatePair.bin");
        // serializer.run_on_model(function);
    }
};

TEST_P(InitGraphStatefulModelImmediatePair, CompareWithRefs) {
    run();
}
namespace {
const std::vector<std::vector<InputShape>> inputShapes = {
    {
        // dynamic shape
        // param0, to simulate cross-attention kv-cache of Whisper model. State should
        // keep the same value of the first iteration no matter how param0 changes.
        {{1, -1},
         {{1, 3}, {1, 3}, {1, 1}, {1, 1}}},  // (B, L)
                                             // param1, to simulate input_ids for first-token, second-token, etc.
        {{1, -1}, {{1, 3}, {1, 3}, {1, 1}, {1, 1}}},  // (B, L)
    },
    {
        // static shape
        {{1, 2}, {{1, 2}}},  // input 0
        {{1, 2}, {{1, 2}}}   // input 1
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_StatefulInitGraph,
                         InitGraphStatefulModelImmediatePair,
                         ::testing::ValuesIn(inputShapes),
                         InitGraphStatefulModelImmediatePair::getTestCaseName);
}  // namespace