// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>

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
        result << "_DirectPair=" << ov::test::utils::bool2str(directPair);
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

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            for (auto iters = 0; iters < 2; iters++) {
                generate_inputs(targetStaticShapeVec);
                validate();
            }
            // Different input shape, reset is required.
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

    virtual void check_init_graph_node() = 0;

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

    const ov::element::Type netPrc = ElementType::f32;
    ov::InferRequest inferRequestRef;
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

        std::vector<InputShape> inputShapes;
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
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{{inputDynamicShapes[1][0], inputDynamicShapes[2][1]}, netPrc, variable_name});

        auto read = std::make_shared<ov::op::v6::ReadValue>(mm_0, variable);
        std::shared_ptr<ov::Node> add_0 = std::make_shared<ov::op::v1::Add>(input_params[0], read);
        add_0->set_friendly_name("add_0");
        auto assign = std::make_shared<ov::op::v6::Assign>(directPair ? read : add_0, variable);
        auto res = std::make_shared<ov::op::v0::Result>(add_0);
        function = std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), input_params);
    }

    void check_init_graph_node() override {
        // Node with friendly name "init_graph/add_1" and init_graph/mm_0 should be moved into subgraph.
        bool found_init_graph_add = false;
        bool found_init_graph_mm = false;
        for (auto node : compiledModel.get_runtime_model()->get_ops()) {
            if (node->get_friendly_name() == "init_graph/add_1") {
                found_init_graph_add = true;
                break;
            }
            if (node->get_friendly_name() == "init_graph/mm_0") {
                found_init_graph_mm = true;
                break;
            }
        }
        EXPECT_FALSE(found_init_graph_add);
        EXPECT_FALSE(found_init_graph_mm);
    }
};

TEST_P(InitGraphStatefulModel, CompareWithRefs) {
    run();
}
namespace {
const std::vector<std::vector<InputShape>> inputShapes = {
    {
        // Dynamic shape.
        {{1, -1}, {{1, 2}, {1, 2}, {1, 1}}},
        {{2, -1}, {{2, 3}, {2, 2}, {2, 1}}},
        {{-1, 2}, {{3, 2}, {2, 2}, {1, 2}}},
    },
    {
        // Static shape.
        {{1, 1}, {{1, 1}}},
        {{4, 2}, {{4, 2}}},
        {{2, 1}, {{2, 1}}},
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

}  // namespace
