// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

/* The main purpose of this test set is to test ReadValue->Assign direct connection optimizations, i.e.
   dropping the MemoryOutput node.
*/

namespace ov {
namespace test {

using namespace CPUTestUtils;

//  ┌────────┐ ┌────────┐
//  │ Param2 │ │ Param1 │ |---------------|
//  └───┬────┘ └────┬───┘ |               |
//      │     |-----------|┌─────────┐    |
//      │     |     │      │Constant │    |
//      │     |     │      └───┬─────┘    |
//      │     | ┌───┴────┐     │          |
//      │     | │Multiply├─────┘          |
//      │     | └───┬────┘                |  <- Optional Init Subgraph
//      │     |     │      ┌─────────┐    |
//      │     |     │      │Constant │    |
//      │     |     │      └───┬─────┘    |
//      │     | ┌───┴────┐     │          |
//      │     | │ Add    ├─────┘          |
//      │     | └───┬────┘                |
//      │     |     │                     |
//      │     |---------------------------|
//      │           │
//      │           │
//      │           │
//      │     ┌─────┴─────┐
//      │     │ ReadValue │
//      │     └─────┬─────┘
//      │           │   \
//      │        ┌──┴──┐ \
//      └────────┤ Add │  \┌────────┐
//               └──┬──┘   │ Assign │
//                  │      └────────┘
//                  │
//             ┌────┴────┐
//             │ Result1 │
//             └─────────┘

typedef std::tuple<
    bool,  // include init subgraph
    CPUSpecificParams
> ReadValueAssignTestParams;

class ReadValueAssignTest : public testing::WithParamInterface<ReadValueAssignTestParams>,
                            virtual public SubgraphBaseTest,
                            public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadValueAssignTestParams> &obj) {
        bool use_init_subgraph = false;
        CPUSpecificParams cpu_params;
        std::tie(use_init_subgraph, cpu_params) = obj.param;

        std::ostringstream results;
        results << "Init_Graph=" << (use_init_subgraph ? "True" : "False") << "_";
        results << CPUTestsBase::getTestCaseName(cpu_params);
        return results.str();
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const ov::Shape tensor_shape = {3, 32, 7, 7};

        InputShape param1_shape = {{-1, 32, -1, -1}, {tensor_shape}};
        InputShape param2_shape = {{-1, -1, -1, -1}, {tensor_shape}};

        bool use_init_subgraph = false;
        CPUSpecificParams cpu_params;
        std::tie(use_init_subgraph, cpu_params) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpu_params;
        selectedType = makeSelectedTypeStr(selectedType, net_prc);

        init_input_shapes({param1_shape, param2_shape});

        ov::ParameterVector params;
        params.push_back(std::make_shared<ov::op::v0::Parameter>(net_prc, inputDynamicShapes[0]));
        params.push_back(std::make_shared<ov::op::v0::Parameter>(net_prc, inputDynamicShapes[1]));
        std::shared_ptr<ov::Node> last_node = params.front();

        if (use_init_subgraph) {
            //build init subgraph
            auto const1 = utils::make_constant(net_prc, tensor_shape);
            auto const2 = utils::make_constant(net_prc, tensor_shape);
            auto multiply = utils::make_eltwise(last_node, const1, utils::EltwiseTypes::MULTIPLY);
            auto add = utils::make_eltwise(multiply, const2, utils::EltwiseTypes::ADD);
            last_node = add;
        }

        const std::string variable_name("variable0");
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{inputDynamicShapes[0], net_prc, variable_name});

        auto read = std::make_shared<ov::op::v6::ReadValue>(last_node, variable);
        auto assign = std::make_shared<ov::op::v6::Assign>(read, variable);
        auto add = utils::make_eltwise(params[1], read, utils::EltwiseTypes::ADD);

        add->get_rt_info() = getCPUInfo();
        auto res = std::make_shared<ov::op::v0::Result>(add);

        function =
            std::make_shared<ov::Model>(ov::ResultVector({res}), ov::SinkVector({assign}), params, "ReadValueAssign");
    }

protected:
    const ov::Shape tensor_shape = {3, 32, 7, 7};
    const ElementType net_prc = element::f32;
};

TEST_P(ReadValueAssignTest, CompareWithRefs) {
    compile_model();
    inferRequest = compiledModel.create_infer_request();
    ASSERT_TRUE(inferRequest);

    // use the Template plugin as a reference

    auto compiledReferenceModel = core->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
    auto inferRequestRef = compiledReferenceModel.create_infer_request();
    ASSERT_TRUE(inferRequestRef);

    generate_inputs(targetStaticShapes.front());
    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
        inferRequestRef.set_tensor(input.first, input.second);
    }

    constexpr int infer_count = 3lu;

    auto&& states = inferRequest.query_state();
    auto&& refStates = inferRequestRef.query_state();

    for (int i = 0; i < infer_count; ++i) {
        // set states

        if (i & 0x1) {
            //reset every odd iteration
            states.front().reset();
            refStates.front().reset();
        } else {
            // generate and set state tensors every even iteration
            using ov::test::utils::InputGenerateData;

            auto tensor =
                ov::test::utils::create_and_fill_tensor(net_prc, tensor_shape, InputGenerateData{0, 10, 1, i});
            states.front().set_state(tensor);
            refStates.front().set_state(tensor);
        }

        inferRequest.infer();
        inferRequestRef.infer();
        auto outputs = function->outputs();

        auto result = inferRequest.get_tensor(outputs[0]);

        auto result_ref = inferRequestRef.get_tensor(outputs[0]);

        ov::test::utils::compare(result, result_ref, 1e-4, 1e-4);
    }
    CheckNumberOfNodesWithTypes(compiledModel, {"MemoryOutput", "Assign"}, 0);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ReadValue_Assign,
    ReadValueAssignTest,
    ::testing::Combine(::testing::Values(true, false),
                       ::testing::Values(CPUSpecificParams{{nchw, nchw}, {nchw}, {""}, "any_type"},
                                         CPUSpecificParams{{nhwc, nhwc}, {nhwc}, {""}, "any_type"})),
    ReadValueAssignTest::getTestCaseName);
}  // namespace test
}  // namespace ov