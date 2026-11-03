// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "openvino/op/multiply.hpp"
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
        const auto& [use_init_subgraph, cpu_params] = obj.param;
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
        const auto& [use_init_subgraph, cpu_params] = this->GetParam();
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

using ReadValueStateReshapeParams = std::tuple<bool, bool, size_t>;

class ReadValueStateReshapeTest : public testing::TestWithParam<ReadValueStateReshapeParams> {};

TEST_P(ReadValueStateReshapeTest, ReinitializeAfterSetState) {
    const auto [fresh_request, direct_assign, replacement_batch] = GetParam();
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{-1, 4}, ov::element::f32, "state"});
    auto two = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.f});
    auto init = std::make_shared<ov::op::v1::Multiply>(parameter, two);
    auto read = std::make_shared<ov::op::v6::ReadValue>(init, variable);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(read, parameter);
    std::shared_ptr<ov::Node> assigned_value = direct_assign ? std::static_pointer_cast<ov::Node>(read) : multiply;
    auto assign = std::make_shared<ov::op::v6::Assign>(assigned_value, variable);
    auto result = std::make_shared<ov::op::v0::Result>(multiply);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::SinkVector{assign}, ov::ParameterVector{parameter});
    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled = core->compile_model(model,
                                        ov::test::utils::DEVICE_CPU,
                                        {ov::hint::inference_precision(ov::element::f32), ov::num_streams(1)});
    auto request = compiled.create_infer_request();

    auto infer_and_check = [&](size_t batch, float input_value, float expected_value) {
        ov::Tensor input(ov::element::f32, {batch, 4});
        std::fill_n(input.data<float>(), input.get_size(), input_value);
        request.set_input_tensor(input);
        request.infer();
        auto output = request.get_output_tensor();
        ASSERT_EQ(output.get_shape(), (ov::Shape{batch, 4}));
        for (size_t i = 0; i < output.get_size(); ++i) {
            ASSERT_FLOAT_EQ(output.data<const float>()[i], expected_value);
        }
    };

    for (size_t iteration = 0; iteration < 3; ++iteration) {
        SCOPED_TRACE(iteration);
        ASSERT_NO_FATAL_FAILURE(infer_and_check(4, 1.f, 2.f));
        // Resetting without an external descriptor change must still recompute the initial state.
        request.reset_state();
        ASSERT_NO_FATAL_FAILURE(infer_and_check(4, 2.f, 8.f));
        {
            ov::Tensor replacement(ov::element::f32, {replacement_batch, 4});
            std::fill_n(replacement.data<float>(), replacement.get_size(), 7.f);
            request.query_state().front().set_state(replacement);
        }
        ASSERT_NO_FATAL_FAILURE(infer_and_check(replacement_batch, 1.f, 7.f));

        if (fresh_request) {
            request = compiled.create_infer_request();
        } else {
            request.reset_state();
        }
        // The initialization input is unchanged; the shared output may have been reshaped externally.
        ASSERT_NO_FATAL_FAILURE(infer_and_check(4, 2.f, 8.f));
        request.reset_state();
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_ReadValueStateReshape,
                         ReadValueStateReshapeTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(size_t{3}, size_t{4}, size_t{5})));
}  // namespace test
}  // namespace ov
