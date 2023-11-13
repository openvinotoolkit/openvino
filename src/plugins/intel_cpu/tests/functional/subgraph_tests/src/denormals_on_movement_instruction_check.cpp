// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/op/op.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

using namespace InferenceEngine;
using namespace ov::test;
using ngraph::helpers::EltwiseTypes;

namespace SubgraphTestsDefinitions {
typedef std::tuple<
        std::vector<InputShape>,                        // Input shapes
        EltwiseTypes,                                   // Eltwise operations
        std::string                                     // Device name
> DenormalTestTuple;

// There is lot of instructions mixed used for different data precision in cpu kernel. For example, movups is used to load int32 data type.
// There could be risky if FTZ/DAZ is enabled and valid for data movement instructions on present microarchitecture.
// This test is to check if denormals flag(FTZ/DAZ) have impact on data movement instructions.
// The kernel work correctly only FTZ/DAZ do not impact on thoes intructions.
class DenormalOnDataMovementInstructionCheck : public testing::WithParamInterface<DenormalTestTuple>,
                                               virtual public SubgraphBaseTest {
public:
static std::string getTestCaseName(const testing::TestParamInfo<DenormalTestTuple> &obj) {
        std::vector<InputShape> inputShapes;
        EltwiseTypes eltwiseOpType;
        std::string targetName;
        std::tie(inputShapes, eltwiseOpType, targetName) = obj.param;
        std::ostringstream results;

        results << "IS=(";
        for (const auto& shape : inputShapes) {
            results << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        results << ")_TS=(";
        for (const auto& shape : inputShapes) {
            for (const auto& item : shape.second) {
                results << ov::test::utils::vec2str(item) << "_";
            }
        }
        results << "eltwiseOpTypes=" << eltwiseOpType << "_";
        results << "targetDevice=" << targetName;

        return results.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        constexpr unsigned seed = 1u;
        constexpr uint32_t denormalsRange = (0xffffffffu >> 9u) - 1;
        testing::internal::Random random(seed);
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            auto tensor = ov::Tensor{funcInput.get_element_type(), targetInputStaticShapes[i]};
            auto data_ptr_int32 = static_cast<int32_t*>(tensor.data());
            size_t shape_size = ov::shape_size(targetInputStaticShapes[i]);
            for (size_t j = 0; j < shape_size; ++j) {
                auto denormal = random.Generate(denormalsRange) + 1;
                int32_t tmp;
                memcpy(&tmp, &denormal, sizeof(int32_t));
                data_ptr_int32[j] = tmp;
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
void SetUp() override {
    // force denormals flag is on
    configuration.insert({"CPU_DENORMALS_OPTIMIZATION", InferenceEngine::PluginConfigParams::YES});
    // force eltwise node path
    configuration.insert({InferenceEngine::PluginConfigInternalParams::KEY_SNIPPETS_MODE,
                              InferenceEngine::PluginConfigInternalParams::DISABLE});
    std::vector<InputShape> inputShapes;
    EltwiseTypes eltwiseOpType;
    std::tie(inputShapes, eltwiseOpType, targetDevice) = this->GetParam();

    init_input_shapes(inputShapes);

    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes)
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ElementType::i32, shape));
    auto eltwise = ngraph::builder::makeEltwise(params[0], params[1], eltwiseOpType);
    function = std::make_shared<ngraph::Function>(eltwise, params, "FTZ/DAZ_impact_on_data_movement_instructions");
}
};

TEST_P(DenormalOnDataMovementInstructionCheck, CompareWithRefs) {
    run();
}

namespace {

std::vector<std::vector<ngraph::Shape>> inputShapes = {
    {{1, 1, 1, 1}, {1, 1, 1, 1}},     // test uni_vmovss with load_scalar
    {{1, 3, 2, 32}, {1, 3, 2, 32}},   // test uni_vmovups with load_vector
};

std::vector<utils::EltwiseTypes> eltwiseOps = {
        utils::EltwiseTypes::ADD, utils::EltwiseTypes::MULTIPLY, utils::EltwiseTypes::SUBTRACT
};

INSTANTIATE_TEST_SUITE_P(smoke_DenormalOnDataMovementInstructionCheck, DenormalOnDataMovementInstructionCheck,
                        ::testing::Combine(
                                ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes)),
                                ::testing::ValuesIn(eltwiseOps),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        DenormalOnDataMovementInstructionCheck::getTestCaseName);

} // namespace

}// namespace SubgraphTestsDefinitions
