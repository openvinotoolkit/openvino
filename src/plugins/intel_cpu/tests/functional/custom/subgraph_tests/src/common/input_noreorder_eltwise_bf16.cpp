// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class InputNoReorderEltwiseBF16 : virtual public SubgraphBaseStaticTest, public CPUTestsBase {
protected:
    virtual void set_output_type_and_config() {
        outType = ov::element::bf16;
        ov::AnyMap additional_config{ov::hint::inference_precision(ov::element::bf16)};
        configuration.insert(additional_config.begin(), additional_config.end());
    }
    void SetUp() override {
        auto netPrecision = inType = ov::element::f32;
        set_output_type_and_config();
        targetDevice = ov::test::utils::DEVICE_CPU;

        ov::Shape inputShape{2, 4, 4, 1};
        auto eltwiseType = ov::test::utils::EltwiseTypes::ADD;

        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputShape)};

        auto tensor = ov::test::utils::create_and_fill_tensor(netPrecision, inputShape);
        auto secondaryInput = std::make_shared<ov::op::v0::Constant>(tensor);

        auto eltwise = ov::test::utils::make_eltwise(input[0], secondaryInput, eltwiseType);

        function = makeNgraphFunction(netPrecision, input, eltwise, "Eltwise");
    }
};

/* FP32 network with enforced BF16 precision.
 * Test that no Reorder (or Convert) is inserted after Input.
 * Eltwise performs the conversion by itself.

    Input[FP32]        Constant[FP32]
          \                 /
           \               /
            X  No Reorder X
             \           /
           Eltwise[FP32->BF16] (or Subgraph[FP32->BF16])
                  |
                  |
             Output[BF16]
*/
TEST_F(InputNoReorderEltwiseBF16, smoke_CompareWithRefs) {
    run();

    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    CheckNumberOfNodesWithTypes(compiledModel, {"Eltwise", "Subgraph"}, 1);
}

class InputNoReorderEltwiseFP16 : public InputNoReorderEltwiseBF16 {
protected:
    void set_output_type_and_config() override {
        outType = ov::element::f16;
        ov::AnyMap additional_config{ov::hint::inference_precision(ov::element::f16)};
        configuration.insert(additional_config.begin(), additional_config.end());
    }
};

TEST_F(InputNoReorderEltwiseFP16, smoke_CompareWithRefs) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }

    run();

    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    CheckNumberOfNodesWithTypes(compiledModel, {"Eltwise", "Subgraph"}, 1);
}

}  // namespace test
}  // namespace ov
