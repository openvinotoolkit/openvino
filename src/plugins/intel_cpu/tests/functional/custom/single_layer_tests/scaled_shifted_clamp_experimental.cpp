// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class ScaledShiftedClampFusionCPUTest : public virtual SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice     = utils::DEVICE_CPU;
        const auto t     = element::f32;
        const Shape s{1, 4, 8, 8};
        auto       param = std::make_shared<op::v0::Parameter>(t, s);
        auto       scale = op::v0::Constant::create(t, Shape{1}, {2.0F});
        auto       bias  = op::v0::Constant::create(t, Shape{1}, {0.5F});
        auto       mul   = std::make_shared<op::v1::Multiply>(param, scale);
        auto       add   = std::make_shared<op::v1::Add>(mul, bias);
        auto       clamp = std::make_shared<op::v0::Clamp>(add, -1.0, 1.0);
        function         = std::make_shared<Model>(clamp, ParameterVector{param});
    }

    void check_fused() {
        const auto exec = compiledModel.get_runtime_model();
        bool       found = false;
        for (const auto& n : exec->get_ordered_ops()) {
            const auto layer = n->get_rt_info().at(exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer == "ScaledShiftedClampExperimental") {
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found) << "ScaledShiftedClampExperimental node not found in exec graph";
    }
};

TEST_F(ScaledShiftedClampFusionCPUTest, smoke_Fuses) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    check_fused();
}

}  // namespace test
}  // namespace ov
