//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <common_test_utils/test_constants.hpp>
#include <vector>

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"

using namespace ov::test::behavior;

namespace {
// TODO: extend to test DRIVER config: E#88902
auto configs = []() {
    return std::vector<ov::AnyMap>{{{"NPU_COMPILER_TYPE", "MLIR"}, {"NPU_COMPILATION_MODE", "ReferenceSW"}}};
};

std::shared_ptr<ov::Model> getFunction() {
    const std::vector<size_t> inputShape = {1, 10, 12};
    const ov::element::Type_t ngPrc = ov::element::Type_t::f32;

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape({inputShape}))};
    params.front()->get_output_tensor(0).set_names({"Parameter_1"});

    auto relu = std::make_shared<ov::op::v0::Relu>(params[0]);
    relu->get_output_tensor(0).set_names({"Relu_2"});

    return std::make_shared<ov::Model>(relu, params, "SimpleActivation");
}

}  // namespace

class NPUInferRequestDynamicTests_NPU3720 : public OVInferRequestDynamicTests {
protected:
    void checkOutputFP16(const ov::Tensor& in, const ov::Tensor& actual) {
        auto net = ie->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
        ov::InferRequest req;
        req = net.create_infer_request();
        auto tensor = req.get_tensor(function->inputs().back().get_any_name());
        tensor.set_shape(in.get_shape());
        for (size_t i = 0; i < in.get_size(); i++) {
            tensor.data<ov::element_type_traits<ov::element::f32>::value_type>()[i] =
                    in.data<ov::element_type_traits<ov::element::f32>::value_type>()[i];
        }
        req.infer();
        OVInferRequestDynamicTests::checkOutput(actual, req.get_output_tensor(0));
    }
};

TEST_P(NPUInferRequestDynamicTests_NPU3720, InferDynamicNetworkWithImport) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].first};
    const std::string inputName = "Parameter_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(1, inOutShapes[1].first[0]), ov::Dimension(1, inOutShapes[1].first[1]),
                         ov::Dimension(1, inOutShapes[1].first[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto execNet = ie->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    const std::string outputName = "Relu_2";
    for (auto& shape : vectorShapes) {
        ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);
        OV_ASSERT_NO_THROW(req = execNet.create_infer_request());
        OV_ASSERT_NO_THROW(req.set_tensor(inputName, inTensor));
        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(req.get_tensor(inputName), req.get_tensor(outputName)));
    }
}

INSTANTIATE_TEST_SUITE_P(
        smoke_BehaviorTests, NPUInferRequestDynamicTests_NPU3720,
        ::testing::Combine(::testing::Values(getFunction()),
                           ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                                   {{1, 10, 12}, {1, 10, 12}}, {{1, 18, 15}, {1, 18, 15}}}),
                           ::testing::Values(ov::test::utils::DEVICE_NPU), ::testing::ValuesIn(configs())),
        ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);
