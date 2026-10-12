// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_request_dynamic_utils.hpp"

#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "common/utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/compiled_model.hpp"

using namespace ov::test::behavior;

namespace {

const std::string inputName = "Parameter_1";
const std::string outputName = "Relu_2";

const std::vector<ov::AnyMap> config = {{{"NPU_COMPILATION_MODE", "ReferenceSW"}}};

}  // namespace

TEST_P(InferRequestDynamicTests, InferDynamicNetwork) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].second};
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(inOutShapes[1].first[0], inOutShapes[1].second[0]),
                         ov::Dimension(inOutShapes[1].first[1], inOutShapes[1].second[1]),
                         ov::Dimension(inOutShapes[1].first[2], inOutShapes[1].second[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto model = ie->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    for (auto& shape : vectorShapes) {
        ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);
        OV_ASSERT_NO_THROW(req = model.create_infer_request());

        if (exceedsUpperBounds(shape, shapes[inputName])) {
            EXPECT_THROW(req.set_tensor(inputName, inTensor), ov::Exception);
            continue;
        }

        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(req.get_tensor(inputName), req.get_tensor(outputName)));
    }
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkSetShape) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].second};
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(inOutShapes[1].first[0], inOutShapes[1].second[0]),
                         ov::Dimension(inOutShapes[1].first[1], inOutShapes[1].second[1]),
                         ov::Dimension(inOutShapes[1].first[2], inOutShapes[1].second[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto model = ie->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = model.create_infer_request());
    auto inputTensor = req.get_tensor(inputName);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));

    for (auto& shape : vectorShapes) {
        OV_ASSERT_NO_THROW(inputTensor.set_shape(shape));

        if (exceedsUpperBounds(shape, shapes[inputName])) {
            EXPECT_THROW(req.infer(), ov::Exception);
            continue;
        }

        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));
    }
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkSetShapeCPUTensor) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].second};
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(inOutShapes[1].first[0], inOutShapes[1].second[0]),
                         ov::Dimension(inOutShapes[1].first[1], inOutShapes[1].second[1]),
                         ov::Dimension(inOutShapes[1].first[2], inOutShapes[1].second[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto model = ie->compile_model(function, target_device, configuration);

    const ov::Shape originalShape = {1, 1, 5};
    auto inputTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, originalShape, 100, 0);

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = model.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(inputName, inputTensor));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));

    for (auto& shape : vectorShapes) {
        OV_ASSERT_NO_THROW(inputTensor.set_shape(shape));

        if (exceedsUpperBounds(shape, shapes[inputName])) {
            EXPECT_THROW(req.infer(), ov::Exception);
            continue;
        }

        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));
    }
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkImportSetShapeCPUTensor) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].second};
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(inOutShapes[1].first[0], inOutShapes[1].second[0]),
                         ov::Dimension(inOutShapes[1].first[1], inOutShapes[1].second[1]),
                         ov::Dimension(inOutShapes[1].first[2], inOutShapes[1].second[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto compiled_model = ie->compile_model(function, target_device, configuration);

    std::stringstream stream;
    compiled_model.export_model(stream);
    auto imported_model = ie->import_model(stream, target_device, configuration);

    const ov::Shape originalShape = {1, 1, 5};
    auto inputTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, originalShape, 100, 0);

    ov::InferRequest req;
    OV_ASSERT_NO_THROW(req = imported_model.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(inputName, inputTensor));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));

    for (auto& shape : vectorShapes) {
        OV_ASSERT_NO_THROW(inputTensor.set_shape(shape));

        if (exceedsUpperBounds(shape, shapes[inputName])) {
            EXPECT_THROW(req.infer(), ov::Exception);
            continue;
        }

        OV_ASSERT_NO_THROW(req.infer());
        OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));
    }
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(OVInferRequestDynamicTests);

INSTANTIATE_TEST_SUITE_P(
    smoke_BehaviorTests,
    InferRequestDynamicTests,
    ::testing::Combine(::testing::Values(InferRequestDynamicTests::getFunction()),
                       ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{
                           {{1, 10, 18}, {6, 12, 15}},
                           {{1, 2, 14}, {5, 11, 18}}}),
                       ::testing::Values(ov::test::utils::DEVICE_NPU),
                       ::testing::ValuesIn(ov::test::utils::mergeConfigs(config,
                                                                        ov::test::utils::quietCompilerLogsConfig))),
    ov::test::utils::appendPlatformTypeTestName<OVInferRequestDynamicTests>);
