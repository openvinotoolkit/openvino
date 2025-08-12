// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/relu.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace behavior {

using OVInferRequestDynamicParams =
    std::tuple<std::shared_ptr<Model>,                                            // ov Model
               std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>,  // input/expected output shapes per
                                                                                  // inference
               std::string,                                                       // Device name
               ov::AnyMap                                                         // Config
               >;

class OVInferRequestDynamicTests : public testing::WithParamInterface<OVInferRequestDynamicParams>,
                                   public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<OVInferRequestDynamicParams> obj);

protected:
    void SetUp() override;
    bool checkOutput(const ov::Tensor& in, const ov::Tensor& actual);

    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    std::shared_ptr<Model> function;
    ov::AnyMap configuration;
    std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> inOutShapes;
};

class InferRequestDynamicTests : public OVInferRequestDynamicTests {
public:
    static std::shared_ptr<ov::Model> getFunction() {
        const std::vector<size_t> inputShape = {1, 10, 12};
        const ov::element::Type_t ngPrc = ov::element::Type_t::f32;

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape({inputShape}))};
        params.front()->get_output_tensor(0).set_names({"Parameter_1"});

        auto relu = std::make_shared<ov::op::v0::Relu>(params[0]);
        relu->get_output_tensor(0).set_names({"Relu_2"});

        return std::make_shared<ov::Model>(relu, params, "SimpleActivation");
    }

protected:
    void checkOutputFP16(const ov::Tensor& in, const ov::Tensor& actual) {
        auto net = core->compile_model(function, ov::test::utils::DEVICE_TEMPLATE);
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

TEST_P(InferRequestDynamicTests, InferDynamicNetwork) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].second};
    const std::string inputName = "Parameter_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(inOutShapes[1].first[0], inOutShapes[1].second[0]),
                         ov::Dimension(inOutShapes[1].first[1], inOutShapes[1].second[1]),
                         ov::Dimension(inOutShapes[1].first[2], inOutShapes[1].second[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto model = core->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    const std::string outputName = "Relu_2";
    for (auto& shape : vectorShapes) {
        ov::Tensor inTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, 100, 0);
        OV_ASSERT_NO_THROW(req = model.create_infer_request());

        bool inferShallFail = false;
        for (auto i = 0; i < shapes[inputName].rank().get_length(); ++i) {
            if (shape[i] > shapes[inputName].get_max_shape()[i]) {
                inferShallFail = true;
            }
        }

        if (!inferShallFail) {
            OV_ASSERT_NO_THROW(req.infer());
            OV_ASSERT_NO_THROW(checkOutputFP16(req.get_tensor(inputName), req.get_tensor(outputName)));
        } else {
            EXPECT_THROW(req.set_tensor(inputName, inTensor), ov::Exception);
        }
    }
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkSetShape) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].second};
    const std::string inputName = "Parameter_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(inOutShapes[1].first[0], inOutShapes[1].second[0]),
                         ov::Dimension(inOutShapes[1].first[1], inOutShapes[1].second[1]),
                         ov::Dimension(inOutShapes[1].first[2], inOutShapes[1].second[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto model = core->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    const std::string outputName = "Relu_2";

    OV_ASSERT_NO_THROW(req = model.create_infer_request());
    auto inputTensor = req.get_tensor(inputName);
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));

    for (auto& shape : vectorShapes) {
        OV_ASSERT_NO_THROW(inputTensor.set_shape(shape));

        bool inferShallFail = false;
        for (auto i = 0; i < shapes[inputName].rank().get_length(); ++i) {
            if (shape[i] > shapes[inputName].get_max_shape()[i]) {
                inferShallFail = true;
            }
        }

        if (!inferShallFail) {
            OV_ASSERT_NO_THROW(req.infer());
            OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));
        } else {
            EXPECT_THROW(req.infer(), ov::Exception);
        }
    }
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkSetShapeCPUTensor) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].second};
    const std::string inputName = "Parameter_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(inOutShapes[1].first[0], inOutShapes[1].second[0]),
                         ov::Dimension(inOutShapes[1].first[1], inOutShapes[1].second[1]),
                         ov::Dimension(inOutShapes[1].first[2], inOutShapes[1].second[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto model = core->compile_model(function, target_device, configuration);

    ov::InferRequest req;
    const std::string outputName = "Relu_2";

    ov::Shape originalShape = {1, 1, 5};

    auto inputTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, originalShape, 100, 0);
    OV_ASSERT_NO_THROW(req = model.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(inputName, inputTensor));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));

    for (auto& shape : vectorShapes) {
        OV_ASSERT_NO_THROW(inputTensor.set_shape(shape));

        bool inferShallFail = false;
        for (auto i = 0; i < shapes[inputName].rank().get_length(); ++i) {
            if (shape[i] > shapes[inputName].get_max_shape()[i]) {
                inferShallFail = true;
            }
        }

        if (!inferShallFail) {
            OV_ASSERT_NO_THROW(req.infer());
            OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));
        } else {
            EXPECT_THROW(req.infer(), ov::Exception);
        }
    }
}

TEST_P(InferRequestDynamicTests, InferDynamicNetworkImportSetShapeCPUTensor) {
    std::vector<ov::Shape> vectorShapes{inOutShapes[0].first, inOutShapes[0].second};
    const std::string inputName = "Parameter_1";
    std::map<std::string, ov::PartialShape> shapes;
    shapes[inputName] = {ov::Dimension(inOutShapes[1].first[0], inOutShapes[1].second[0]),
                         ov::Dimension(inOutShapes[1].first[1], inOutShapes[1].second[1]),
                         ov::Dimension(inOutShapes[1].first[2], inOutShapes[1].second[2])};
    OV_ASSERT_NO_THROW(function->reshape(shapes));

    auto compiled_model = core->compile_model(function, target_device, configuration);

    std::stringstream stream;
    compiled_model.export_model(stream);
    auto imported_model = core->import_model(stream, target_device, configuration);

    ov::InferRequest req;
    const std::string outputName = "Relu_2";

    ov::Shape originalShape = {1, 1, 5};

    auto inputTensor = ov::test::utils::create_and_fill_tensor(ov::element::f32, originalShape, 100, 0);
    OV_ASSERT_NO_THROW(req = imported_model.create_infer_request());
    OV_ASSERT_NO_THROW(req.set_tensor(inputName, inputTensor));
    OV_ASSERT_NO_THROW(req.infer());
    OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));

    for (auto& shape : vectorShapes) {
        OV_ASSERT_NO_THROW(inputTensor.set_shape(shape));

        bool inferShallFail = false;
        for (auto i = 0; i < shapes[inputName].rank().get_length(); ++i) {
            if (shape[i] > shapes[inputName].get_max_shape()[i]) {
                inferShallFail = true;
            }
        }

        if (!inferShallFail) {
            OV_ASSERT_NO_THROW(req.infer());
            OV_ASSERT_NO_THROW(checkOutputFP16(inputTensor, req.get_tensor(outputName)));
        } else {
            EXPECT_THROW(req.infer(), ov::Exception);
        }
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
