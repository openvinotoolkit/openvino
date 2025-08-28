// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"

namespace {
using ov::test::InputShape;

using ReadValueAssignParams = std::tuple<
    InputShape,        // input shapes
    ov::element::Type  // input precision
>;

class ReadValueAssignGPUTest : virtual public ov::test::SubgraphBaseTest,
                               public testing::WithParamInterface<ReadValueAssignParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadValueAssignParams>& obj) {
        const auto& [input_shapes, input_precision] = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({input_shapes.first}) << "_";
        result << "TS=";
        for (const auto& shape : input_shapes.second) {
            result << ov::test::utils::partialShape2str({shape}) << "_";
        }
        result << "Precision=" << input_precision;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [input_shapes, input_precision] = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;

        init_input_shapes({input_shapes});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(input_precision, shape));
        }
        auto read_value = std::make_shared<ov::op::v3::ReadValue>(params.at(0), "v0");
        auto add = std::make_shared<ov::op::v1::Add>(read_value, params.at(0));
        auto assign = std::make_shared<ov::op::v3::Assign>(add, "v0");
        auto res = std::make_shared<ov::op::v0::Result>(add);
        function = std::make_shared<ov::Model>(ov::ResultVector { res }, ov::SinkVector { assign }, params);
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto data_tensor = ov::Tensor{funcInputs[0].get_element_type(), targetInputStaticShapes[0]};
        auto data = data_tensor.data<ov::element_type_traits<ov::element::i32>::value_type>();
        auto len = ov::shape_size(targetInputStaticShapes[0]);
        for (size_t i = 0; i < len; i++) {
            data[i] = static_cast<int>(i);
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), data_tensor});
    }
};

TEST_P(ReadValueAssignGPUTest, Inference) {
    run();
}

TEST_P(ReadValueAssignGPUTest, Inference_cached) {
    std::stringstream ss;
    ss << "gpu_model_cache_" << std::hash<std::string>{}(
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) +
          std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
    std::string cacheDirName = ss.str();
    {
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
        core->set_property(ov::cache_dir(cacheDirName));
        compile_model();
    }
    {
        run();
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
    }
}

const std::vector<InputShape> input_shapes_dyn = {
    {{-1, -1, -1, -1}, {{7, 4, 20, 20}, {19, 4, 20, 20}}}
};

INSTANTIATE_TEST_SUITE_P(smoke_ReadValueAssign_Static, ReadValueAssignGPUTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation({{7, 4, 20, 20}})),
                                            ::testing::Values(ov::element::i32)),
                         ReadValueAssignGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReadValueAssign_Dynamic, ReadValueAssignGPUTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_dyn),
                                            ::testing::Values(ov::element::i32)),
                         ReadValueAssignGPUTest::getTestCaseName);
} // namespace
