// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "behavior/compiled_model/import_export.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/core/model_util.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::element::Type_t> netPrecisions = {
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64,
    ov::element::u8,
    ov::element::u16,
    ov::element::u32,
    ov::element::u64,
    ov::element::f16,
    ov::element::f32,
};
const ov::AnyMap empty_property = {};

using ov::op::v0::Parameter, ov::op::v0::Result;

INSTANTIATE_TEST_SUITE_P(smoke_serialization,
                         OVClassCompiledModelImportExportTestP,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         ::testing::PrintToStringParamName());

TEST_P(OVClassCompiledModelImportExportTestP, importExportModelWithTypeRelaxedExtension) {
    // Create model with interpolate which v0 and v4 are supported by TypeRelaxedExtension
    constexpr auto elementType = ov::element::f32;
    auto core = ov::test::utils::PluginCache::get().core();
    std::shared_ptr<ov::Model> model;

    {
        using ov::op::v4::Interpolate;

        ov::ParameterVector inputs;
        auto data = std::make_shared<Parameter>(elementType, ov::PartialShape{1, 3, 64, 64});
        data->set_friendly_name("data");
        auto output_shape = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{2});
        output_shape->set_friendly_name("output_shape");
        auto scales = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{2});
        scales->set_friendly_name("scales");

        Interpolate::InterpolateAttrs attrs{};
        attrs.antialias = false;
        attrs.pads_begin = {0, 0, 0, 0};
        attrs.pads_end = {0, 0, 0, 0};
        attrs.cube_coeff = -0.75;
        auto interpolate = std::make_shared<ov::op::TypeRelaxed<Interpolate>>(data, output_shape, scales, attrs);

        auto result = std::make_shared<Result>(interpolate);
        result->set_friendly_name("result");
        model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                            ov::ParameterVector{data, output_shape, scales},
                                            "Interpolate");
        ov::util::set_tensors_names(ov::AUTO, *model);
    }

    auto execNet = core->compile_model(model, target_device);
    std::stringstream strm;
    execNet.export_model(strm);

    auto importedCompiledModel = core->import_model(strm, target_device);
    EXPECT_EQ(model->inputs().size(), 3);
    EXPECT_EQ(model->inputs().size(), importedCompiledModel.inputs().size());
    EXPECT_NO_THROW(importedCompiledModel.input("data").get_node());
    EXPECT_THROW(importedCompiledModel.input("param"), ov::Exception);

    EXPECT_EQ(model->outputs().size(), 1);
    EXPECT_EQ(model->outputs().size(), importedCompiledModel.outputs().size());
    EXPECT_NO_THROW(importedCompiledModel.output());
    EXPECT_EQ(model->output(0).get_tensor().get_names(), importedCompiledModel.output(0).get_tensor().get_names());
    EXPECT_NO_THROW(importedCompiledModel.output("result").get_node());
    EXPECT_THROW(importedCompiledModel.output("param"), ov::Exception);

    EXPECT_EQ(elementType, importedCompiledModel.input("data").get_element_type());
    EXPECT_EQ(elementType, importedCompiledModel.output("result").get_element_type());
}

INSTANTIATE_TEST_SUITE_P(smoke_serialization,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(empty_property)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

TEST_P(OVCompiledModelGraphUniqueNodeNamesTest, CheckUniqueNodeNames) {
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(model, target_device);
    auto exec_graph = compiled_model.get_runtime_model();

    int numReorders = 0;
    int expectedReorders = 2;
    std::unordered_set<std::string> names;
    ASSERT_NE(exec_graph, nullptr);

    for (const auto& op : exec_graph->get_ops()) {
        const auto& rtInfo = op->get_rt_info();
        auto it = rtInfo.find(ov::exec_model_info::LAYER_TYPE);
        ASSERT_NE(rtInfo.end(), it);
        auto opType = it->second.as<std::string>();

        if (opType == "Reorder") {
            numReorders++;
        }
    }

    ASSERT_EQ(numReorders, expectedReorders)
        << "Expected reorders: " << expectedReorders << ", actual reorders: " << numReorders;
};

const std::vector<ov::element::Type> netPrc = {
    ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape,
                         OVCompiledModelGraphUniqueNodeNamesTest,
                         ::testing::Combine(::testing::ValuesIn(netPrc),
                                            ::testing::Values(ov::Shape{1, 2, 5, 5}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         OVCompiledModelGraphUniqueNodeNamesTest::getTestCaseName);

}  // namespace
