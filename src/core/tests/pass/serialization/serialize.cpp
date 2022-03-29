// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/serialize.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/graph_comparator.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "read_ir.hpp"
#include "util/test_common.hpp"

using SerializationParams = std::tuple<std::string, std::string>;

class SerializationTest : public ov::test::TestsCommon, public testing::WithParamInterface<SerializationParams> {
public:
    std::string m_model_path;
    std::string m_binary_path;
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    void CompareSerialized(std::function<void(const std::shared_ptr<ov::Model>&)> serializer) {
        auto expected = ov::test::readModel(m_model_path, m_binary_path);
        auto orig = ov::clone_model(*expected);
        serializer(expected);
        auto result = ov::test::readModel(m_out_xml_path, m_out_bin_path);
        const auto fc = FunctionsComparator::with_default()
                            .enable(FunctionsComparator::ATTRIBUTES)
                            .enable(FunctionsComparator::CONST_VALUES);
        const auto res = fc.compare(result, expected);
        const auto res2 = fc.compare(expected, orig);
        EXPECT_TRUE(res.valid) << res.message;
        EXPECT_TRUE(res2.valid) << res2.message;
    }

    void SetUp() override {
        m_model_path = ov::util::path_join({SERIALIZED_ZOO, "ir/", std::get<0>(GetParam())});
        if (!std::get<1>(GetParam()).empty()) {
            m_binary_path = ov::util::path_join({SERIALIZED_ZOO, "ir/", std::get<1>(GetParam())});
        }

        const std::string test_name = GetTestName() + "_" + GetTimestamp();
        m_out_xml_path = test_name + ".xml";
        m_out_bin_path = test_name + ".bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
    }
};

TEST_P(SerializationTest, CompareFunctions) {
    CompareSerialized([this](const std::shared_ptr<ov::Model>& m) {
        ov::pass::Serialize(m_out_xml_path, m_out_bin_path).run_on_model(m);
    });
}

TEST_P(SerializationTest, SerializeHelper) {
    CompareSerialized([this](const std::shared_ptr<ov::Model>& m) {
        ov::serialize(m, m_out_xml_path, m_out_bin_path);
    });
}

INSTANTIATE_TEST_SUITE_P(
    IRSerialization,
    SerializationTest,
    testing::Values(std::make_tuple("add_abc.xml", "add_abc.bin"),
                    std::make_tuple("add_abc_f64.xml", ""),
                    std::make_tuple("add_abc_bin.xml", ""),
                    std::make_tuple("split_equal_parts_2d.xml", "split_equal_parts_2d.bin"),
                    std::make_tuple("addmul_abc.xml", "addmul_abc.bin"),
                    std::make_tuple("add_abc_initializers.xml", "add_abc_initializers.bin"),
                    std::make_tuple("add_abc_initializers.xml", "add_abc_initializers_f32_nan_const.bin"),
                    std::make_tuple("add_abc_initializers_nan_const.xml", "add_abc_initializers_nan_const.bin"),
                    std::make_tuple("add_abc_initializers_u1_const.xml", "add_abc_initializers_u1_const.bin"),
                    std::make_tuple("experimental_detectron_roi_feature_extractor.xml", ""),
                    std::make_tuple("experimental_detectron_roi_feature_extractor_opset6.xml", ""),
                    std::make_tuple("experimental_detectron_detection_output.xml", ""),
                    std::make_tuple("experimental_detectron_detection_output_opset6.xml", ""),
                    std::make_tuple("nms5.xml", "nms5.bin"),
                    std::make_tuple("shape_of.xml", ""),
                    std::make_tuple("dynamic_input_shape.xml", ""),
                    std::make_tuple("pad_with_shape_of.xml", ""),
                    std::make_tuple("conv_with_rt_info.xml", ""),
                    std::make_tuple("loop_2d_add.xml", "loop_2d_add.bin"),
                    std::make_tuple("nms5_dynamism.xml", "nms5_dynamism.bin"),
                    std::make_tuple("if_diff_case.xml", "if_diff_case.bin"),
                    std::make_tuple("if_body_without_parameters.xml", "if_body_without_parameters.bin")));

#ifdef ENABLE_OV_ONNX_FRONTEND

INSTANTIATE_TEST_SUITE_P(ONNXSerialization,
                         SerializationTest,
                         testing::Values(std::make_tuple("add_abc.onnx", ""),
                                         std::make_tuple("split_equal_parts_2d.onnx", ""),
                                         std::make_tuple("addmul_abc.onnx", ""),
                                         std::make_tuple("add_abc_initializers.onnx", "")));

#endif
