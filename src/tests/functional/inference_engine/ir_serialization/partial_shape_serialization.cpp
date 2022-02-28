// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <gtest/gtest.h>

#include <common_test_utils/file_utils.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/opsets/opset8.hpp>
#include <transformations/rt_info/attributes.hpp>
#include "openvino/frontend/manager.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph/ngraph.hpp"
#include "transformations/serialize.hpp"

class PartialShapeSerializationTest : public CommonTestUtils::TestsCommon {
protected:
    std::string test_name = GetTestName() + "_" + GetTimestamp();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        CommonTestUtils::removeIRFiles(m_out_xml_path, m_out_bin_path);
    }

    std::shared_ptr<ngraph::Function> getWithIRFrontend(const std::string& model_path,
                                                        const std::string& weights_path) {
        ov::frontend::FrontEnd::Ptr FE;
        ov::frontend::InputModel::Ptr inputModel;

        ov::AnyVector params{model_path, weights_path};

        FE = manager.load_by_model(params);
        if (FE)
            inputModel = FE->load(params);

        if (inputModel)
            return FE->convert(inputModel);

        return nullptr;
    }

private:
    ov::frontend::FrontEndManager manager;
};

TEST_F(PartialShapeSerializationTest, pshape_serialize) {
    auto check_shape = [](const ov::PartialShape& shape1, ov::PartialShape shape2) {
        ASSERT_EQ(shape1.rank().is_dynamic(), shape2.rank().is_dynamic());
        if (shape1.rank().is_dynamic())
            return;
        ASSERT_EQ(shape1.size(), shape2.size());
        for (auto i = 0; i < shape1.size(); i++) {
            auto dim1 = shape1[i];
            auto dim2 = shape2[i];
            ASSERT_EQ(dim1.is_dynamic(), dim2.is_dynamic());
            ASSERT_EQ(dim1.get_min_length(), dim2.get_min_length());
            ASSERT_EQ(dim1.get_max_length(), dim2.get_max_length());
        }
    };

    std::shared_ptr<ov::Model> function;
    {
        auto pshape = ov::PartialShape{-1, ov::Dimension(-1, 20), ov::Dimension(10, -1), ov::Dimension(2, 100)};
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, pshape);
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        auto result = std::make_shared<ov::opset8::Result>(add);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data});
    }

    ov::pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path);
    m.run_passes(function);

    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto pshape_ref = ov::PartialShape{-1, ov::Dimension(-1, 20), ov::Dimension(10, -1), ov::Dimension(2, 100)};
    check_shape(f->get_parameters()[0]->get_partial_shape(), pshape_ref);
}

TEST_F(PartialShapeSerializationTest, pshape_serialize_dymamic_rank) {
    auto check_shape = [](const ov::PartialShape& shape1, ov::PartialShape shape2) {
        ASSERT_EQ(shape1.rank().is_dynamic(), shape2.rank().is_dynamic());
        if (shape1.rank().is_dynamic())
            return;
        ASSERT_EQ(shape1.size(), shape2.size());
        for (auto i = 0; i < shape1.size(); i++) {
            auto dim1 = shape1[i];
            auto dim2 = shape2[i];
            ASSERT_EQ(dim1.is_dynamic(), dim2.is_dynamic());
            ASSERT_EQ(dim1.get_min_length(), dim2.get_min_length());
            ASSERT_EQ(dim1.get_max_length(), dim2.get_max_length());
        }
    };

    std::shared_ptr<ov::Model> function;
    {
        auto pshape = ov::PartialShape::dynamic();
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, pshape);
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        auto result = std::make_shared<ov::opset8::Result>(add);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data});
    }

    ov::pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path);
    m.run_passes(function);

    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto pshape_ref = ov::PartialShape::dynamic();
    check_shape(f->get_parameters()[0]->get_partial_shape(), pshape_ref);
}
