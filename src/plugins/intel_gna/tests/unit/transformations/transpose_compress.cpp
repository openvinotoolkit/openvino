// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_compress.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"

using namespace ov::opset12;

namespace transpose_compress_test {

struct TestData {
    ov::Shape shape_src;
    ov::AxisVector tr_order_src;
    ov::Shape shape_ref;
    ov::AxisVector tr_order_ref;
};

typedef std::tuple<TestData>  // Transpose order
    test_params;

class TransposeCompressTest : public ov::test::TestsCommon, public ::testing::WithParamInterface<test_params> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<test_params>& obj) {
        TestData test_shapes;
        std::tie(test_shapes) = obj.param;

        std::ostringstream result;
        result << "InputShape=" << test_shapes.shape_src << "_";
        result << "TransposeOrder=" << test_shapes.tr_order_src << "_";

        return result.str();
    }

    virtual void set_test_model() {
        auto param = std::make_shared<Parameter>(m_type, m_test_shapes.shape_src);

        auto transpose_const = std::make_shared<Constant>(ov::element::i32,
                                                          ov::Shape{m_test_shapes.tr_order_src.size()},
                                                          m_test_shapes.tr_order_src);
        auto transpose = std::make_shared<Transpose>(param, transpose_const);

        m_shape_out = transpose->get_output_shape(0);

        auto result = std::make_shared<Result>(transpose);
        m_model_test =
            std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "transpose_compress");
    }

    virtual void set_ref_model() {
        auto param = std::make_shared<Parameter>(m_type, m_test_shapes.shape_src);

        auto shape_in_const = std::make_shared<Constant>(ov::element::i32,
                                                         ov::Shape{m_test_shapes.shape_ref.size()},
                                                         m_test_shapes.shape_ref);
        auto shape_in = std::make_shared<Reshape>(param, shape_in_const, false);

        auto transpose_const = std::make_shared<Constant>(ov::element::i8,
                                                          ov::Shape{m_test_shapes.tr_order_ref.size()},
                                                          m_test_shapes.tr_order_ref);
        auto transpose = std::make_shared<Transpose>(shape_in, transpose_const);

        auto shape_out_const = std::make_shared<Constant>(ov::element::i32, ov::Shape{m_shape_out.size()}, m_shape_out);
        auto shape_out = std::make_shared<Reshape>(transpose, shape_out_const, false);

        auto result = std::make_shared<Result>(shape_out);
        m_model_ref =
            std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "transpose_compress");
    }

    void SetUp() override {
        std::tie(m_test_shapes) = this->GetParam();
        set_test_model();
        set_ref_model();
    };

    void Validate() {
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::TransposeCompress>();
        m.run_passes(m_model_test);

        check_rt_info(m_model_test);

        const FunctionsComparator func_comparator =
            FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
        const FunctionsComparator::Result result = func_comparator(m_model_test, m_model_ref);

        ASSERT_TRUE(result.valid) << result.message;
    }

    void Run() {
        SetUp();
        Validate();
    }

public:
    TestData m_test_shapes;
    ov::Shape m_shape_out;
    ov::element::Type m_type = ov::element::f32;
    std::shared_ptr<ov::Model> m_model_test;
    std::shared_ptr<ov::Model> m_model_ref;
};

class TransposeCompressNegTest : public TransposeCompressTest {
    void set_ref_model() override {
        m_model_ref = m_model_test->clone();
    }
};

TEST_P(TransposeCompressTest, CompareWithRefs) {
    Run();
}

TEST_P(TransposeCompressNegTest, CompareWithRefs) {
    Run();
}

const std::vector<TestData> test_shapes = {{{1, 2, 3}, {1, 2, 0}, {1, 6}, {1, 0}},
                                           {{1, 2, 4}, {1, 2, 0}, {1, 8}, {1, 0}},
                                           {{2, 2, 4}, {1, 2, 0}, {2, 8}, {1, 0}},
                                           {{2, 2, 4, 4}, {2, 3, 0, 1}, {4, 16}, {1, 0}}};

const std::vector<TestData> test_neg_shapes = {{{1, 2, 3, 4}, {1, 0, 2, 3}, {}, {}},
                                               {{1, 2, 3, 4}, {0, 2, 1, 3}, {}, {}},
                                               {{1, 2, 3, 4}, {2, 3, 0, 1}, {}, {}},
                                               {{10, 20}, {1, 0}, {}, {}}};

INSTANTIATE_TEST_SUITE_P(smoke_transpose_compress_test,
                         TransposeCompressTest,
                         ::testing::Combine(::testing::ValuesIn(test_shapes)),
                         TransposeCompressTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_transpose_compress_test,
                         TransposeCompressNegTest,
                         ::testing::Combine(::testing::ValuesIn(test_neg_shapes)),
                         TransposeCompressNegTest::getTestCaseName);

}  // namespace transpose_compress_test
