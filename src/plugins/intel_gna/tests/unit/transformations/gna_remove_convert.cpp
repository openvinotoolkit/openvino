// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "transformations/remove_converts.hpp"

namespace testing {

typedef std::tuple<ov::element::Type,  // Net precision
                   ov::element::Type   // Convert precision
                   >
    removeConvertTestParams;

class RemoveInputConvertTest : public ov::test::TestsCommon,
                               public ::testing::WithParamInterface<removeConvertTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<removeConvertTestParams>& obj) {
        ov::element::Type net_precision, target_precision;
        std::tie(net_precision, target_precision) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << net_precision << "_";
        result << "trgPRC=" << target_precision << "_";

        return result.str();
    }
    void SetUp() override;
    virtual void Validate();
    virtual void Run();

public:
    std::shared_ptr<ngraph::Function> func_, ref_func_no_convert_, ref_func_convert_;
    ov::element::Type net_precision_, target_precision_;
};

void RemoveInputConvertTest::Run() {
    SetUp();
    Validate();
}

void RemoveInputConvertTest::SetUp() {
    const ngraph::Shape input_shape{10};

    std::tie(net_precision_, target_precision_) = this->GetParam();

    // test function
    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(target_precision_, input_shape);
        auto conversion =
            ngraph::builder::makeConversion(params, net_precision_, ngraph::helpers::ConversionTypes::CONVERT);
        auto add_const = ngraph::opset8::Constant::create(net_precision_, input_shape, {10});
        auto add = std::make_shared<ngraph::opset8::Add>(conversion, add_const);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        func_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                   ngraph::ParameterVector{params},
                                                   "Conversion");
    }

    // ref function convert should be removed
    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(net_precision_, input_shape);
        auto add_const = ngraph::opset8::Constant::create(net_precision_, input_shape, {10});
        auto add = std::make_shared<ngraph::opset8::Add>(params, add_const);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        ref_func_no_convert_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                                  ngraph::ParameterVector{params},
                                                                  "Conversion");
    }

    // ref function convert should not be removed
    ref_func_convert_ = ngraph::clone_function(*func_);
}

void RemoveInputConvertTest::Validate() {
    ngraph::pass::Manager m;
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gna::pass::RemoveInputConvert>();
    m.run_passes(func_);
    ASSERT_NO_THROW(check_rt_info(func_));

    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    FunctionsComparator::Result result;
    if (std::count(ov::intel_gna::pass::kSupportedInputConverts.begin(),
                   ov::intel_gna::pass::kSupportedInputConverts.end(),
                   std::make_pair(target_precision_, net_precision_)) == 0) {
        result = func_comparator(func_, ref_func_convert_);
    } else {
        result = func_comparator(func_, ref_func_no_convert_);
    }

    ASSERT_TRUE(result.valid);
}

class RemoveOutputConvertTest : public RemoveInputConvertTest {
public:
    void SetUp() override {
        const ngraph::Shape input_shape{10};

        std::tie(net_precision_, target_precision_) = this->GetParam();

        // test function
        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(net_precision_, input_shape);
            auto add_const = ngraph::opset8::Constant::create(net_precision_, input_shape, {10});
            auto add = std::make_shared<ngraph::opset8::Add>(params, add_const);
            auto conversion =
                ngraph::builder::makeConversion(add, target_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto result = std::make_shared<ngraph::opset8::Result>(conversion);
            func_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                       ngraph::ParameterVector{params},
                                                       "Conversion");
        }

        // ref function
        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(net_precision_, input_shape);
            auto add_const = ngraph::opset8::Constant::create(net_precision_, input_shape, {10});
            auto add = std::make_shared<ngraph::opset8::Add>(params, add_const);

            auto result = std::make_shared<ngraph::opset8::Result>(add);
            ref_func_no_convert_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                                      ngraph::ParameterVector{params},
                                                                      "Conversion");
        }

        // ref function convert should not be removed
        ref_func_convert_ = ngraph::clone_function(*func_);
    }
    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::RemoveOutputConvert>();
        m.run_passes(func_);
        ASSERT_NO_THROW(check_rt_info(func_));

        const FunctionsComparator func_comparator =
            FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
        FunctionsComparator::Result result;
        if (std::count(ov::intel_gna::pass::kSupportedOutputConverts.begin(),
                       ov::intel_gna::pass::kSupportedOutputConverts.end(),
                       std::make_pair(net_precision_, target_precision_)) == 0) {
            result = func_comparator(func_, ref_func_convert_);
        } else {
            result = func_comparator(func_, ref_func_no_convert_);
        }

        ASSERT_TRUE(result.valid);
    }
};

class LeaveConvertTest : public RemoveInputConvertTest {
public:
    void SetUp() override {
        const ngraph::Shape input_shape{10};

        std::tie(net_precision_, target_precision_) = this->GetParam();

        // test function
        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(net_precision_, input_shape);
            auto add_const = ngraph::opset8::Constant::create(net_precision_, input_shape, {10});
            auto add1 = std::make_shared<ngraph::opset8::Add>(params, add_const);
            auto conversion =
                ngraph::builder::makeConversion(add1, net_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto add2 = std::make_shared<ngraph::opset8::Add>(conversion, add_const);
            auto result = std::make_shared<ngraph::opset8::Result>(add2);
            func_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                       ngraph::ParameterVector{params},
                                                       "Conversion");
        }

        // ref function
        ref_func_convert_ = ngraph::clone_function(*func_);
    }
    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::intel_gna::pass::RemoveInputConvert>();
        m.register_pass<ov::intel_gna::pass::RemoveOutputConvert>();
        m.run_passes(func_);
        ASSERT_NO_THROW(check_rt_info(func_));

        const FunctionsComparator func_comparator =
            FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
        const FunctionsComparator::Result result = func_comparator(func_, ref_func_convert_);
        ASSERT_TRUE(result.valid);
    }
};

class RemoveMultiInputsConvertTest : public RemoveInputConvertTest {
public:
    void SetUp() override {
        std::tie(net_precision_, target_precision_) = this->GetParam();
        const ngraph::Shape input_shape{1, 10};

        // test function
        {
            ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(target_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(target_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(target_precision_, input_shape)};
            auto convert1 =
                ngraph::builder::makeConversion(input[0], net_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto convert2 =
                ngraph::builder::makeConversion(input[1], net_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto convert3 =
                ngraph::builder::makeConversion(input[2], net_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto mul1 = ngraph::builder::makeEltwise(convert1, convert2, ngraph::helpers::EltwiseTypes::ADD);
            auto mul2 = ngraph::builder::makeEltwise(convert3, mul1, ngraph::helpers::EltwiseTypes::ADD);
            auto result = std::make_shared<ngraph::opset8::Result>(mul2);
            func_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "multiple_input");
        }

        // ref function
        {
            ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape)};
            auto mul1 = ngraph::builder::makeEltwise(input[0], input[1], ngraph::helpers::EltwiseTypes::ADD);
            auto mul2 = ngraph::builder::makeEltwise(input[2], mul1, ngraph::helpers::EltwiseTypes::ADD);
            auto result = std::make_shared<ngraph::opset8::Result>(mul2);
            ref_func_no_convert_ =
                std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "multiple_input");
        }

        // ref function convert should not be removed
        ref_func_convert_ = ngraph::clone_function(*func_);
    }
};

class RemoveMultiOutputsConvertTest : public RemoveOutputConvertTest {
public:
    void SetUp() override {
        std::tie(net_precision_, target_precision_) = this->GetParam();
        const ngraph::Shape input_shape{1, 10};
        // test function
        {
            ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape)};
            auto mul1 = ngraph::builder::makeEltwise(input[0], input[1], ngraph::helpers::EltwiseTypes::ADD);
            auto mul2 = ngraph::builder::makeEltwise(input[2], input[3], ngraph::helpers::EltwiseTypes::ADD);
            auto convert1 =
                ngraph::builder::makeConversion(mul1, target_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto convert2 =
                ngraph::builder::makeConversion(mul2, target_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto result1 = std::make_shared<ngraph::opset8::Result>(convert1);
            auto result2 = std::make_shared<ngraph::opset8::Result>(convert2);

            func_ =
                std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2}, input, "multiple_output");
        }

        // ref function
        {
            ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape)};
            auto mul1 = ngraph::builder::makeEltwise(input[0], input[1], ngraph::helpers::EltwiseTypes::ADD);
            auto mul2 = ngraph::builder::makeEltwise(input[2], input[3], ngraph::helpers::EltwiseTypes::ADD);
            auto result1 = std::make_shared<ngraph::opset8::Result>(mul1);
            auto result2 = std::make_shared<ngraph::opset8::Result>(mul2);

            ref_func_no_convert_ =
                std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2}, input, "multiple_output");
        }

        // ref function convert should not be removed
        ref_func_convert_ = ngraph::clone_function(*func_);
    }
};

class RemoveOutputConvertConnectedToLayerTest : public RemoveOutputConvertTest {
public:
    void SetUp() override {
        std::tie(net_precision_, target_precision_) = this->GetParam();
        const ngraph::Shape input_shape{1, 10};
        // test function
        {
            ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape)};
            auto mul1 = ngraph::builder::makeEltwise(input[0], input[1], ngraph::helpers::EltwiseTypes::ADD);
            auto mul2 = ngraph::builder::makeEltwise(input[2], input[3], ngraph::helpers::EltwiseTypes::ADD);
            auto mul3 = ngraph::builder::makeEltwise(mul1, mul2, ngraph::helpers::EltwiseTypes::ADD);
            auto convert1 =
                ngraph::builder::makeConversion(mul1, target_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto convert2 =
                ngraph::builder::makeConversion(mul2, target_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto convert3 =
                ngraph::builder::makeConversion(mul3, target_precision_, ngraph::helpers::ConversionTypes::CONVERT);
            auto result1 = std::make_shared<ngraph::opset8::Result>(convert1);
            auto result2 = std::make_shared<ngraph::opset8::Result>(convert2);
            auto result3 = std::make_shared<ngraph::opset8::Result>(convert3);

            func_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                                       input,
                                                       "multiple_output");
        }

        // ref function
        {
            ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape),
                                      std::make_shared<ov::op::v0::Parameter>(net_precision_, input_shape)};
            auto mul1 = ngraph::builder::makeEltwise(input[0], input[1], ngraph::helpers::EltwiseTypes::ADD);
            auto mul2 = ngraph::builder::makeEltwise(input[2], input[3], ngraph::helpers::EltwiseTypes::ADD);
            auto mul3 = ngraph::builder::makeEltwise(mul1, mul2, ngraph::helpers::EltwiseTypes::ADD);
            auto result1 = std::make_shared<ngraph::opset8::Result>(mul1);
            auto result2 = std::make_shared<ngraph::opset8::Result>(mul2);
            auto result3 = std::make_shared<ngraph::opset8::Result>(mul3);

            ref_func_no_convert_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2, result3},
                                                                      input,
                                                                      "multiple_output");
        }

        // ref function convert should not be removed
        ref_func_convert_ = ngraph::clone_function(*func_);
    }
};

ov::element::TypeVector netTypes = {ov::element::f16,
                                    ov::element::f32,
                                    ov::element::i8,
                                    ov::element::u8,
                                    ov::element::i16,
                                    ov::element::i32,
                                    ov::element::i64};

ov::element::TypeVector targetTypes = {ov::element::f16,
                                       ov::element::f32,
                                       ov::element::i8,
                                       ov::element::u8,
                                       ov::element::i16,
                                       ov::element::i32,
                                       ov::element::i64};

TEST_P(RemoveInputConvertTest, CompareWithRefs) {
    Run();
}

TEST_P(RemoveOutputConvertTest, CompareWithRefs) {
    Run();
}

TEST_P(LeaveConvertTest, CompareWithRefs) {
    Run();
}

TEST_P(RemoveMultiInputsConvertTest, CompareWithRefs) {
    Run();
}

TEST_P(RemoveMultiOutputsConvertTest, CompareWithRefs) {
    Run();
}

TEST_P(RemoveOutputConvertConnectedToLayerTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         RemoveInputConvertTest,
                         ::testing::Combine(::testing::ValuesIn(netTypes), ::testing::ValuesIn(targetTypes)),
                         RemoveInputConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         RemoveOutputConvertTest,
                         ::testing::Combine(::testing::ValuesIn(netTypes), ::testing::ValuesIn(targetTypes)),
                         RemoveOutputConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         LeaveConvertTest,
                         ::testing::Combine(::testing::ValuesIn(netTypes), ::testing::ValuesIn(targetTypes)),
                         LeaveConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         RemoveMultiInputsConvertTest,
                         ::testing::Combine(::testing::ValuesIn(netTypes), ::testing::ValuesIn(targetTypes)),
                         RemoveMultiInputsConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         RemoveMultiOutputsConvertTest,
                         ::testing::Combine(::testing::ValuesIn(netTypes), ::testing::ValuesIn(targetTypes)),
                         RemoveMultiOutputsConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         RemoveOutputConvertConnectedToLayerTest,
                         ::testing::Combine(::testing::ValuesIn(netTypes), ::testing::ValuesIn(targetTypes)),
                         RemoveOutputConvertConnectedToLayerTest::getTestCaseName);
}  // namespace testing
