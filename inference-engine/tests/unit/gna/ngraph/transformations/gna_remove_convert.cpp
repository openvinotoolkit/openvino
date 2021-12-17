// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include "ngraph_functions/builders.hpp"
#include <transformations/init_node_info.hpp>
#include "transformations/remove_converts.hpp"
#include "transformations/serialize.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"

namespace testing {

typedef std::tuple<
        ov::element::Type,     // Net precision
        ov::element::Type      // Convert precision
> removeConvertTestParams;

class RemoveConvertTest: public CommonTestUtils::TestsCommon,
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
    std::shared_ptr<ngraph::Function> function_, ref_function_;
};

void RemoveConvertTest::Run() {
    SetUp();
    Validate();
}

void RemoveConvertTest::SetUp() {
    ov::element::Type net_precision, target_precision;
    const ngraph::Shape input_shape{10};

    std::tie(net_precision, target_precision) = this->GetParam();

    // test function
    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(target_precision, input_shape);
        auto conversion = ngraph::builder::makeConversion(params, net_precision, ngraph::helpers::ConversionTypes::CONVERT);
        auto add_const = ngraph::opset8::Constant::create(net_precision, input_shape, {10});
        auto add = std::make_shared<ngraph::opset8::Add>(conversion, add_const);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        function_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                       ngraph::ParameterVector{params},
                                                       "Conversion");
    }

    // ref function
    {
        auto params = std::make_shared<ngraph::opset8::Parameter>(net_precision, input_shape);
        auto add_const = ngraph::opset8::Constant::create(net_precision, input_shape, {10});
        auto add = std::make_shared<ngraph::opset8::Add>(params, add_const);

        auto result = std::make_shared<ngraph::opset8::Result>(add);
        ref_function_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                           ngraph::ParameterVector{params},
                                                           "Conversion");
    }
}

void RemoveConvertTest::Validate() {
    ngraph::pass::Manager m;
    m.register_pass<ngraph::pass::InitNodeInfo>();
    m.register_pass<GNAPluginNS::RemoveInputConvert>();
    m.run_passes(function_);
    ASSERT_NO_THROW(check_rt_info(function_));

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function_, ref_function_);
    ASSERT_TRUE(result.valid);
}

class RemoveOutputConvertTest: public RemoveConvertTest {
public:
    void SetUp() override {
        ov::element::Type net_precision, target_precision;
        const ngraph::Shape input_shape{10};

        std::tie(net_precision, target_precision) = this->GetParam();

        // test function
        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(net_precision, input_shape);
            auto add_const = ngraph::opset8::Constant::create(net_precision, input_shape, {10});
            auto add = std::make_shared<ngraph::opset8::Add>(params, add_const);
            auto conversion = ngraph::builder::makeConversion(add, target_precision, ngraph::helpers::ConversionTypes::CONVERT);
            auto result = std::make_shared<ngraph::opset8::Result>(conversion);
            function_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Conversion");
        }

        // ref function
        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(net_precision, input_shape);
            auto add_const = ngraph::opset8::Constant::create(net_precision, input_shape, {10});
            auto add = std::make_shared<ngraph::opset8::Add>(params, add_const);

            auto result = std::make_shared<ngraph::opset8::Result>(add);
            ref_function_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{params},
                                                            "Conversion");
        }
    }
    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::RemoveOutputConvert>();
        m.run_passes(function_);
        ASSERT_NO_THROW(check_rt_info(function_));

        const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
        const FunctionsComparator::Result result = func_comparator(function_, ref_function_);
        ASSERT_TRUE(result.valid);
    }
};

class LeaveConvertTest: public RemoveConvertTest {
public:
    void SetUp() override {
        ov::element::Type net_precision, target_precision;
        const ngraph::Shape input_shape{10};

        std::tie(net_precision, target_precision) = this->GetParam();

        // test function
        {
            auto params = std::make_shared<ngraph::opset8::Parameter>(net_precision, input_shape);
            auto add_const = ngraph::opset8::Constant::create(net_precision, input_shape, {10});
            auto add1 = std::make_shared<ngraph::opset8::Add>(params, add_const);
            auto conversion = ngraph::builder::makeConversion(add1, net_precision, ngraph::helpers::ConversionTypes::CONVERT);
            auto add2 = std::make_shared<ngraph::opset8::Add>(conversion, add_const);
            auto result = std::make_shared<ngraph::opset8::Result>(add2);
            function_ = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ngraph::ParameterVector{params},
                                                        "Conversion");
        }

        // ref function
        ref_function_ = ngraph::clone_function(*function_);
    }
    void Validate() override {
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::RemoveInputConvert>();
        m.register_pass<GNAPluginNS::RemoveOutputConvert>();
        m.run_passes(function_);
        ASSERT_NO_THROW(check_rt_info(function_));

        const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
        const FunctionsComparator::Result result = func_comparator(function_, ref_function_);
        ASSERT_TRUE(result.valid);
    }
};


ov::element::TypeVector netTypes = {
    ov::element::f16,
    ov::element::f32,
    ov::element::i8,
    ov::element::u8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64
};

ov::element::TypeVector targetTypes = {
    ov::element::f16,
    ov::element::f32,
    ov::element::i8,
    ov::element::u8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64
};

TEST_P(RemoveConvertTest, CompareWithRefs) {
    Run();
}

TEST_P(RemoveOutputConvertTest, CompareWithRefs) {
    Run();
}

TEST_P(LeaveConvertTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(TransformationTests, RemoveConvertTest,
                         ::testing::Combine(
                                ::testing::ValuesIn(netTypes),
                                ::testing::ValuesIn(targetTypes)),
                         RemoveConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests, RemoveOutputConvertTest,
                         ::testing::Combine(
                                ::testing::ValuesIn(netTypes),
                                ::testing::ValuesIn(targetTypes)),
                         RemoveOutputConvertTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TransformationTests, LeaveConvertTest,
                         ::testing::Combine(
                                ::testing::ValuesIn(netTypes),
                                ::testing::ValuesIn(targetTypes)),
                         LeaveConvertTest::getTestCaseName);

} // namespace testing
