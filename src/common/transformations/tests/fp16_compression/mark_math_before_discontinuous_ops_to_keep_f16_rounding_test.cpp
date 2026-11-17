// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/fp16_compression/mark_math_before_discontinuous_ops_to_keep_f16_rounding.hpp"

#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/acos.hpp"
#include "openvino/op/acosh.hpp"
#include "openvino/op/asin.hpp"
#include "openvino/op/asinh.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/atanh.hpp"
#include "openvino/op/ceiling.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/cosh.hpp"
#include "openvino/op/floor.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sinh.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/softsign.hpp"
#include "openvino/op/tan.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace {

using NodeBuilder = std::function<std::shared_ptr<ov::Node>(const ov::Output<ov::Node>&)>;

struct ConsumerCase {
    std::string name;
    NodeBuilder build;
};

template <class Op>
NodeBuilder unary() {
    return [](const ov::Output<ov::Node>& in) {
        return std::make_shared<Op>(in);
    };
}

template <class Op>
NodeBuilder with_two_constants(float first, float second) {
    return [=](const ov::Output<ov::Node>& in) {
        return std::make_shared<Op>(in,
                                    ov::op::v0::Constant::create(ov::element::f16, {}, {first}),
                                    ov::op::v0::Constant::create(ov::element::f16, {}, {second}));
    };
}

NodeBuilder round(ov::op::v5::Round::RoundMode mode) {
    return [=](const ov::Output<ov::Node>& in) {
        return std::make_shared<ov::op::v5::Round>(in, mode);
    };
}

// Math output is the first input, the second one is a constant.
template <class Op>
NodeBuilder math_first() {
    return [](const ov::Output<ov::Node>& in) {
        return std::make_shared<Op>(in, ov::op::v0::Constant::create(ov::element::f16, {}, {2.0f}));
    };
}

// Math output is the second input, the first one is a constant.
template <class Op>
NodeBuilder math_second() {
    return [](const ov::Output<ov::Node>& in) {
        return std::make_shared<Op>(ov::op::v0::Constant::create(ov::element::f16, {}, {2.0f}), in);
    };
}

const std::vector<NodeBuilder> math_builders = {
    unary<ov::op::v0::Cos>(),
    unary<ov::op::v0::Cosh>(),
    unary<ov::op::v0::Sin>(),
    unary<ov::op::v0::Sinh>(),
    unary<ov::op::v0::Acos>(),
    unary<ov::op::v3::Acosh>(),
    unary<ov::op::v0::Asin>(),
    unary<ov::op::v3::Asinh>(),
    unary<ov::op::v0::Atan>(),
    unary<ov::op::v3::Atanh>(),
    unary<ov::op::v0::Tan>(),
    unary<ov::op::v0::Sign>(),
    unary<ov::op::v4::SoftPlus>(),
    unary<ov::op::v9::SoftSign>(),
    with_two_constants<ov::op::v0::Selu>(1.67326f, 1.0507f),
    with_two_constants<ov::op::v0::HardSigmoid>(0.2f, 0.5f),
};

const std::vector<ConsumerCase> consumer_cases = {
    {"Floor", unary<ov::op::v0::Floor>()},
    {"Ceiling", unary<ov::op::v0::Ceiling>()},
    {"RoundHalfToEven", round(ov::op::v5::Round::RoundMode::HALF_TO_EVEN)},
    {"RoundHalfAwayFromZero", round(ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO)},
    {"Sign", unary<ov::op::v0::Sign>()},
    {"ModFirstInput", math_first<ov::op::v1::Mod>()},
    {"ModSecondInput", math_second<ov::op::v1::Mod>()},
    {"FloorModFirstInput", math_first<ov::op::v1::FloorMod>()},
    {"FloorModSecondInput", math_second<ov::op::v1::FloorMod>()},
};

std::string math_name(const NodeBuilder& build_math) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    return build_math(input)->get_type_name();
}

// Builds Parameter -> Producer -> Consumer; the Producer node is marked when `mark_producer` is set.
std::shared_ptr<ov::Model> make_model(const NodeBuilder& build_producer,
                                      const NodeBuilder& build_consumer,
                                      bool mark_producer) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1});
    auto producer = build_producer(input);
    if (mark_producer) {
        ov::disable_conversion(producer, ov::element::f16, ov::element::f32);
    }
    auto consumer = build_consumer(producer);
    return std::make_shared<ov::Model>(ov::OutputVector{consumer}, ov::ParameterVector{input});
}

class MarkMathBeforeDiscontinuousOpsTestBase : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
        manager.register_pass<ov::pass::MarkMathBeforeDiscontinuousOpsToKeepF16Rounding>();
    }
};

using MathAndConsumer = std::tuple<NodeBuilder, ConsumerCase>;

class MathFeedingDiscontinuousOpTest : public MarkMathBeforeDiscontinuousOpsTestBase,
                                       public testing::WithParamInterface<MathAndConsumer> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<MathAndConsumer>& obj) {
        const auto& [build_math, consumer] = obj.param;
        return math_name(build_math) + "_" + consumer.name;
    }
};

TEST_P(MathFeedingDiscontinuousOpTest, MathIsMarked) {
    const auto& [build_math, consumer] = GetParam();
    model = make_model(build_math, consumer.build, false);
    model_ref = make_model(build_math, consumer.build, true);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         MathFeedingDiscontinuousOpTest,
                         testing::Combine(testing::ValuesIn(math_builders), testing::ValuesIn(consumer_cases)),
                         MathFeedingDiscontinuousOpTest::get_test_case_name);

class MathNotFeedingDiscontinuousOpTest : public MarkMathBeforeDiscontinuousOpsTestBase,
                                          public testing::WithParamInterface<NodeBuilder> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<NodeBuilder>& obj) {
        return math_name(obj.param);
    }
};

TEST_P(MathNotFeedingDiscontinuousOpTest, MathIsNotMarked) {
    model = make_model(GetParam(), unary<ov::op::v0::Relu>(), false);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         MathNotFeedingDiscontinuousOpTest,
                         testing::ValuesIn(math_builders),
                         MathNotFeedingDiscontinuousOpTest::get_test_case_name);

class NonMathFeedingDiscontinuousOpTest : public MarkMathBeforeDiscontinuousOpsTestBase,
                                          public testing::WithParamInterface<ConsumerCase> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<ConsumerCase>& obj) {
        return obj.param.name;
    }
};

TEST_P(NonMathFeedingDiscontinuousOpTest, ProducerIsNotMarked) {
    model = make_model(unary<ov::op::v0::Relu>(), GetParam().build, false);
}

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         NonMathFeedingDiscontinuousOpTest,
                         testing::ValuesIn(consumer_cases),
                         NonMathFeedingDiscontinuousOpTest::get_test_case_name);

}  // namespace
