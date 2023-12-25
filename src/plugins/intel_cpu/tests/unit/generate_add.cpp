// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>
#include <array>

#include <snippets/generator.hpp>
#include <snippets/op/subgraph.hpp>

#include <openvino/core/model.hpp>

#include <ov_models/utils/ov_helpers.hpp>

using namespace testing;

inline auto gen_inputs(const ov::Shape& shape, size_t n = 2) -> std::vector<std::vector<std::uint8_t>> {
    std::vector<std::vector<std::uint8_t>> referenceInputs(n);
    for (size_t k = 0; k < n; k++) {
        referenceInputs[k].resize(ov::shape_size(shape)*sizeof(float));
        float* in0 = reinterpret_cast<float*>(&referenceInputs[k][0]);

        for (size_t i = 0; i < ov::shape_size(shape); i++) {
            if (k % 3 == 0) {
                in0[i] = i / 2048.f;
            } else if (k % 3 == 1) {
                in0[i] = 1 - i / 2048.f;
            } else {
                in0[i] = i / 1024.f;
            }
        }
    }
    return referenceInputs;
}

inline auto compare(std::shared_ptr<ov::Model>& s, std::shared_ptr<ov::Model>& f, std::vector<std::vector<std::uint8_t>>& in) -> bool{
    auto act = ngraph::helpers::interpreterFunction(s, in);
    auto exp = ngraph::helpers::interpreterFunction(f, in);

    const float* pexp = reinterpret_cast<float*>(&exp[0].second[0]);
    const float* pact = reinterpret_cast<float*>(&act[0].second[0]);

    bool isCorrect = true;
    for (size_t i = 0; i < ov::shape_size(f->get_result()->get_shape()); i++) {
        if (std::abs(pexp[i]-pact[i]) > std::numeric_limits<float>::epsilon()
            || std::isnan(pexp[i]) != std::isnan(pact[i])) {
            isCorrect = false;
            std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << " diff " << std::abs(pexp[i]-pact[i]) << std::endl;
        }
    }
    return isCorrect;
}

inline auto wrapAsSnippet(std::shared_ptr<ov::Model>& f,
                          const ov::Shape& shape0,
                          const ov::Shape& shape1) -> std::shared_ptr<ov::Model> {
    auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape0);
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape1);
    auto snippet = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{input0, input1}, f->clone());
    return std::make_shared<ov::Model>(ov::NodeVector{snippet}, ov::ParameterVector{input0, input1});
}

inline auto wrapAsSnippet(std::shared_ptr<ov::Model>& f, const ov::Shape& shape0)
    -> std::shared_ptr<ov::Model> {
    auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape0);
    auto snippet = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{input0}, f->clone());
    return std::make_shared<ov::Model>(ov::NodeVector{snippet}, ov::ParameterVector{input0});
}

// Todo: Reimplement Snippets tests, so they won't require evaluate method for OP Subgraph
//  In more detail, we can't use interpreter backend for op::Subgraph evaluation, since it will
//  depend on the snippets lib in this (which is not allowed).
TEST(SnippetsTests, GenerateAddParams) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};

    auto f = ([] (const ov::Shape& shape) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto add    = std::make_shared<ov::opset1::Add>(input0, input1);

        return std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input0, input1});
    })(shape);

    auto s = wrapAsSnippet(f, shape, shape);
    auto referenceInputs = gen_inputs(shape, 2);
    bool isCorrect = compare(s, f, referenceInputs);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddConstant) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};

    auto f = ([] (const ov::Shape& shape) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);

        std::vector<float> vals(ov::shape_size(shape));
        for (size_t i = 0; i < ov::shape_size(shape); i++) {
            vals[i] = 1-i/2048.f;
        }
        auto input1 = std::make_shared<ov::opset1::Constant>(ov::element::f32, shape, vals);
        auto add    = std::make_shared<ov::opset1::Add>(input0, input1);

        return std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input0});
    })(shape);

    auto s = wrapAsSnippet(f, shape);
    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddConstantScalar) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};

    auto f = ([] (const ov::Shape& shape) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto input1 = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>({42.f}));
        auto add    = std::make_shared<ov::opset1::Add>(input0, input1);

        return std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input0});
    })(shape);

    auto s = wrapAsSnippet(f, shape);
    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddConstantScalarEmptySize) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};

    auto f = ([] (const ov::Shape& shape) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto input1 = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape(), std::vector<float>({42.f}));
        auto add    = std::make_shared<ov::opset1::Add>(input0, input1);

        return std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input0});
    })(shape);

    auto s = wrapAsSnippet(f, shape);
    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);
    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddBroadcastX2Edges) {
    GTEST_SKIP();
    auto shape0 = ov::Shape{1, 4, 16, 31};
    auto shape1 = ov::Shape{1, 4, 16, 1};

    auto f = ([] (const ov::Shape& shape0, const ov::Shape& shape1) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape0);
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape1);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape1);

        auto add    = std::make_shared<ov::opset1::Add>(input0, input1);
        auto mul    = std::make_shared<ov::opset1::Add>(input1, input2);

        auto sub = std::make_shared<ov::opset1::Add>(add, mul);

        return std::make_shared<ov::Model>(ov::NodeVector{sub}, ov::ParameterVector{input0, input1, input2});
    })(shape0, shape1);

    auto s = ([f] (const ov::Shape& shape0, const ov::Shape& shape1) -> std::shared_ptr<ov::Model>{
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape0);
        auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape1);
        auto input4 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape1);
        auto snippet = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{input2, input3, input4}, f->clone());
        return std::make_shared<ov::Model>(ov::NodeVector{snippet}, ov::ParameterVector{input2, input3, input4});
    })(shape0, shape1);

    std::vector<std::vector<std::uint8_t>> referenceInputs(3);
    referenceInputs[0].resize(ov::shape_size(shape0)*sizeof(float));
    referenceInputs[1].resize(ov::shape_size(shape1)*sizeof(float));
    referenceInputs[2].resize(ov::shape_size(shape1)*sizeof(float));

    float* in0 = reinterpret_cast<float*>(&referenceInputs[0][0]);
    float* in1 = reinterpret_cast<float*>(&referenceInputs[1][0]);
    float* in2 = reinterpret_cast<float*>(&referenceInputs[2][0]);
    for (size_t i = 0; i < ov::shape_size(shape0); i++) {
        in0[i] = i/2048.f;
    }

    in1[0] = 1.f;
    in2[0] = 0.42f;

    auto act = ngraph::helpers::interpreterFunction(s, referenceInputs);
    auto exp = ngraph::helpers::interpreterFunction(f, referenceInputs);

    const float* pexp = reinterpret_cast<float*>(&exp[0].second[0]);
    const float* pact = reinterpret_cast<float*>(&act[0].second[0]);

    bool isCorrect = true;
    for (size_t i = 0; i < ov::shape_size(shape0); i++) {
        if (pexp[i] != pact[i]) {
            isCorrect = false;
            std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << std::endl;
        }
    }

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddBroadcastY) {
    GTEST_SKIP();
    auto shape0 = ov::Shape{1, 4, 16, 31};
    auto shape1 = ov::Shape{1, 4,  1, 31};

    auto f = ([] (const ov::Shape& shape0, const ov::Shape& shape1) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape0);
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape1);
        auto add    = std::make_shared<ov::opset1::Add>(input0, input1);

        return std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input0, input1});
    })(shape0, shape1);

    auto s = wrapAsSnippet(f, shape0, shape1);

    std::vector<std::vector<std::uint8_t>> referenceInputs(2);
    referenceInputs[0].resize(ov::shape_size(shape0)*sizeof(float));
    referenceInputs[1].resize(ov::shape_size(shape1)*sizeof(float));

    float* in0 = reinterpret_cast<float*>(&referenceInputs[0][0]);
    float* in1 = reinterpret_cast<float*>(&referenceInputs[1][0]);
    for (size_t i = 0; i < ov::shape_size(shape0); i++) {
        in0[i] = i / 2048.f;
    }
    for (size_t i = 0; i < ov::shape_size(shape1); i++) {
        in1[i] = 1 - i / 2048.f;
    }

    auto act = ngraph::helpers::interpreterFunction(s, referenceInputs);
    auto exp = ngraph::helpers::interpreterFunction(f, referenceInputs);

    const float* pexp = reinterpret_cast<float*>(&exp[0].second[0]);
    const float* pact = reinterpret_cast<float*>(&act[0].second[0]);

    bool isCorrect = true;
    for (size_t i = 0; i < ov::shape_size(shape0); i++) {
        if (pexp[i] != pact[i]) {
            isCorrect = false;
            std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << std::endl;
        }
    }

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddNegate) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};

    auto f = ([] (const ov::Shape& shape) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto add    = std::make_shared<ov::opset1::Add>(input0, input1);
        auto nagate = std::make_shared<ov::opset1::Negative>(add);

        return std::make_shared<ov::Model>(ov::NodeVector{nagate}, ov::ParameterVector{input0, input1});
    })(shape);

    auto s = ([f] (const ov::Shape& shape) -> std::shared_ptr<ov::Model>{
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto snippet = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{input2, input3}, f->clone());
        return std::make_shared<ov::Model>(ov::NodeVector{snippet}, ov::ParameterVector{input2, input3});
    })(shape);

    auto referenceInputs = gen_inputs(shape, 2);
    bool isCorrect = compare(s, f, referenceInputs);

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddNegateAdd) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto add = std::make_shared<ov::opset1::Add>(input1, input2);
    auto nagate = std::make_shared<ov::opset1::Negative>(add);
    auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto add2 = std::make_shared<ov::opset1::Add>(nagate, input3);
    std::shared_ptr<ov::Model> f = std::make_shared<ov::Model>(ov::NodeVector{add2}, ov::ParameterVector{input1, input2, input3});

    auto input11 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto input21 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto input31 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto snippet = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{input11, input21, input31}, f->clone());
    std::shared_ptr<ov::Model>  s = std::make_shared<ov::Model>(ov::NodeVector{snippet}, ov::ParameterVector{input11, input21, input31});

    auto referenceInputs = gen_inputs(shape, 3);
    bool isCorrect = compare(s, f, referenceInputs);

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddNegateAddMultiEdge) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto add    = std::make_shared<ov::opset1::Add>(input1, input2);
    auto nagate = std::make_shared<ov::opset1::Negative>(add);
    auto add2 = std::make_shared<ov::opset1::Add>(nagate, input1);
    std::shared_ptr<ov::Model> f = std::make_shared<ov::Model>(ov::NodeVector{add2}, ov::ParameterVector{input1, input2});

    auto s = wrapAsSnippet(f, shape, shape);
    auto referenceInputs = gen_inputs(shape, 2);
    bool isCorrect = compare(s, f, referenceInputs);

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateAddNegateAddMultiEdgeConst) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto input2 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.42});
    auto add = std::make_shared<ov::opset1::Add>(input1, input2);
    auto nagate = std::make_shared<ov::opset1::Negative>(add);
    auto add2 = std::make_shared<ov::opset1::Add>(nagate, input1);
    std::shared_ptr<ov::Model> f = std::make_shared<ov::Model>(ov::NodeVector{add2}, ov::ParameterVector{input1});

    auto input11 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto snippet = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{input11}, f->clone());
    std::shared_ptr<ov::Model>  s = std::make_shared<ov::Model>(ov::NodeVector{snippet}, ov::ParameterVector{input11});

    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

TEST(SnippetsTests, GenerateErf) {
    GTEST_SKIP();
    auto shape = ov::Shape{1, 4, 16, 31};

    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto gelu   = std::make_shared<ov::opset1::Erf>(input1);
    std::shared_ptr<ov::Model> f = std::make_shared<ov::Model>(ov::NodeVector{gelu}, ov::ParameterVector{input1});

    auto input11 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
    auto snippet = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{input11}, f->clone());
    std::shared_ptr<ov::Model> s = std::make_shared<ov::Model>(ov::NodeVector{snippet}, ov::ParameterVector{input11});

    auto referenceInputs = gen_inputs(shape, 1);
    bool isCorrect = compare(s, f, referenceInputs);

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}

// ToDO: implement tile selection logic & broadcast emission to make it working. Broadcast substitution works
TEST(SnippetsTests, GenerateAddBroadcastAutomatic) {
    GTEST_SKIP();
    std::array<ov::Shape, 3> shapes {
        ov::Shape{1, 4, 16, 31},
        ov::Shape{1, 4, 16, 1},
        ov::Shape{1, 4, 16, 1}
    };

    auto f = ([] (const ov::Shape& shape0, const ov::Shape& shape1, const ov::Shape& shape2) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape0);
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape1);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape2);

        auto add = std::make_shared<ov::opset1::Add>(input0, input1);
        auto mul = std::make_shared<ov::opset1::Multiply>(input1, input2);
        auto sub = std::make_shared<ov::opset1::Subtract>(add, mul);

        return std::make_shared<ov::Model>(ov::NodeVector{sub}, ov::ParameterVector{input0, input1, input2});
    })(shapes[0], shapes[1], shapes[2]);

    auto s = ([f] (const ov::Shape& shape0, const ov::Shape& shape1, const ov::Shape& shape2) -> std::shared_ptr<ov::Model>{
        auto input0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape0);
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape1);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape2);
        auto snippet = std::make_shared<ov::snippets::op::Subgraph>(ov::OutputVector{input0, input1, input2}, f->clone());
        return std::make_shared<ov::Model>(ov::NodeVector{snippet}, ov::ParameterVector{input0, input1, input2});
    })(shapes[0], shapes[1], shapes[2]);

    std::vector<std::vector<std::uint8_t>> referenceInputs(3);
    for (size_t k = 0; k < referenceInputs.size(); k++) {
        referenceInputs[k].resize(ov::shape_size(shapes[k]) * sizeof(float));

        auto in0 = reinterpret_cast<float*>(&referenceInputs[k][0]);
        for (size_t i = 0; i < ov::shape_size(shapes[k]); i++) {
            in0[i] = k == 0 ? i/2048.f : (k == 1 ? 1.f : 0.42f);
        }
    }

    auto exp = ngraph::helpers::interpreterFunction(f, referenceInputs);
    auto act = ngraph::helpers::interpreterFunction(s, referenceInputs);

    const float* pexp = reinterpret_cast<float*>(&exp[0].second[0]);
    const float* pact = reinterpret_cast<float*>(&act[0].second[0]);

    bool isCorrect = true;
    for (size_t i = 0; i < ov::shape_size(shapes[0]); i++) {
        if (pexp[i] != pact[i]) {
            isCorrect = false;
            std::cout << i << " expected " << pexp[i] << " actual " << pact[i] << std::endl;
        }
    }

    ASSERT_TRUE(isCorrect) << "snippet and native implementation differs";
}
