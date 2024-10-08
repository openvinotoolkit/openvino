// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_matmul.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/op/subgraph.hpp>
#include "common_test_utils/node_builders/constant.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {
std::shared_ptr<ov::Node> make_matmul_b_input(const ov::element::Type& precision,
                                              const ov::PartialShape& shape,
                                              MatMulType type,
                                              ov::ParameterVector& params) {
    std::shared_ptr<ov::Node> result;
    switch (type) {
        case MatMulType::FullyConnected:
            return ov::test::utils::make_constant(precision, shape.to_shape());
        case MatMulType::MatMul: {
            auto param = std::make_shared<op::v0::Parameter>(precision, shape);
            params.push_back(param);
            return param;
        }
        default:
            OPENVINO_THROW("Unexpected MatMulType is passed in make_matmul_b_input");
    }
}

std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<ov::Node>& in, bool signed_interval) {
    static const float i8_fq_il = -35.0172004;
    static const float i8_fq_ih = 34.7436294;
    static const float u8_fq_il = 0;
    static const float u8_fq_ih = 0.820726;
    const auto low = signed_interval ? i8_fq_il : u8_fq_il;
    const auto high = signed_interval ? i8_fq_ih : u8_fq_ih;
    return ov::test::utils::make_fake_quantize(in, ov::element::f32, 256, {1}, {low}, {high}, {low}, {high});
}
} // namespace

std::ostream &operator<<(std::ostream& os, MatMulType type) {
    switch (type) {
        case MatMulType::MatMul:
            return os << "MatMul";
        case MatMulType::FullyConnected:
            return os << "FullyConnected";
        default:
            OPENVINO_THROW("Unexpected MatMulType.");
    }
}

MatMulFunctionBase::MatMulFunctionBase(const std::vector<PartialShape>& inputShapes,
                                       MatMulType type,
                                       const std::vector<ov::element::Type>& precisions)
    : SnippetsFunctionBase(inputShapes),
      precisions(precisions),
      matmul_type(type) {
    if (!precisions.empty()) {
        OPENVINO_ASSERT(precisions.size() == 2, "Got invalid number of input element types");
        const bool is_f32 = ov::snippets::utils::everyone_is(element::f32, precisions[0], precisions[1]);
        const bool is_int8 = ov::snippets::utils::one_of(precisions[0], element::i8, element::u8) && precisions[1] == element::i8;
        const bool is_bf16 = ov::snippets::utils::everyone_is(element::bf16, precisions[0], precisions[1]);
        OPENVINO_ASSERT(is_f32 || is_bf16 || is_int8, "Invalid precisions");
    }
}

void MatMulFunctionBase::validate_function(const std::shared_ptr<Model> &f) const {
    OPENVINO_ASSERT(f != nullptr, "The test requires Model to be defined");
    const auto count_of_shapes = input_shapes.size();
    const auto idces_to_remove = get_constant_input_idces();
    OPENVINO_ASSERT(std::all_of(idces_to_remove.begin(), idces_to_remove.end(), [&count_of_shapes](size_t x) { return x < count_of_shapes; }),
                    "constant_input_idces must be less than input shapes size");

    std::vector<ov::PartialShape> shapes_to_check;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        if (idces_to_remove.count(i) == 0)
            shapes_to_check.push_back(input_shapes[i]);
    }
    SnippetsFunctionBase::validate_params_shape(shapes_to_check, f->get_parameters());
}

std::shared_ptr<ov::Model> MatMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    ov::ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precisions[1], input_shapes[1], matmul_type, params);
    std::shared_ptr<Node> matmul;
    if (precisions[1] == ov::element::i8) {
        matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
            std::vector<element::Type>{ov::element::f32, element::f32},
            std::vector<element::Type>{element::f32},
            ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(data1, element::f32).get(),
            false, transpose_b);
    } else {
        matmul = std::make_shared<op::v0::MatMul>(data0, data1, false, transpose_b);
    }
    return std::make_shared<ov::Model>(NodeVector{matmul}, params);
}
std::shared_ptr<ov::Model> MatMulFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    ov::ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precisions[1], input_shapes[1], matmul_type, params);
    auto indata0 = std::make_shared<op::v0::Parameter>(precisions[0], data0->get_output_partial_shape(0));
    auto indata1 = std::make_shared<op::v0::Parameter>(precisions[1], data1->get_output_partial_shape(0));
    std::shared_ptr<Node> matmul;
    if (precisions[1] == ov::element::i8) {
        matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ov::op::TemporaryReplaceOutputType(indata0, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(indata1, element::f32).get(),
                false, transpose_b);
    } else {
        matmul = std::make_shared<op::v0::MatMul>(indata0, indata1, false, transpose_b);
    }
    const auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0, data1},
                                                                std::make_shared<ov::Model>(NodeVector{matmul},
                                                                                            ParameterVector{indata0, indata1}));
    return std::make_shared<ov::Model>(NodeVector{subgraph}, params);
}
std::shared_ptr<ov::Model> FQMatMulFunction::initOriginal() const {
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 1, 3});
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    ParameterVector params{data0};

    auto in0 = make_fake_quantize(data0, true);
    if (pos == 0) {
        in0 = std::make_shared<op::v1::Transpose>(in0, const_order);
    }

    auto data1 = make_matmul_b_input(ov::element::i8, input_shapes[1], matmul_type, params);
    auto convert = std::make_shared<op::v0::Convert>(data1, ov::element::f32);
    auto deq_mul = std::make_shared<op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{0.00499185826});
    auto mul = std::make_shared<op::v1::Multiply>(convert, deq_mul);
    std::shared_ptr<ov::Node> in1 = mul;
    if (pos == 1) {
        in1 = std::make_shared<op::v1::Transpose>(in1, const_order);
    }
    auto matmul = std::make_shared<op::v0::MatMul>(in0, in1);
    std::shared_ptr<ov::Node> out = matmul;
    if (pos == 2) {
        out = std::make_shared<op::v1::Transpose>(out, const_order);
    }
    return std::make_shared<ov::Model>(NodeVector{out}, params);
}
std::shared_ptr<ov::Model> MatMulBiasFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precision, input_shapes[1], matmul_type, params);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    params.push_back(data2);

    std::shared_ptr<Node> matmul;
    if (precisions[1]  == ov::element::i8) {
        matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                std::vector<element::Type>{ element::f32, element::f32 },
                std::vector<element::Type>{ element::f32 },
                ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    } else {
        matmul = std::make_shared<op::v0::MatMul>(data0, data1);
    }
    auto bias = std::make_shared<op::v1::Add>(matmul, data2);
    return std::make_shared<ov::Model>(NodeVector{bias}, params);
}
std::shared_ptr<ov::Model> MatMulBiasQuantizedFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precisions[1], input_shapes[1], matmul_type, params);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    params.push_back(data2);

    auto matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
        std::vector<element::Type>{ov::element::f32, element::f32},
        std::vector<element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    auto fq2 = make_fake_quantize(matmul, true);
    auto bias = std::make_shared<op::v1::Add>(fq2, data2);
    return std::make_shared<ov::Model>(NodeVector{bias}, params);
}
std::shared_ptr<ov::Model> MatMulsQuantizedFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precisions[1], input_shapes[1], matmul_type, params);
    auto data2 = make_matmul_b_input(precision, input_shapes[2], matmul_type, params);
    auto matmul0 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
        std::vector<element::Type>{ov::element::f32, element::f32},
        std::vector<element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    auto fq0 = make_fake_quantize(matmul0, false);
    auto fq2 = make_fake_quantize(data2, true);
    auto new_shape = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{4},
                                                            std::vector<uint64_t>{1, 1, input_shapes[2].get_shape()[0], input_shapes[2].get_shape()[1]});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(fq2, new_shape, false);
    auto matmul1 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
        std::vector<element::Type>{ov::element::f32, element::f32},
        std::vector<element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(fq0, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(reshape, element::f32).get());
    auto fq3 = make_fake_quantize(matmul1, true);
    return std::make_shared<ov::Model>(NodeVector{fq3}, params);
}
std::shared_ptr<ov::Model> Transpose0213MatMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precisions[1], input_shapes[1], matmul_type, params);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 1, 3});
    std::shared_ptr<Node> result;
    switch (transpose_position) {
        case 0: {
            auto transpose = std::make_shared<op::v1::Transpose>(data0, const_order);
            if (precisions[1] == ov::element::i8) {
                result = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                    std::vector<element::Type>{ov::element::f32, element::f32},
                    std::vector<element::Type>{element::f32},
                    ov::op::TemporaryReplaceOutputType(transpose, element::f32).get(),
                    ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
            } else {
                result = std::make_shared<op::v0::MatMul>(transpose, data1);
            }
            break;
        } case 1: {
            auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
            if (precisions[1] == ov::element::i8) {
                result = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                    std::vector<element::Type>{ov::element::f32, element::f32},
                    std::vector<element::Type>{element::f32},
                    ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                    ov::op::TemporaryReplaceOutputType(transpose, element::f32).get());
            } else {
                result = std::make_shared<op::v0::MatMul>(data0, transpose);
            }
            break;
        } case 2: {
            std::shared_ptr<ov::Node> matmul;
            if (precisions[1] == ov::element::i8) {
                matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
                    std::vector<element::Type>{ov::element::f32, element::f32},
                    std::vector<element::Type>{element::f32},
                    ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
                    ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
            } else {
                matmul = std::make_shared<op::v0::MatMul>(data0, data1);
            }
            result = std::make_shared<op::v1::Transpose>(matmul, const_order);
            break;
        }
    }
    return std::make_shared<ov::Model>(NodeVector{result}, params);
}

std::shared_ptr<ov::Model> TransposeMatMulFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
    auto matmul = std::make_shared<op::v0::MatMul>(data0, transpose);
    return std::make_shared<ov::Model>(NodeVector{matmul}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> TransposeMatMulBiasFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
    auto matmul = std::make_shared<op::v0::MatMul>(data0, transpose);
    auto bias = std::make_shared<op::v1::Add>(matmul, data2);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2});
}
std::shared_ptr<ov::Model> TransposeMulMatMulBiasFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto data2 = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto data3 = std::make_shared<op::v0::Parameter>(precision, input_shapes[3]);
    auto const_order = std::make_shared<op::v0::Constant>(ov::element::i32, Shape {4}, std::vector<int>{0, 2, 3, 1});
    auto transpose = std::make_shared<op::v1::Transpose>(data1, const_order);
    auto mul = std::make_shared<op::v1::Multiply>(transpose, data2);
    auto matmul = std::make_shared<op::v0::MatMul>(data0, mul);
    auto bias = std::make_shared<op::v1::Add>(matmul, data3);
    return std::make_shared<ov::Model>(NodeVector{bias}, ParameterVector{data0, data1, data2, data3});
}
std::shared_ptr<ov::Model> MatMulsQuantizedSoftmaxFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precisions[0], input_shapes[0]);
    ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precisions[1], input_shapes[1], matmul_type, params);
    auto data2 = make_matmul_b_input(precision, input_shapes[2], matmul_type, params);
    auto matmul0 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
        std::vector<element::Type>{ov::element::f32, element::f32},
        std::vector<element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(data1, element::f32).get());
    auto softmax = std::make_shared<ov::op::v8::Softmax>(matmul0, -1);
    auto fq0 = make_fake_quantize(softmax, false);
    auto fq2 = make_fake_quantize(data2, true);
    auto new_shape = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{4},
                                                            std::vector<uint64_t>{1, 1, input_shapes[2].get_shape()[0], input_shapes[2].get_shape()[1]});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(fq2, new_shape, false);
    auto matmul1 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
        std::vector<element::Type>{ov::element::f32, element::f32},
        std::vector<element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(fq0, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(reshape, element::f32).get());
    auto fq3 = make_fake_quantize(matmul1, true);
    return std::make_shared<ov::Model>(NodeVector{fq3}, params);
}

std::shared_ptr<ov::Model> MatMulEltwiseChainFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precision, input_shapes[1], matmul_type, params);

    const auto matmul = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
        std::vector<element::Type>{element::f32, element::f32},
        std::vector<element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(data1, element::f32).get());

    auto scale = ov::test::utils::make_constant(precision, {});
    auto mul = std::make_shared<ov::op::v1::Multiply>(matmul, scale);

    ov::Shape bias_shape(matmul->get_output_partial_shape(0).size(), 1);
    auto OC = *matmul->get_output_partial_shape(0).rbegin();
    if (OC.is_static())
        bias_shape.back() = OC.get_length();
    auto bias = ov::test::utils::make_constant(precision, bias_shape);
    auto bias_op = std::make_shared<op::v1::Add>(mul, bias);

    auto add = std::make_shared<op::v1::Add>(matmul, bias_op);
    return std::make_shared<ov::Model>(NodeVector{add}, params);
}

std::shared_ptr<ov::Model> MatMulEltwiseChainCascadeFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    ParameterVector params{data0};
    auto data1 = make_matmul_b_input(precision, input_shapes[1], matmul_type, params);
    auto data2 = make_matmul_b_input(precision, input_shapes[2], matmul_type, params);

    const auto matmul1 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
        std::vector<element::Type>{element::f32, element::f32},
        std::vector<element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(data0, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(data1, element::f32).get());

    auto build_eltwise_chain = [&](const ov::Output<ov::Node>& out) {
        auto scale = ov::test::utils::make_constant(precision, {});
        auto mul = std::make_shared<ov::op::v1::Multiply>(out, scale);

        ov::Shape bias_shape(out.get_partial_shape().size(), 1);
        auto OC = *out.get_partial_shape().rbegin();
        if (OC.is_static())
            bias_shape.back() = OC.get_length();
        auto bias = ov::test::utils::make_constant(precision, bias_shape);
        auto bias_op = std::make_shared<op::v1::Add>(mul, bias);
        return bias_op;
    };

    auto eltwise_chain_1 = build_eltwise_chain(matmul1);
    auto add = std::make_shared<op::v1::Add>(matmul1, eltwise_chain_1);

    const auto matmul2 = std::make_shared<op::TypeRelaxed<op::v0::MatMul>>(
        std::vector<element::Type>{element::f32, element::f32},
        std::vector<element::Type>{element::f32},
        ov::op::TemporaryReplaceOutputType(add, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(data2, element::f32).get());

    auto eltwise_chain_2 = build_eltwise_chain(matmul2);
    return std::make_shared<ov::Model>(NodeVector{eltwise_chain_2}, params);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
