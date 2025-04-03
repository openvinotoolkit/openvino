// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/snippets/x64/pass/fuse_brgemm_cpu_postops.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "dnnl_types.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/scalar.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

using namespace testing;
using namespace ov::intel_cpu;

namespace {
std::shared_ptr<ov::intel_cpu::BrgemmCPU> make_brgemm(const ov::OutputVector& inputs,
                                                      const BrgemmCPU::PostopsConfig& postops = {}) {
    return std::make_shared<BrgemmCPU>(inputs,
                                       ov::intel_cpu::BrgemmCPU::BRGEMM_TYPE::STAND_ALONE,
                                       std::vector<ov::snippets::modifier::MemoryAccess::PortDescriptor>{},
                                       ov::snippets::modifier::MemoryAccess::PortDescriptor{0, 0},
                                       std::vector<size_t>{},
                                       std::vector<size_t>{},
                                       std::vector<size_t>{},
                                       postops);
}

std::shared_ptr<ov::Node> make_eltwise(const ov::Output<ov::Node>& brgemm,
                                       const ov::Output<ov::Node>& postop_input,
                                       const ov::Node::type_info_t& op_type) {
    if (op_type == ov::opset1::Multiply::get_type_info_static()) {
        return std::make_shared<ov::opset1::Multiply>(brgemm, postop_input);
    } else if (op_type == ov::opset1::Add::get_type_info_static()) {
        return std::make_shared<ov::opset1::Add>(brgemm, postop_input);
    } else if (op_type == ov::opset1::Maximum::get_type_info_static()) {
        return std::make_shared<ov::opset1::Maximum>(brgemm, postop_input);
    } else if (op_type == ov::opset1::Minimum::get_type_info_static()) {
        return std::make_shared<ov::opset1::Minimum>(brgemm, postop_input);
    } else {
        OPENVINO_THROW("Unsupported operation type: ", op_type.name);
    }
}

static const ov::PartialShape brgemm_a_shape{-1, -1, -1, -1};
static const ov::PartialShape brgemm_b_shape{-1, -1, -1, 128};
}  // namespace

class FuseBrgemmCPUPostopsTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::FuseBrgemmCPUPostops>(external_params_idces);
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    }

    void TearDown() override {
        TransformationTestsF::TearDown();
        ASSERT_EQ(external_params_idces, expected_external_params_idces);
    }

    std::set<size_t> external_params_idces = {};
    std::set<size_t> expected_external_params_idces = {};
};

using FuseConvertParams = std::tuple<std::pair<ov::element::Type, ov::element::Type>,  // Brgemm input precisions
                                     ov::element::Type>;  // Convert destination element type

class FuseConvertTests : public FuseBrgemmCPUPostopsTests, public WithParamInterface<FuseConvertParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseConvertParams> obj) {
        auto [input_precisions, convert_dst_type] = obj.param;
        std::ostringstream result;
        result << "InputPrecisions=(" << input_precisions.first << "_" << input_precisions.second
               << ")_ConvertDstType=" << convert_dst_type;
        return result.str();
    }

    static std::shared_ptr<ov::Model> get_model(const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
                                                ov::element::Type convert_dst_type) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, brgemm_b_shape);
        auto brgemm = make_brgemm({input1, input2});
        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, convert_dst_type);

        return std::make_shared<ov::Model>(ov::NodeVector{convert}, ov::ParameterVector{input1, input2});
    }

    static std::shared_ptr<ov::Model> get_ref_model(
        const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
        ov::element::Type convert_dst_type) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = convert_dst_type;
        auto ref_brgemm = make_brgemm({input1, input2}, postops);

        return std::make_shared<ov::Model>(ov::NodeVector{ref_brgemm}, ov::ParameterVector{input1, input2});
    }

protected:
    void SetUp() override {
        FuseBrgemmCPUPostopsTests::SetUp();
        auto [input_precisions, convert_dst_type] = this->GetParam();
        model = get_model(input_precisions, convert_dst_type);
        model_ref = get_ref_model(input_precisions, convert_dst_type);
    }
};

using FuseScalarEltwiseParams = std::tuple<std::pair<ov::element::Type, ov::element::Type>,  // Brgemm input precisions
                                           ov::Node::type_info_t>;                           // Scalar operation type

class FuseScalarEltwiseTests : public FuseBrgemmCPUPostopsTests, public WithParamInterface<FuseScalarEltwiseParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseScalarEltwiseParams> obj) {
        auto [input_precisions, scalar_op_type] = obj.param;
        std::ostringstream result;
        result << "InputPrecisions=(" << input_precisions.first << "_" << input_precisions.second
               << ")_ScalarOp=" << scalar_op_type.name;
        return result.str();
    }

    static std::shared_ptr<ov::Model> get_model(const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
                                                const ov::Node::type_info_t& scalar_op_type) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, brgemm_b_shape);
        std::shared_ptr<ov::Node> brgemm = make_brgemm({input1, input2});
        if (brgemm->get_output_element_type(0) != ov::element::f32) {
            brgemm = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, ov::element::f32);
        }

        auto scalar = std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, 2.f);
        auto scalar_op = make_eltwise(brgemm, scalar, scalar_op_type);
        return std::make_shared<ov::Model>(ov::NodeVector{scalar_op}, ov::ParameterVector{input1, input2});
    }

    static std::shared_ptr<ov::Model> get_ref_model(
        const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
        const ov::Node::type_info_t& scalar_op_type) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = ov::element::f32;

        if (scalar_op_type == ov::opset1::Multiply::get_type_info_static()) {
            postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_linear, 2.0f, 0.0f);
        } else if (scalar_op_type == ov::opset1::Add::get_type_info_static()) {
            postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_linear, 1.0f, 2.0f);
        } else if (scalar_op_type == ov::opset1::Maximum::get_type_info_static()) {
            postops.post_ops.append_eltwise(1.0f,
                                            dnnl::impl::alg_kind_t::dnnl_eltwise_clip,
                                            2.0f,
                                            std::numeric_limits<float>::max());
        } else if (scalar_op_type == ov::opset1::Minimum::get_type_info_static()) {
            postops.post_ops.append_eltwise(1.0f,
                                            dnnl::impl::alg_kind_t::dnnl_eltwise_clip,
                                            -std::numeric_limits<float>::max(),
                                            2.0f);
        } else {
            OPENVINO_THROW("Unsupported scalar operation type: ", scalar_op_type.name);
        }

        auto ref_brgemm = make_brgemm({input1, input2}, postops);
        return std::make_shared<ov::Model>(ov::NodeVector{ref_brgemm}, ov::ParameterVector{input1, input2});
    }

protected:
    void SetUp() override {
        FuseBrgemmCPUPostopsTests::SetUp();
        auto [input_precisions, scalar_op_type] = this->GetParam();
        model = get_model(input_precisions, scalar_op_type);
        model_ref = get_ref_model(input_precisions, scalar_op_type);
    }
};

using FuseBinaryEltwiseParams = std::tuple<std::pair<ov::element::Type, ov::element::Type>,  // Brgemm input precisions
                                           ov::Node::type_info_t,                            // Binary operation type
                                           ov::PartialShape>;                                // Postop input shape

class FuseBinaryEltwiseTests : public FuseBrgemmCPUPostopsTests, public WithParamInterface<FuseBinaryEltwiseParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseBinaryEltwiseParams> obj) {
        auto [input_precisions, binary_op_type, postop_input_shape] = obj.param;
        std::ostringstream result;
        result << "InputPrecisions=(" << input_precisions.first << "_" << input_precisions.second
               << ")_BinaryOp=" << binary_op_type.name << "_PostopInputShape=" << postop_input_shape;
        return result.str();
    }

    static std::shared_ptr<ov::Model> get_model(const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
                                                const ov::Node::type_info_t& binary_op_type,
                                                const ov::PartialShape& postop_input_shape) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, brgemm_b_shape);
        std::shared_ptr<ov::Node> brgemm = make_brgemm({input1, input2});
        if (brgemm->get_output_element_type(0) != ov::element::f32) {
            brgemm = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, ov::element::f32);
        }

        auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, postop_input_shape);
        ov::Output<ov::Node> postop_input = input3;
        if (postop_input_shape.size() < brgemm_b_shape.size()) {
            postop_input = std::make_shared<ov::snippets::op::RankNormalization>(input3, brgemm_b_shape.size() - postop_input_shape.size(), 0);
        }
        auto binary_op = make_eltwise(brgemm, postop_input, binary_op_type);
        return std::make_shared<ov::Model>(ov::NodeVector{binary_op},
                                           ov::ParameterVector{input1, input2, input3});
    }

    static std::shared_ptr<ov::Model> get_ref_model(
        const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
        const ov::Node::type_info_t& binary_op_type,
        const ov::PartialShape& postop_input_shape) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, brgemm_b_shape);
        auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, postop_input_shape);
        ov::Output<ov::Node> postop_input = input3;
        if (postop_input_shape.size() < brgemm_b_shape.size()) {
            postop_input = std::make_shared<ov::snippets::op::RankNormalization>(input3, brgemm_b_shape.size() - postop_input_shape.size(), 0);
        }

        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = ov::element::f32;
        postops.binary_postops_offset = 0;

        const auto OC_dim = *postop_input_shape.rbegin();
        const size_t OC = OC_dim.get_length();
        DnnlBlockedMemoryDesc memory_desc(ov::element::f32, Shape{1, OC});
        if (binary_op_type == ov::opset1::Multiply::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_mul, memory_desc.getDnnlDesc().get());
        } else if (binary_op_type == ov::opset1::Add::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_add, memory_desc.getDnnlDesc().get());
        } else if (binary_op_type == ov::opset1::Maximum::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_max, memory_desc.getDnnlDesc().get());
        } else if (binary_op_type == ov::opset1::Minimum::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_min, memory_desc.getDnnlDesc().get());
        } else {
            OPENVINO_THROW("Unsupported binary operation type: ", binary_op_type.name);
        }

        auto ref_brgemm = make_brgemm({input1, input2, postop_input}, postops);
        return std::make_shared<ov::Model>(ov::NodeVector{ref_brgemm},
                                           ov::ParameterVector{input1, input2, input3});
    }

protected:
    void SetUp() override {
        expected_external_params_idces = {2};
        FuseBrgemmCPUPostopsTests::SetUp();
        auto [input_precisions, binary_op_type, postop_input_shape] = this->GetParam();
        model = get_model(input_precisions, binary_op_type, postop_input_shape);
        model_ref = get_ref_model(input_precisions, binary_op_type, postop_input_shape);
    }
};

TEST_P(FuseConvertTests, CompareFunctions) {}
TEST_P(FuseScalarEltwiseTests, CompareFunctions) {}
TEST_P(FuseBinaryEltwiseTests, CompareFunctions) {}

const std::vector<std::pair<ov::element::Type, ov::element::Type>> input_precisions = {
    {ov::element::f32, ov::element::f32},
    {ov::element::i8, ov::element::i8},
    {ov::element::u8, ov::element::i8},
    {ov::element::bf16, ov::element::bf16}};

const ov::element::TypeVector convert_dst_types = {ov::element::f32, ov::element::i8, ov::element::u8};

INSTANTIATE_TEST_SUITE_P(FuseBrgemmCPUPostopsTests,
                         FuseConvertTests,
                         ::testing::Combine(::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(convert_dst_types)),
                         FuseConvertTests::getTestCaseName);

const std::vector<ov::Node::type_info_t> eltwise_postop_types = {ov::opset1::Multiply::get_type_info_static(),
                                                                 ov::opset1::Add::get_type_info_static(),
                                                                 ov::opset1::Maximum::get_type_info_static(),
                                                                 ov::opset1::Minimum::get_type_info_static()};

INSTANTIATE_TEST_SUITE_P(FuseBrgemmCPUPostopsTests,
                         FuseScalarEltwiseTests,
                         ::testing::Combine(::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(eltwise_postop_types)),
                         FuseScalarEltwiseTests::getTestCaseName);

const std::vector<ov::PartialShape> postop_input_shapes = {{1, 1, 1, 128}, {128}};

const std::vector<std::pair<ov::element::Type, ov::element::Type>> binary_postops_input_precisions = {
    {ov::element::i8, ov::element::i8},
    {ov::element::u8, ov::element::i8},
    {ov::element::bf16, ov::element::bf16}};

INSTANTIATE_TEST_SUITE_P(FuseBrgemmCPUPostopsTests,
                         FuseBinaryEltwiseTests,
                         ::testing::Combine(::testing::ValuesIn(binary_postops_input_precisions),
                                            ::testing::ValuesIn(eltwise_postop_types),
                                            ::testing::ValuesIn(postop_input_shapes)),
                         FuseBinaryEltwiseTests::getTestCaseName);

TEST_F(FuseBrgemmCPUPostopsTests, NegativeUnsupportedConvertDstType) {
    model = FuseConvertTests::get_model({ov::element::bf16, ov::element::bf16}, ov::element::bf16);
}

TEST_F(FuseBrgemmCPUPostopsTests, NegativeBinaryUnsupportedPrecision) {
    model = FuseBinaryEltwiseTests::get_model({ov::element::f32, ov::element::f32},
                                              ov::opset1::Add::get_type_info_static(),
                                              {1, 1, 1, 128});
}

TEST_F(FuseBrgemmCPUPostopsTests, NegativeBinaryUnsupportedShape) {
    model = FuseBinaryEltwiseTests::get_model({ov::element::bf16, ov::element::bf16},
                                              ov::opset1::Add::get_type_info_static(),
                                              {-1, -1, 1, 128});
}

TEST_F(FuseBrgemmCPUPostopsTests, NegativeBinarySharedPostopInput) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::bf16, brgemm_a_shape);
    auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::bf16, brgemm_b_shape);
    auto shared_postop_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{128});

    std::shared_ptr<ov::Node> brgemm_1 = make_brgemm({input1, input2});
    std::shared_ptr<ov::Node> brgemm_2 = make_brgemm({input1, input2});

    auto binary_op_1 = make_eltwise(brgemm_1, shared_postop_input, ov::opset1::Multiply::get_type_info_static());
    auto binary_op_2 = make_eltwise(brgemm_2, shared_postop_input, ov::opset1::Multiply::get_type_info_static());
    model = std::make_shared<ov::Model>(ov::NodeVector{binary_op_1, binary_op_2},
                                        ov::ParameterVector{input1, input2, shared_postop_input});
}

TEST_F(FuseBrgemmCPUPostopsTests, NegativeBrgemmWithSeveralConsumers) {
    auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, brgemm_a_shape);
    auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, brgemm_b_shape);
    std::shared_ptr<ov::Node> brgemm = make_brgemm({input1, input2});

    auto scalar = std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, 2.f);
    auto scalar_op = make_eltwise(brgemm, scalar, ov::opset1::Multiply::get_type_info_static());

    // Additional consumer prevents postops fusion
    auto relu = std::make_shared<ov::opset1::Relu>(brgemm);
    model = std::make_shared<ov::Model>(ov::NodeVector{scalar_op, relu}, ov::ParameterVector{input1, input2});
}
