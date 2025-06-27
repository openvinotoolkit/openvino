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
#include "snippets/op/buffer.hpp"
#include "snippets/op/convert_saturation.hpp"
#include "snippets/op/rank_normalization.hpp"
#include "snippets/op/scalar.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

using namespace testing;
using namespace ov::intel_cpu;

namespace {
std::shared_ptr<ov::intel_cpu::BrgemmCPU> make_brgemm(const ov::OutputVector& main_inputs,
                                                      const BrgemmCPU::PostopsConfig& postops = {},
                                                      const ov::OutputVector& postop_inputs = {}) {
    const auto& a_precision = main_inputs[0].get_element_type();
    const auto& b_precision = main_inputs[1].get_element_type();
    dnnl::impl::cpu::x64::cpu_isa_t isa;
    if (a_precision == ov::element::f32 && b_precision == ov::element::f32) {
        isa = dnnl::impl::cpu::x64::cpu_isa_t::avx512_core;
    } else if (a_precision == ov::element::u8 && b_precision == ov::element::i8) {
        isa = dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_vnni;
    } else if (a_precision == ov::element::i8 && b_precision == ov::element::i8) {
        isa = dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_vnni;
    } else if (a_precision == ov::element::bf16 && b_precision == ov::element::bf16) {
        isa = dnnl::impl::cpu::x64::cpu_isa_t::avx512_core_amx;
    } else {
        OPENVINO_THROW("Unsupported input precisions: ", a_precision, " and ", b_precision);
    }
    ov::intel_cpu::brgemm_utils::BrgemmConfig brgemm_config(isa, a_precision, b_precision, b_precision, false, false);

    auto create_brgemm_cpu = [&brgemm_config, postop_inputs, &postops](const ov::OutputVector& postprocessed_inputs) {
        ov::OutputVector all_inputs = postprocessed_inputs;
        all_inputs.insert(all_inputs.end(), postop_inputs.begin(), postop_inputs.end());
        return std::make_shared<BrgemmCPU>(all_inputs,
                                           brgemm_config,
                                           std::vector<ov::snippets::modifier::MemoryAccess::PortDescriptor>{},
                                           ov::snippets::modifier::MemoryAccess::PortDescriptor{0, 0},
                                           std::vector<size_t>{},
                                           std::vector<size_t>{},
                                           std::vector<size_t>{},
                                           postops);
    };

    if (brgemm_config.is_amx()) {
        auto scratch = std::make_shared<ov::snippets::op::Buffer>(ov::Shape{BrgemmCPU::SCRATCH_BYTE_SIZE});
        return create_brgemm_cpu({main_inputs[0], main_inputs[1], scratch});
    }
    if (brgemm_config.with_compensations()) {
        auto brgemm_repacking = std::make_shared<BrgemmCopyB>(main_inputs[1], brgemm_config);
        return create_brgemm_cpu({main_inputs[0], brgemm_repacking->output(0), brgemm_repacking->output(1)});
    }
    if (brgemm_config.with_wei_repacking()) {
        auto brgemm_repacking = std::make_shared<BrgemmCopyB>(main_inputs[1], brgemm_config);
        return create_brgemm_cpu({main_inputs[0], brgemm_repacking->output(0)});
    }
    return create_brgemm_cpu(main_inputs);
}

std::shared_ptr<ov::Node> make_eltwise(const ov::Output<ov::Node>& brgemm,
                                       const ov::Output<ov::Node>& postop_input,
                                       const ov::Node::type_info_t& op_type) {
    if (op_type == ov::opset1::Multiply::get_type_info_static()) {
        return std::make_shared<ov::opset1::Multiply>(brgemm, postop_input);
    } else if (op_type == ov::opset1::Add::get_type_info_static()) {
        return std::make_shared<ov::opset1::Add>(brgemm, postop_input);
    } else if (op_type == ov::opset1::Subtract::get_type_info_static()) {
        return std::make_shared<ov::opset1::Subtract>(brgemm, postop_input);
    } else if (op_type == ov::opset1::Maximum::get_type_info_static()) {
        return std::make_shared<ov::opset1::Maximum>(brgemm, postop_input);
    } else if (op_type == ov::opset1::Minimum::get_type_info_static()) {
        return std::make_shared<ov::opset1::Minimum>(brgemm, postop_input);
    } else {
        OPENVINO_THROW("Unsupported operation type: ", op_type.name);
    }
}

static const ov::PartialShape brgemm_a_shape{-1, -1, -1, 1024};
static const ov::PartialShape brgemm_b_shape{-1, -1, 1024, 128};
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

    std::set<size_t> expected_external_params_idces = {};

private:
    std::set<size_t> external_params_idces = {};
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
                                                ov::element::Type convert_dst_type,
                                                const ov::PartialShape& a_shape = brgemm_a_shape,
                                                const ov::PartialShape& b_shape = brgemm_b_shape) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, b_shape);
        auto brgemm = make_brgemm({input1, input2});
        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, convert_dst_type);

        return std::make_shared<ov::Model>(ov::OutputVector{convert}, ov::ParameterVector{input1, input2});
    }

    static std::shared_ptr<ov::Model> get_ref_model(
        const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
        ov::element::Type convert_dst_type) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = convert_dst_type;
        auto ref_brgemm = make_brgemm({input1, input2}, postops);

        return std::make_shared<ov::Model>(ov::OutputVector{ref_brgemm}, ov::ParameterVector{input1, input2});
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
        return std::make_shared<ov::Model>(ov::OutputVector{scalar_op}, ov::ParameterVector{input1, input2});
    }

    static std::shared_ptr<ov::Model> get_ref_model(
        const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
        const ov::Node::type_info_t& scalar_op_type) {
        auto input1 = std::make_shared<ov::opset1::Parameter>(input_precisions.first, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(input_precisions.second, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        if (ov::snippets::op::Brgemm::get_output_type(input_precisions.first, input_precisions.second) !=
            ov::element::f32) {
            postops.forced_output_type = ov::element::f32;
        }

        if (scalar_op_type == ov::opset1::Multiply::get_type_info_static()) {
            postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_linear, 2.0f, 0.0f);
        } else if (scalar_op_type == ov::opset1::Add::get_type_info_static()) {
            postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_linear, 1.0f, 2.0f);
        } else if (scalar_op_type == ov::opset1::Subtract::get_type_info_static()) {
            postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_linear, 1.0f, -2.0f);
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
        return std::make_shared<ov::Model>(ov::OutputVector{ref_brgemm}, ov::ParameterVector{input1, input2});
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
        return std::make_shared<ov::Model>(ov::OutputVector{binary_op},
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
        if (ov::snippets::op::Brgemm::get_output_type(input_precisions.first, input_precisions.second) !=
            ov::element::f32) {
            postops.forced_output_type = ov::element::f32;
        }
        postops.binary_postops_offset = 0;

        const auto OC_dim = *postop_input_shape.rbegin();
        const size_t OC = OC_dim.get_length();
        DnnlBlockedMemoryDesc memory_desc(ov::element::f32, Shape{1, OC});
        if (binary_op_type == ov::opset1::Multiply::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_mul, memory_desc.getDnnlDesc().get());
        } else if (binary_op_type == ov::opset1::Add::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_add, memory_desc.getDnnlDesc().get());
        } else if (binary_op_type == ov::opset1::Subtract::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_sub, memory_desc.getDnnlDesc().get());
        } else if (binary_op_type == ov::opset1::Maximum::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_max, memory_desc.getDnnlDesc().get());
        } else if (binary_op_type == ov::opset1::Minimum::get_type_info_static()) {
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_min, memory_desc.getDnnlDesc().get());
        } else {
            OPENVINO_THROW("Unsupported binary operation type: ", binary_op_type.name);
        }

        auto ref_brgemm = make_brgemm({input1, input2}, postops, {postop_input});
        return std::make_shared<ov::Model>(ov::OutputVector{ref_brgemm},
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
                                                                 ov::opset1::Subtract::get_type_info_static(),
                                                                 ov::opset1::Maximum::get_type_info_static(),
                                                                 ov::opset1::Minimum::get_type_info_static()};

INSTANTIATE_TEST_SUITE_P(FuseBrgemmCPUPostopsTests,
                         FuseScalarEltwiseTests,
                         ::testing::Combine(::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(eltwise_postop_types)),
                         FuseScalarEltwiseTests::getTestCaseName);

const std::vector<ov::PartialShape> postop_input_shapes = {{1, 1, 1, 128}, {128}};

INSTANTIATE_TEST_SUITE_P(FuseBrgemmCPUPostopsTests,
                         FuseBinaryEltwiseTests,
                         ::testing::Combine(::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(eltwise_postop_types),
                                            ::testing::ValuesIn(postop_input_shapes)),
                         FuseBinaryEltwiseTests::getTestCaseName);

TEST_F(FuseBrgemmCPUPostopsTests, FuseUnaryEltwiseHalfToEven) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        auto brgemm = make_brgemm({input1, input2});
        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, ov::element::f32);
        auto round = std::make_shared<ov::op::v5::Round>(convert, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);

        model = std::make_shared<ov::Model>(ov::OutputVector{round}, ov::ParameterVector{input1, input2});
    }

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = ov::element::f32;
        postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_round_half_to_even, 0.0f, 0.0f);
        auto ref_brgemm = make_brgemm({input1, input2}, postops);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{ref_brgemm}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(FuseBrgemmCPUPostopsTests, FuseUnaryEltwiseHalfAwayFromZero) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        auto brgemm = make_brgemm({input1, input2});
        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, ov::element::f32);
        auto round = std::make_shared<ov::op::v5::Round>(convert, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);

        model = std::make_shared<ov::Model>(ov::OutputVector{round}, ov::ParameterVector{input1, input2});
    }

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = ov::element::f32;
        postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_round_half_away_from_zero, 0.0f, 0.0f);
        auto ref_brgemm = make_brgemm({input1, input2}, postops);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{ref_brgemm}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(FuseBrgemmCPUPostopsTests, FuseUnaryEltwiseRelu) {
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        auto brgemm = make_brgemm({input1, input2});
        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, ov::element::f32);
        auto relu = std::make_shared<ov::op::v0::Relu>(convert);

        model = std::make_shared<ov::Model>(ov::OutputVector{relu}, ov::ParameterVector{input1, input2});
    }

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = ov::element::f32;
        postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_relu, 0.0f, 0.0f);
        auto ref_brgemm = make_brgemm({input1, input2}, postops);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{ref_brgemm}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(FuseBrgemmCPUPostopsTests, FuseScaleShift) {
    const float scale_val = 2.f;
    const float shift_val = 3.f;
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        auto brgemm = make_brgemm({input1, input2});
        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, ov::element::f32);

        auto scale = std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, scale_val);
        auto shift = std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, shift_val);
        auto scale_op = std::make_shared<ov::opset1::Multiply>(convert, scale);
        auto shift_op = std::make_shared<ov::opset1::Add>(scale_op, shift);

        model = std::make_shared<ov::Model>(ov::OutputVector{shift_op}, ov::ParameterVector{input1, input2});
    }

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = ov::element::f32;
        postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_linear, scale_val, shift_val);
        auto ref_brgemm = make_brgemm({input1, input2}, postops);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{ref_brgemm}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(FuseBrgemmCPUPostopsTests, FuseClip) {
    const float in_low = 0.f;
    const float in_high = 6.f;
    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        auto brgemm = make_brgemm({input1, input2});
        auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm, ov::element::f32);

        auto max_scalar = std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, in_low);
        auto min_scalar = std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, in_high);
        auto max_op = std::make_shared<ov::opset1::Maximum>(convert, max_scalar);
        auto min_op = std::make_shared<ov::opset1::Minimum>(max_op, min_scalar);

        model = std::make_shared<ov::Model>(ov::OutputVector{min_op}, ov::ParameterVector{input1, input2});
    }

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_a_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_b_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = ov::element::f32;
        postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_clip, in_low, in_high);
        auto ref_brgemm = make_brgemm({input1, input2}, postops);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{ref_brgemm}, ov::ParameterVector{input1, input2});
    }
}

TEST_F(FuseBrgemmCPUPostopsTests, BrgemmPostopsCascade) {
    const float in_low = 0.f;
    const float in_high = 6.f;
    const float scalar_mul = 2.f;
    const ov::PartialShape brgemm_input_shape{1, 1, 128, 128};

    {
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_input_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_input_shape);
        ov::ParameterVector parameters{input1, input2};

        auto build_sequence = [&](const std::shared_ptr<ov::Node>& input_node) {
            auto convert = std::make_shared<ov::snippets::op::ConvertSaturation>(input_node, ov::element::f32);

            auto binary_mul_param =
                std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 1, 128});
            parameters.push_back(binary_mul_param);
            auto binary_mul = std::make_shared<ov::opset1::Multiply>(convert, binary_mul_param);

            auto binary_add_param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{128});
            parameters.push_back(binary_add_param);
            auto binary_add_input = std::make_shared<ov::snippets::op::RankNormalization>(binary_add_param, 3, 0);
            auto binary_add = std::make_shared<ov::opset1::Add>(binary_mul, binary_add_input);

            auto scalar_max_node =
                std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, in_low);
            auto max_op = std::make_shared<ov::opset1::Maximum>(binary_add, scalar_max_node);

            auto scalar_min_node =
                std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, in_high);
            auto min_op = std::make_shared<ov::opset1::Minimum>(max_op, scalar_min_node);

            auto scalar_mul_node =
                std::make_shared<ov::snippets::op::Scalar>(ov::element::f32, ov::Shape{}, scalar_mul);
            auto mul_op = std::make_shared<ov::opset1::Multiply>(min_op, scalar_mul_node);
            return std::make_shared<ov::snippets::op::ConvertSaturation>(mul_op, ov::element::u8);
        };

        auto brgemm1 = make_brgemm({input1, input2});
        auto sequence1 = build_sequence(brgemm1);
        auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_input_shape);
        parameters.push_back(input3);
        auto brgemm2 = make_brgemm({sequence1, input3});
        auto sequence2 = build_sequence(brgemm2);
        auto input4 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_input_shape);
        parameters.push_back(input4);
        auto brgemm3 = make_brgemm({sequence2, input4});
        auto final_convert = std::make_shared<ov::snippets::op::ConvertSaturation>(brgemm3, ov::element::f32);

        model = std::make_shared<ov::Model>(ov::OutputVector{final_convert}, parameters);
    }

    {
        DnnlBlockedMemoryDesc memory_desc(ov::element::f32, Shape{1, 128});
        auto create_postops = [&](size_t binary_postops_offset = 0) {
            BrgemmCPU::PostopsConfig postops;
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_mul, memory_desc.getDnnlDesc().get());
            postops.post_ops.append_binary(dnnl::impl::alg_kind_t::dnnl_binary_add, memory_desc.getDnnlDesc().get());
            postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_clip, in_low, in_high);
            postops.post_ops.append_eltwise(1.0f, dnnl::impl::alg_kind_t::dnnl_eltwise_linear, scalar_mul, 0.0f);
            postops.binary_postops_offset = binary_postops_offset;
            postops.forced_output_type = ov::element::u8;
            return postops;
        };
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::u8, brgemm_input_shape);
        auto input2 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_input_shape);
        auto binary_mul_param1 =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 1, 128});
        auto binary_add_param1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{128});
        auto binary_add_input1 = std::make_shared<ov::snippets::op::RankNormalization>(binary_add_param1, 3, 0);
        auto brgemm1 = make_brgemm({input1, input2}, create_postops(0), {binary_mul_param1, binary_add_input1});

        auto input3 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_input_shape);
        auto binary_mul_param2 =
            std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 1, 128});
        auto binary_add_param2 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{128});
        auto binary_add_input2 = std::make_shared<ov::snippets::op::RankNormalization>(binary_add_param2, 3, 0);
        auto brgemm2 = make_brgemm({brgemm1, input3}, create_postops(2), {binary_mul_param2, binary_add_input2});

        auto input4 = std::make_shared<ov::opset1::Parameter>(ov::element::i8, brgemm_input_shape);
        BrgemmCPU::PostopsConfig postops;
        postops.forced_output_type = ov::element::f32;
        auto brgemm3 = make_brgemm({brgemm2, input4}, postops);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{brgemm3},
                                                ov::ParameterVector{input1,
                                                                    input2,
                                                                    binary_mul_param1,
                                                                    binary_add_param1,
                                                                    input3,
                                                                    binary_mul_param2,
                                                                    binary_add_param2,
                                                                    input4});
        // Binary postops' inputs became external parameters
        expected_external_params_idces = {2, 3, 5, 6};
    }
}

TEST_F(FuseBrgemmCPUPostopsTests, NegativeBF16WithInternalBlockingUnsupportedConvertDstType) {
    model = FuseConvertTests::get_model({ov::element::bf16, ov::element::bf16},
                                        ov::element::bf16,
                                        {-1, -1, -1, 37},
                                        {-1, -1, 37, 128});
}

TEST_F(FuseBrgemmCPUPostopsTests, NegativeFP32UnsupportedConvertDstType) {
    model = FuseConvertTests::get_model({ov::element::f32, ov::element::f32}, ov::element::u8);
}

TEST_F(FuseBrgemmCPUPostopsTests, NegativeScalarUnsupportedPrecision) {
    model = FuseScalarEltwiseTests::get_model({ov::element::f32, ov::element::f32},
                                              ov::opset1::Add::get_type_info_static());
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
    model = std::make_shared<ov::Model>(ov::OutputVector{binary_op_1, binary_op_2},
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
    model = std::make_shared<ov::Model>(ov::OutputVector{scalar_op, relu}, ov::ParameterVector{input1, input2});
}