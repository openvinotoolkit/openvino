// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gated_residual_fusion.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/symbol.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_ops/rms.hpp"

using namespace testing;
using namespace ov;

namespace {

constexpr size_t C = 4;
constexpr size_t D = 8;
constexpr size_t L = 6;
constexpr size_t CD = C * D;
constexpr size_t GROUPS = 2;
constexpr float EPS = 1e-6F;

using Gr = op::internal::GatedResidual;

// Reshape with special zero: a 0 keeps the (dynamic) dimension of the input.
std::shared_ptr<op::v1::Reshape> reshape_to(const Output<Node>& src, const std::vector<int64_t>& shape) {
    auto target = op::v0::Constant::create(element::i64, Shape{shape.size()}, shape);
    return std::make_shared<op::v1::Reshape>(src, target, true);
}

// Only the last dimension of the activations is fixed: the leading ones (rows) are dynamic.
PartialShape with_last(PartialShape rows, size_t last) {
    rows.push_back(Dimension(static_cast<int64_t>(last)));
    return rows;
}

// Reshape target that keeps `leading` dimensions and appends `tail`.
std::vector<int64_t> keep_rows(size_t leading, const std::vector<int64_t>& tail) {
    std::vector<int64_t> target(leading, 0);
    target.insert(target.end(), tail.begin(), tail.end());
    return target;
}

std::shared_ptr<op::v0::Constant> scalar(float v) {
    return op::v0::Constant::create(element::f32, Shape{}, {v});
}

std::shared_ptr<op::v0::Constant> filled_f32(const Shape& shape, float value) {
    return op::v0::Constant::create(element::f32, shape, std::vector<float>(shape_size(shape), value));
}

std::shared_ptr<op::v0::Constant> u4_constant(const Shape& shape, uint8_t byte) {
    Tensor tensor(element::u4, shape);
    std::memset(tensor.data(), byte, tensor.get_byte_size());
    return std::make_shared<op::v0::Constant>(tensor);
}

std::shared_ptr<op::v0::Constant> i4_constant(const Shape& shape, uint8_t byte) {
    Tensor tensor(element::i4, shape);
    std::memset(tensor.data(), byte, tensor.get_byte_size());
    return std::make_shared<op::v0::Constant>(tensor);
}

// How the zero point of a compressed weight is stored.
enum class Zp { PerGroup, MidScalar, None };

// The constants the op takes for a compressed [n, k] weight: [n, k] weights, [n, G] scale and zero point.
struct CompressedParts {
    Output<Node> weights;
    Output<Node> scale;
    Output<Node> zp;  // empty without a zero point
};

CompressedParts compressed_parts(size_t n, size_t k, Zp zp) {
    CompressedParts parts;
    // A scalar zero point of 2^(bits - 1) on unsigned weights is the symmetric form: the op takes signed weights.
    parts.weights = zp == Zp::MidScalar ? i4_constant(Shape{n, k}, 0x5A ^ 0x88)->output(0)
                                        : u4_constant(Shape{n, k}, 0x5A)->output(0);
    parts.scale = filled_f32(Shape{n, GROUPS}, 0.02F);
    if (zp == Zp::PerGroup) {
        parts.zp = u4_constant(Shape{n, GROUPS}, 0x33);
    }
    return parts;
}

// NNCF-style compressed weight [N, G, K/G] -> Convert [-> Subtract(zp)] -> Multiply(scale) -> Reshape [N, K].
Output<Node> compressed_weight(size_t n, size_t k, Zp zp) {
    auto weights = u4_constant(Shape{n, GROUPS, k / GROUPS}, 0x5A);
    Output<Node> value = std::make_shared<op::v0::Convert>(weights, element::f32);
    if (zp != Zp::None) {
        auto zp_const =
            zp == Zp::PerGroup ? u4_constant(Shape{n, GROUPS, 1}, 0x33) : u4_constant(Shape{1}, 0x88);  // 8 = 2^(4 - 1)
        value = std::make_shared<op::v1::Subtract>(value, std::make_shared<op::v0::Convert>(zp_const, element::f32));
    }
    auto mul = std::make_shared<op::v1::Multiply>(value, filled_f32(Shape{n, GROUPS, 1}, 0.02F));
    return reshape_to(mul, {static_cast<int64_t>(n), static_cast<int64_t>(k)})->output(0);
}

enum class Norm { Decomposed, FusedRms, GammalessRms, GammalessRmsNoGain };
enum class Scale { Multiply, Divide, Folded };  // the 1/C of a projection
enum class Mix { SumScaled, SumUnscaled, Mean };

struct ReadOptions {
    Norm norm = Norm::Decomposed;
    Scale low_scale = Scale::Multiply;
    Scale gate_scale = Scale::Multiply;
    Mix mix = Mix::SumScaled;
    bool with_gate = false;
    bool compressed = false;
    Zp zp = Zp::PerGroup;
    int64_t reduce_axis = -2;  // branch axis
    size_t leading = 2;        // dynamic leading dimensions of the activations
};

// The projection weight of a GEMM: [N, K] stored with transpose_b.
Output<Node> weight(size_t n, size_t k, const ReadOptions& opt) {
    return opt.compressed ? compressed_weight(n, k, opt.zp) : filled_f32(Shape{n, k}, 0.01F)->output(0);
}

// MatMul(x, W^T) with its 1/C applied the way the options say.
Output<Node> scaled_projection(const Output<Node>& x, size_t n, size_t k, Scale scale, const ReadOptions& opt) {
    const auto w = weight(n, k, opt);
    switch (scale) {
    case Scale::Folded: {
        // 1/C folded into the weights (plain weights only).
        return std::make_shared<op::v0::MatMul>(x,
                                                std::make_shared<op::v1::Multiply>(w, scalar(1.0F / C)),
                                                false,
                                                true);
    }
    case Scale::Divide:
        return std::make_shared<op::v1::Divide>(std::make_shared<op::v0::MatMul>(x, w, false, true),
                                                scalar(static_cast<float>(C)));
    case Scale::Multiply:
    default:
        return std::make_shared<op::v1::Multiply>(std::make_shared<op::v0::MatMul>(x, w, false, true),
                                                  scalar(1.0F / C));
    }
}

// The per-branch normalized, flattened activations [.., C*D].
Output<Node> getNormalizedFlat(const ReadOptions& opt, const Output<Node>& residual, const Output<Node>& gain) {
    const auto split = keep_rows(opt.leading, {static_cast<int64_t>(C), static_cast<int64_t>(D)});
    const auto flat = keep_rows(opt.leading, {static_cast<int64_t>(CD)});
    auto r3 = reshape_to(residual, split);
    switch (opt.norm) {
    case Norm::FusedRms:
        return reshape_to(std::make_shared<ov::op::internal::RMS>(r3, filled_f32(Shape{C, D}, 1.0F), EPS), flat);
    case Norm::GammalessRms:
        return std::make_shared<op::v1::Multiply>(reshape_to(std::make_shared<ov::op::internal::RMS>(r3, EPS), flat),
                                                  gain);
    case Norm::GammalessRmsNoGain:
        return reshape_to(std::make_shared<ov::op::internal::RMS>(r3, EPS), flat);
    case Norm::Decomposed:
    default: {
        auto sq = std::make_shared<op::v1::Power>(r3, scalar(2.0F));
        auto mean =
            std::make_shared<op::v1::ReduceMean>(sq, op::v0::Constant::create(element::i64, Shape{1}, {-1}), true);
        auto rms = std::make_shared<op::v0::Sqrt>(std::make_shared<op::v1::Add>(mean, scalar(EPS)));
        auto normalized = std::make_shared<op::v1::Multiply>(r3, std::make_shared<op::v1::Divide>(scalar(1.0F), rms));
        return std::make_shared<op::v1::Multiply>(reshape_to(normalized, flat), gain);
    }
    }
}

// The write gate S = 2 * sigmoid(Rbar W_inject^T / C).
Output<Node> getWriteGate(const ReadOptions& opt, const Output<Node>& rbar_flat) {
    auto gate = std::make_shared<op::v0::Sigmoid>(scaled_projection(rbar_flat, C, CD, opt.gate_scale, opt));
    return std::make_shared<op::v1::Multiply>(gate, scalar(2.0F));
}

// Canonical READ graph of the layer, decomposed as checkpoints export it.
std::shared_ptr<Model> getReadModel(const ReadOptions& opt) {
    auto residual =
        std::make_shared<op::v0::Parameter>(element::f32, with_last(PartialShape::dynamic(opt.leading), CD));
    auto gain = std::make_shared<op::v0::Parameter>(element::f32, Shape{CD});
    const auto split = keep_rows(opt.leading, {static_cast<int64_t>(C), static_cast<int64_t>(D)});
    const auto rbar_flat = getNormalizedFlat(opt, residual, gain);

    auto act = std::make_shared<op::v4::Swish>(scaled_projection(rbar_flat, L, CD, opt.low_scale, opt));
    auto up = std::make_shared<op::v0::MatMul>(act, weight(CD, L, opt), false, true);
    auto read_gate = std::make_shared<op::v0::Sigmoid>(up);
    Output<Node> mixed;
    if (opt.mix == Mix::Mean) {
        auto weighted = std::make_shared<op::v1::Multiply>(reshape_to(read_gate, split), reshape_to(rbar_flat, split));
        mixed =
            std::make_shared<op::v1::ReduceMean>(weighted,
                                                 op::v0::Constant::create(element::i64, Shape{1}, {opt.reduce_axis}),
                                                 false);
    } else {
        auto weighted = reshape_to(std::make_shared<op::v1::Multiply>(read_gate, rbar_flat), split);
        mixed = std::make_shared<op::v1::ReduceSum>(weighted,
                                                    op::v0::Constant::create(element::i64, Shape{1}, {opt.reduce_axis}),
                                                    false);
        if (opt.mix == Mix::SumScaled) {
            mixed = std::make_shared<op::v1::Multiply>(mixed, scalar(1.0F / C));
        }
    }

    ResultVector results{std::make_shared<op::v0::Result>(mixed)};
    if (opt.with_gate) {
        results.push_back(std::make_shared<op::v0::Result>(getWriteGate(opt, rbar_flat)));
    }
    return std::make_shared<Model>(results, ParameterVector{residual, gain});
}

// The model the fusion must produce for the same options. With `keep_gate_branch` the op has no write-gate input and
// the gate stays decomposed on the normalization, which the original still computes.
std::shared_ptr<Model> getReadModelRef(const ReadOptions& opt, bool keep_gate_branch = false) {
    auto residual =
        std::make_shared<op::v0::Parameter>(element::f32, with_last(PartialShape::dynamic(opt.leading), CD));
    auto gain = std::make_shared<op::v0::Parameter>(element::f32, Shape{CD});

    Output<Node> gain_flat = gain;
    if (opt.norm == Norm::FusedRms) {
        // The pass flattens the RMS gamma [C, D] with a plain Reshape.
        gain_flat = std::make_shared<op::v1::Reshape>(
            filled_f32(Shape{C, D}, 1.0F),
            op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(CD)}),
            false);
    } else if (opt.norm == Norm::GammalessRmsNoGain) {
        gain_flat = filled_f32(Shape{CD}, 1.0F);
    }

    const bool op_gate = opt.with_gate && !keep_gate_branch;
    OutputVector inputs{residual, gain_flat};
    const auto add_weight = [&](size_t n, size_t k) -> CompressedParts {
        if (!opt.compressed) {
            inputs.push_back(filled_f32(Shape{n, k}, 0.01F));
            return {};
        }
        return compressed_parts(n, k, opt.zp);
    };
    if (opt.compressed) {
        const auto down = compressed_parts(L, CD, opt.zp);
        const auto up = compressed_parts(CD, L, opt.zp);
        inputs.insert(inputs.end(), {down.weights, up.weights, down.scale, up.scale});
        if (down.zp.get_node() != nullptr) {
            inputs.insert(inputs.end(), {down.zp, up.zp});
        }
        if (op_gate) {
            const auto inject = compressed_parts(C, CD, opt.zp);
            inputs.insert(inputs.end(), {inject.weights, inject.scale});
            if (inject.zp.get_node() != nullptr) {
                inputs.push_back(inject.zp);
            }
        }
    } else {
        add_weight(L, CD);
        add_weight(CD, L);
        if (op_gate) {
            add_weight(C, CD);
        }
    }

    auto gr = std::make_shared<Gr>(inputs, Gr::Mode::READ, C, D, L, EPS, false, op_gate);
    ResultVector results{std::make_shared<op::v0::Result>(gr->output(0))};
    if (op_gate) {
        results.push_back(std::make_shared<op::v0::Result>(gr->output(1)));
    } else if (opt.with_gate) {
        results.push_back(std::make_shared<op::v0::Result>(getWriteGate(opt, getNormalizedFlat(opt, residual, gain))));
    }
    return std::make_shared<Model>(results, ParameterVector{residual, gain});
}

// R' = R + Reshape(S * Y), the exporter's form: the Multiply is flattened to [.., C*D] and added to the flat residual.
// `y_rows` are the leading dimensions of Y.
std::shared_ptr<Model> getWriteModel(const PartialShape& rows, const PartialShape& y_rows) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, CD));
    auto y = std::make_shared<op::v0::Parameter>(element::f32, with_last(y_rows, D));
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, C));

    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto scaled = std::make_shared<op::v1::Multiply>(s3, y3);
    auto flat = reshape_to(scaled, keep_rows(rows.size(), {static_cast<int64_t>(CD)}));
    auto out = std::make_shared<op::v1::Add>(residual, flat);
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                   ParameterVector{residual, y, gate});
}

// The same update with the Add in the split [.., C, D] domain: the form the sink produces.
std::shared_ptr<Model> getWriteSplitModel(const PartialShape& rows, const PartialShape& y_rows) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, CD));
    auto y = std::make_shared<op::v0::Parameter>(element::f32, with_last(y_rows, D));
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, C));

    auto r3 = reshape_to(residual, keep_rows(rows.size(), {static_cast<int64_t>(C), static_cast<int64_t>(D)}));
    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto updated = std::make_shared<op::v1::Add>(r3, std::make_shared<op::v1::Multiply>(s3, y3));
    auto out = reshape_to(updated, keep_rows(rows.size(), {static_cast<int64_t>(CD)}));
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                   ParameterVector{residual, y, gate});
}

// The fused WRITE op.
std::shared_ptr<Model> getWriteRef(const PartialShape& rows) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, CD));
    auto y = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, D));
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, C));
    auto gr = std::make_shared<Gr>(OutputVector{residual, y, gate}, Gr::Mode::WRITE, C, D, 0, 0.0F, false, false);
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(gr)},
                                   ParameterVector{residual, y, gate});
}

}  // namespace

class GatedResidualFusionTests : public TransformationTestsF {
public:
    GatedResidualFusionTests() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::GatedResidualFusion>();
    }
};

// The CPU plugin's choice: the Add of the WRITE pattern is moved, not fused into the op.
class GatedResidualWriteSinkReshapeTests : public GatedResidualFusionTests {
protected:
    void SetUp() override {
        GatedResidualFusionTests::SetUp();
        manager.get_pass_config()->disable<pass::GatedResidualWriteFusion>();
    }
};

TEST_F(GatedResidualFusionTests, Read_DecomposedNorm) {
    model = getReadModel({});
    model_ref = getReadModelRef({});
}

TEST_F(GatedResidualFusionTests, Read_FusedRms) {
    ReadOptions opt;
    opt.norm = Norm::FusedRms;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_GammalessRms) {
    ReadOptions opt;
    opt.norm = Norm::GammalessRms;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_GammalessRmsWithoutGain) {
    ReadOptions opt;
    opt.norm = Norm::GammalessRmsNoGain;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_ReduceMeanMix) {
    ReadOptions opt;
    opt.mix = Mix::Mean;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_DownScaleFoldedIntoWeights) {
    ReadOptions opt;
    opt.low_scale = Scale::Folded;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_DownScaleAsDivide) {
    ReadOptions opt;
    opt.low_scale = Scale::Divide;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_WriteGate) {
    ReadOptions opt;
    opt.with_gate = true;
    // The gate Result is fed by the op, which keeps the name of the mix.
    disable_result_friendly_names_check();
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_WriteGateScaleAsDivide) {
    ReadOptions opt;
    opt.with_gate = true;
    // The gate Result is fed by the op, which keeps the name of the mix.
    disable_result_friendly_names_check();
    opt.gate_scale = Scale::Divide;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_WriteGateScaleFoldedIntoWeights) {
    ReadOptions opt;
    opt.with_gate = true;
    // The gate Result is fed by the op, which keeps the name of the mix.
    disable_result_friendly_names_check();
    opt.gate_scale = Scale::Folded;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_OneDynamicRowDimension) {
    ReadOptions opt;
    opt.with_gate = true;
    // The gate Result is fed by the op, which keeps the name of the mix.
    disable_result_friendly_names_check();
    opt.leading = 1;  // [rows, C*D]
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_CompressedWeightsWithZeroPoints) {
    ReadOptions opt;
    opt.compressed = true;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_CompressedWeightsWithoutZeroPoints) {
    ReadOptions opt;
    opt.compressed = true;
    opt.zp = Zp::None;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_CompressedSymmetricScalarZeroPointBecomesSignedWeights) {
    ReadOptions opt;
    opt.compressed = true;
    opt.zp = Zp::MidScalar;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_CompressedWeightsWithWriteGate) {
    ReadOptions opt;
    opt.compressed = true;
    opt.with_gate = true;
    // The gate Result is fed by the op, which keeps the name of the mix.
    disable_result_friendly_names_check();
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_CallbackRejectsTheFusedOp) {
    model = getReadModel({});
    manager.get_pass_config()->set_callback<pass::GatedResidualReadFusion>([](const std::shared_ptr<const Node>&) {
        return true;
    });
}

TEST_F(GatedResidualFusionTests, Read_RejectedWriteGateIsRetriedWithoutIt) {
    ReadOptions opt;
    opt.with_gate = true;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt, /*keep_gate_branch=*/true);
    // Only the form with the write-gate input is rejected, e.g. because its weights are not supported.
    manager.get_pass_config()->set_callback<pass::GatedResidualReadFusion>([](const std::shared_ptr<const Node>& node) {
        return node->get_input_size() > 4;
    });
}

TEST_F(GatedResidualFusionTests, Read_MixWithoutScaleIsNotFused) {
    ReadOptions opt;
    opt.mix = Mix::SumUnscaled;
    model = getReadModel(opt);
}

TEST_F(GatedResidualFusionTests, Read_ReduceOverNonBranchAxisIsNotFused) {
    ReadOptions opt;
    opt.reduce_axis = 1;  // the token axis
    model = getReadModel(opt);
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_AddMovesNextToTheMultiply) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    model_ref = getWriteSplitModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_PagedAttentionRows) {
    model = getWriteModel(PartialShape::dynamic(1), PartialShape::dynamic(1));
    model_ref = getWriteSplitModel(PartialShape::dynamic(1), PartialShape::dynamic(1));
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_ThreeRowDimensions) {
    model = getWriteModel(PartialShape::dynamic(3), PartialShape::dynamic(3));
    model_ref = getWriteSplitModel(PartialShape::dynamic(3), PartialShape::dynamic(3));
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_StaticRows) {
    model = getWriteModel(PartialShape{2, 3}, PartialShape{2, 3});
    model_ref = getWriteSplitModel(PartialShape{2, 3}, PartialShape{2, 3});
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_BroadcastRowsKeepTheirBroadcasting) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape{1, 1});
    model_ref = getWriteSplitModel(PartialShape::dynamic(2), PartialShape{1, 1});
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_StaticBroadcastRowsKeepTheirBroadcasting) {
    model = getWriteModel(PartialShape{2, 3}, PartialShape{1, 3});
    model_ref = getWriteSplitModel(PartialShape{2, 3}, PartialShape{1, 3});
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_ReshapeThatChangesTheRowsIsNotRewritten) {
    // [2, 3, C, D] flattened to [6, C*D] no longer lines up with the rows of the residual.
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, static_cast<int64_t>(CD)});
    auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, static_cast<int64_t>(D)});
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, static_cast<int64_t>(C)});
    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto flat = reshape_to(std::make_shared<op::v1::Multiply>(s3, y3), {6, static_cast<int64_t>(CD)});
    auto out = std::make_shared<op::v1::Add>(residual, flat);
    model = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                    ParameterVector{residual, y, gate});
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_DynamicHiddenSizeIsNotRewritten) {
    // D is dynamic but carries a symbol, which a named pattern dimension would bind to.
    auto symbol = std::make_shared<Symbol>();
    Dimension dynamic_d = Dimension::dynamic();
    dynamic_d.set_symbol(symbol);
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, with_last(PartialShape::dynamic(2), CD));
    auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, -1, dynamic_d});
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, with_last(PartialShape::dynamic(2), C));
    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto flat = reshape_to(std::make_shared<op::v1::Multiply>(s3, y3), {0, 0, static_cast<int64_t>(CD)});
    auto out = std::make_shared<op::v1::Add>(residual, flat);
    model = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                    ParameterVector{residual, y, gate});
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_CallbackKeepsTheOriginalGraph) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    manager.get_pass_config()->set_callback<pass::GatedResidualWriteSinkReshape>(
        [](const std::shared_ptr<const Node>&) {
            return true;
        });
}

TEST_F(GatedResidualFusionTests, Write_ExporterFormIsFusedByDefault) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    model_ref = getWriteRef(PartialShape::dynamic(2));
}

TEST_F(GatedResidualFusionTests, Write_SplitFormIsFusedByDefault) {
    model = getWriteSplitModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    model_ref = getWriteRef(PartialShape::dynamic(2));
}

TEST_F(GatedResidualFusionTests, Write_RowsThatMayBroadcastAreNotFused) {
    // The fused op indexes Y and the gate per residual row, so rows that may broadcast stay a plain graph; the sink
    // still moves the Add.
    model = getWriteModel(PartialShape::dynamic(2), PartialShape{1, 1});
    model_ref = getWriteSplitModel(PartialShape::dynamic(2), PartialShape{1, 1});
}

TEST_F(GatedResidualFusionTests, Write_BothWritePassesDisabledLeavesTheGraph) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    manager.get_pass_config()->disable<pass::GatedResidualWriteFusion>();
    manager.get_pass_config()->disable<pass::GatedResidualWriteSinkReshape>();
}
