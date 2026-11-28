// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/gated_residual_fusion.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <tuple>

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

// Not a power of two: 1/C is not exact in any floating-point type.
constexpr size_t C = 3;
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

// Dynamic rows that share their symbols: the residual, Y and the gate of one layer have the same number of rows.
PartialShape symbolic_rows(size_t rank) {
    PartialShape rows = PartialShape::dynamic(rank);
    for (auto& dim : rows) {
        dim.set_symbol(std::make_shared<Symbol>());
    }
    return rows;
}

// Reshape target that keeps `leading` dimensions and appends `tail`.
std::vector<int64_t> keep_rows(size_t leading, const std::vector<int64_t>& tail) {
    std::vector<int64_t> target(leading, 0);
    target.insert(target.end(), tail.begin(), tail.end());
    return target;
}

// A scalar of the compute type, stored as `type` and converted back when that is not f32.
Output<Node> scalar(float v, const element::Type& type = element::f32) {
    auto constant = op::v0::Constant::create(type, Shape{}, {v});
    if (type == element::f32) {
        return constant;
    }
    return std::make_shared<op::v0::Convert>(constant, element::f32);
}

std::shared_ptr<op::v0::Constant> filled_f32(const Shape& shape, float value) {
    return op::v0::Constant::create(element::f32, shape, std::vector<float>(shape_size(shape), value));
}

std::shared_ptr<op::v0::Constant> u4_constant(const Shape& shape, uint8_t byte) {
    Tensor tensor(element::u4, shape);
    std::memset(tensor.data(), byte, tensor.get_byte_size());
    return std::make_shared<op::v0::Constant>(tensor);
}

// How the zero point of a compressed weight is stored.
enum class Zp { PerGroup, MidScalar, None };

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
    int64_t reduce_axis = -2;     // branch axis
    size_t leading = 2;           // dynamic leading dimensions of the activations
    bool gate_to_result = false;  // the write gate is a model output itself, not an input of a later op
    bool share_gate = false;      // another model output reads the write gate's Sigmoid
    // Errors of the 1/C scalings: the effective scale is `factor / C`.
    float low_scale_factor = 1.0F;
    float gate_scale_factor = 1.0F;
    size_t gain_size = CD;                    // elements of the norm gain; C*D is what the op takes
    bool gain_plus_one = false;               // the gain is `w + 1` computed in the graph (an unfolded checkpoint)
    size_t gain_leading_ones = 0;             // the gain is stored as [1, .., 1, gain_size], as checkpoints export it
    size_t inject_rows = C;                   // rows of W_inject; the op takes C
    element::Type scale_type = element::f32;  // the stored type of the 1/C scalars
    int64_t rms_axis = -1;                    // axis of a fused RMS; the op normalizes the last one
};

// The RMS gamma of the FusedRms form: [C, D] with `gain_leading_ones` ones in front.
Shape getGamma(const ReadOptions& opt) {
    Shape shape(opt.gain_leading_ones, 1);
    shape.push_back(C);
    shape.push_back(D);
    return shape;
}

// The projection weight of a GEMM: [N, K] stored with transpose_b.
Output<Node> weight(size_t n, size_t k, const ReadOptions& opt) {
    return opt.compressed ? compressed_weight(n, k, opt.zp) : filled_f32(Shape{n, k}, 0.01F)->output(0);
}

// MatMul(x, W^T) with its 1/C applied the way the options say.
Output<Node> scaled_projection(const Output<Node>& x,
                               size_t n,
                               size_t k,
                               Scale scale,
                               float factor,
                               const ReadOptions& opt) {
    const auto w = weight(n, k, opt);
    switch (scale) {
    case Scale::Folded: {
        // 1/C folded into the weights (plain weights only).
        return std::make_shared<op::v0::MatMul>(
            x,
            std::make_shared<op::v1::Multiply>(w, scalar(factor / C, opt.scale_type)),
            false,
            true);
    }
    case Scale::Divide:
        return std::make_shared<op::v1::Divide>(std::make_shared<op::v0::MatMul>(x, w, false, true),
                                                scalar(static_cast<float>(C) / factor, opt.scale_type));
    case Scale::Multiply:
    default:
        return std::make_shared<op::v1::Multiply>(std::make_shared<op::v0::MatMul>(x, w, false, true),
                                                  scalar(factor / C, opt.scale_type));
    }
}

// The per-branch normalized, flattened activations [.., C*D].
Output<Node> getNormalizedFlat(const ReadOptions& opt, const Output<Node>& residual, const Output<Node>& gain) {
    const auto split = keep_rows(opt.leading, {static_cast<int64_t>(C), static_cast<int64_t>(D)});
    const auto flat = keep_rows(opt.leading, {static_cast<int64_t>(CD)});
    auto r3 = reshape_to(residual, split);
    switch (opt.norm) {
    case Norm::FusedRms:
        return reshape_to(std::make_shared<ov::op::internal::RMS>(r3,
                                                                  filled_f32(getGamma(opt), 1.0F),
                                                                  EPS,
                                                                  element::dynamic,
                                                                  opt.rms_axis),
                          flat);
    case Norm::GammalessRms:
        return std::make_shared<op::v1::Multiply>(
            reshape_to(std::make_shared<ov::op::internal::RMS>(r3, EPS, element::dynamic, opt.rms_axis), flat),
            gain);
    case Norm::GammalessRmsNoGain:
        return reshape_to(std::make_shared<ov::op::internal::RMS>(r3, EPS, element::dynamic, opt.rms_axis), flat);
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
// With `share_gate`, an extra Result of the Sigmoid is appended to `extra_results`.
Output<Node> getWriteGate(const ReadOptions& opt, const Output<Node>& rbar_flat, ResultVector& extra_results) {
    auto gate = std::make_shared<op::v0::Sigmoid>(
        scaled_projection(rbar_flat, opt.inject_rows, CD, opt.gate_scale, opt.gate_scale_factor, opt));
    if (opt.share_gate) {
        extra_results.push_back(std::make_shared<op::v0::Result>(gate));
    }
    return std::make_shared<op::v1::Multiply>(gate, scalar(2.0F));
}

// The write gate as the graph uses it: split to [.., C, 1] for the WRITE that follows (the common form), or a model
// output itself.
Output<Node> gate_consumer(const ReadOptions& opt, const Output<Node>& gate) {
    if (opt.gate_to_result) {
        return gate;
    }
    return std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
}

// The norm gain input of the model: [gain_size] with `gain_leading_ones` ones in front.
std::shared_ptr<op::v0::Parameter> getGain(const ReadOptions& opt) {
    Shape shape(opt.gain_leading_ones, 1);
    shape.push_back(opt.gain_size);
    return std::make_shared<op::v0::Parameter>(element::f32, shape);
}

// Canonical READ graph of the layer, decomposed as checkpoints export it.
std::shared_ptr<Model> getReadModel(const ReadOptions& opt) {
    auto residual =
        std::make_shared<op::v0::Parameter>(element::f32, with_last(PartialShape::dynamic(opt.leading), CD));
    auto gain = getGain(opt);
    const auto split = keep_rows(opt.leading, {static_cast<int64_t>(C), static_cast<int64_t>(D)});
    Output<Node> gain_value = gain;
    if (opt.gain_plus_one) {
        gain_value = std::make_shared<op::v1::Add>(gain, scalar(1.0F));
    }
    const auto rbar_flat = getNormalizedFlat(opt, residual, gain_value);

    auto act =
        std::make_shared<op::v4::Swish>(scaled_projection(rbar_flat, L, CD, opt.low_scale, opt.low_scale_factor, opt));
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
            mixed = std::make_shared<op::v1::Multiply>(mixed, scalar(1.0F / C, opt.scale_type));
        }
    }

    ResultVector results{std::make_shared<op::v0::Result>(mixed)};
    if (opt.with_gate) {
        ResultVector extra;
        results.push_back(std::make_shared<op::v0::Result>(gate_consumer(opt, getWriteGate(opt, rbar_flat, extra))));
        results.insert(results.end(), extra.begin(), extra.end());
    }
    return std::make_shared<Model>(results, ParameterVector{residual, gain});
}

// The model the fusion must produce for the same options. With `keep_gate_branch` the op has no write-gate input and
// the gate stays decomposed on the normalization, which the original still computes.
std::shared_ptr<Model> getReadModelRef(const ReadOptions& opt, bool keep_gate_branch = false) {
    auto residual =
        std::make_shared<op::v0::Parameter>(element::f32, with_last(PartialShape::dynamic(opt.leading), CD));
    auto gain = getGain(opt);

    Output<Node> gain_flat = gain;
    if (opt.gain_plus_one) {
        // The op takes the computed gain as it is.
        gain_flat = std::make_shared<op::v1::Add>(gain, scalar(1.0F));
    }
    if (opt.norm == Norm::FusedRms) {
        // The pass flattens the RMS gamma [C, D] with a plain Reshape.
        gain_flat = std::make_shared<op::v1::Reshape>(
            filled_f32(getGamma(opt), 1.0F),
            op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(CD)}),
            false);
    } else if (opt.norm == Norm::GammalessRmsNoGain) {
        gain_flat = filled_f32(Shape{CD}, 1.0F);
    } else if (opt.gain_leading_ones > 0) {
        // A gain that is not [CD] is flattened with a plain Reshape.
        gain_flat = std::make_shared<op::v1::Reshape>(
            gain,
            op::v0::Constant::create(element::i64, Shape{1}, {static_cast<int64_t>(CD)}),
            false);
    }

    const bool op_gate = opt.with_gate && !keep_gate_branch;
    OutputVector inputs{residual, gain_flat};
    // The weights are fed as they are in the model, a decompression subgraph of compressed ones included.
    inputs.push_back(weight(L, CD, opt));
    inputs.push_back(weight(CD, L, opt));
    if (op_gate) {
        inputs.push_back(weight(C, CD, opt));
    }

    auto gr = std::make_shared<Gr>(inputs, Gr::Config{Gr::Mode::READ, C, D, L, EPS, op_gate});
    ResultVector results{std::make_shared<op::v0::Result>(gr->output(0))};
    if (op_gate) {
        results.push_back(std::make_shared<op::v0::Result>(gate_consumer(opt, gr->output(1))));
    } else if (opt.with_gate) {
        ResultVector extra;
        results.push_back(std::make_shared<op::v0::Result>(
            gate_consumer(opt, getWriteGate(opt, getNormalizedFlat(opt, residual, gain), extra))));
        results.insert(results.end(), extra.begin(), extra.end());
    }
    return std::make_shared<Model>(results, ParameterVector{residual, gain});
}

// R' = R + Reshape(S * Y), the exporter's form: the Multiply is flattened to [.., C*D] and added to the flat residual.
// `y_rows` are the leading dimensions of Y.
// `shared` makes another model output read the Multiply or its flattened form.
enum class Shared { None, Scaled, ScaledFlat };

std::shared_ptr<Model> getWriteModel(const PartialShape& rows,
                                     const PartialShape& y_rows,
                                     Shared shared = Shared::None,
                                     const element::Type& type = element::f32) {
    auto residual = std::make_shared<op::v0::Parameter>(type, with_last(rows, CD));
    auto y = std::make_shared<op::v0::Parameter>(type, with_last(y_rows, D));
    auto gate = std::make_shared<op::v0::Parameter>(type, with_last(rows, C));

    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto scaled = std::make_shared<op::v1::Multiply>(s3, y3);
    auto flat = reshape_to(scaled, keep_rows(rows.size(), {static_cast<int64_t>(CD)}));
    auto out = std::make_shared<op::v1::Add>(residual, flat);
    ResultVector results{std::make_shared<op::v0::Result>(out)};
    if (shared != Shared::None) {
        results.push_back(
            std::make_shared<op::v0::Result>(shared == Shared::Scaled ? scaled->output(0) : flat->output(0)));
    }
    return std::make_shared<Model>(results, ParameterVector{residual, y, gate});
}

// The same update with the Add in the split [.., C, D] domain: the form the sink produces.
std::shared_ptr<Model> getWriteSplitModel(const PartialShape& rows,
                                          const PartialShape& y_rows,
                                          const element::Type& type = element::f32) {
    auto residual = std::make_shared<op::v0::Parameter>(type, with_last(rows, CD));
    auto y = std::make_shared<op::v0::Parameter>(type, with_last(y_rows, D));
    auto gate = std::make_shared<op::v0::Parameter>(type, with_last(rows, C));

    auto r3 = reshape_to(residual, keep_rows(rows.size(), {static_cast<int64_t>(C), static_cast<int64_t>(D)}));
    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto updated = std::make_shared<op::v1::Add>(r3, std::make_shared<op::v1::Multiply>(s3, y3));
    auto out = reshape_to(updated, keep_rows(rows.size(), {static_cast<int64_t>(CD)}));
    return std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                   ParameterVector{residual, y, gate});
}

// The fused WRITE op.
// With `shared`, the Multiply or its flattened form stays for the other model output that reads it.
std::shared_ptr<Model> getWriteRef(const PartialShape& rows, Shared shared = Shared::None) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, CD));
    auto y = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, D));
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, with_last(rows, C));
    auto gr = std::make_shared<Gr>(OutputVector{residual, y, gate}, Gr::Config{Gr::Mode::WRITE, C, D, 0, 0.0F, false});
    ResultVector results{std::make_shared<op::v0::Result>(gr)};
    if (shared != Shared::None) {
        auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
        auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
        Output<Node> scaled = std::make_shared<op::v1::Multiply>(s3, y3);
        if (shared == Shared::ScaledFlat) {
            scaled = reshape_to(scaled, keep_rows(rows.size(), {static_cast<int64_t>(CD)}));
        }
        results.push_back(std::make_shared<op::v0::Result>(scaled));
    }
    return std::make_shared<Model>(results, ParameterVector{residual, y, gate});
}

}  // namespace

class GatedResidualFusionTests : public TransformationTestsF {
public:
    GatedResidualFusionTests() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

protected:
    virtual bool fuse_write() const {
        return true;
    }

    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::GatedResidualFusion>(fuse_write());
    }
};

// The CPU plugin's choice: the Add of the WRITE pattern is moved, not fused into the op.
class GatedResidualWriteSinkReshapeTests : public GatedResidualFusionTests {
protected:
    bool fuse_write() const override {
        return false;
    }
};

// The forms of the READ pattern the fusion accepts: each case changes one option of the canonical graph.
struct ReadCase {
    std::string name;
    std::function<void(ReadOptions&)> configure;
};

class GatedResidualFusionReadTests : public testing::WithParamInterface<ReadCase>, public GatedResidualFusionTests {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReadCase>& info) {
        return info.param.name;
    }
};

TEST_P(GatedResidualFusionReadTests, Fused) {
    ReadOptions opt;
    GetParam().configure(opt);
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

INSTANTIATE_TEST_SUITE_P(GatedResidualFusion,
                         GatedResidualFusionReadTests,
                         ::testing::Values(ReadCase{"DecomposedNorm", [](ReadOptions&) {}},
                                           ReadCase{"FusedRms",
                                                    [](ReadOptions& o) {
                                                        o.norm = Norm::FusedRms;
                                                    }},
                                           ReadCase{"GammalessRms",
                                                    [](ReadOptions& o) {
                                                        o.norm = Norm::GammalessRms;
                                                    }},
                                           ReadCase{"GammalessRmsWithoutGain",
                                                    [](ReadOptions& o) {
                                                        o.norm = Norm::GammalessRmsNoGain;
                                                    }},
                                           // An unfolded checkpoint: the gain is computed as `w + 1` in the graph.
                                           ReadCase{"DecomposedNormGainPlusOne",
                                                    [](ReadOptions& o) {
                                                        o.gain_plus_one = true;
                                                    }},
                                           ReadCase{"DecomposedNormGainWithLeadingOnes",
                                                    [](ReadOptions& o) {
                                                        o.gain_leading_ones = 2;
                                                    }},
                                           ReadCase{"FusedRmsGammaWithLeadingOnes",
                                                    [](ReadOptions& o) {
                                                        o.norm = Norm::FusedRms;
                                                        o.gain_leading_ones = 2;
                                                    }},
                                           ReadCase{"GammalessRmsGainWithLeadingOnes",
                                                    [](ReadOptions& o) {
                                                        o.norm = Norm::GammalessRms;
                                                        o.gain_leading_ones = 2;
                                                    }},
                                           ReadCase{"ReduceMeanMix",
                                                    [](ReadOptions& o) {
                                                        o.mix = Mix::Mean;
                                                    }},
                                           ReadCase{"DownScaleFoldedIntoWeights",
                                                    [](ReadOptions& o) {
                                                        o.low_scale = Scale::Folded;
                                                    }},
                                           ReadCase{"DownScaleAsDivide",
                                                    [](ReadOptions& o) {
                                                        o.low_scale = Scale::Divide;
                                                    }},
                                           ReadCase{"WriteGate",
                                                    [](ReadOptions& o) {
                                                        o.with_gate = true;
                                                    }},
                                           ReadCase{"WriteGateScaleAsDivide",
                                                    [](ReadOptions& o) {
                                                        o.with_gate = true;
                                                        o.gate_scale = Scale::Divide;
                                                    }},
                                           ReadCase{"WriteGateScaleFoldedIntoWeights",
                                                    [](ReadOptions& o) {
                                                        o.with_gate = true;
                                                        o.gate_scale = Scale::Folded;
                                                    }},
                                           ReadCase{"OneDynamicRowDimension",
                                                    [](ReadOptions& o) {
                                                        o.with_gate = true;
                                                        o.leading = 1;  // [rows, C*D]
                                                    }}),
                         GatedResidualFusionReadTests::getTestCaseName);

// Compressed weights reach the op as their decompression subgraph; ConvertGatedResidualToCompressed unpacks it.
using CompressedSourceParams = std::tuple<Zp, bool>;  // zero point form, write gate

class GatedResidualFusionCompressedSourceTests : public testing::WithParamInterface<CompressedSourceParams>,
                                                 public GatedResidualFusionTests {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CompressedSourceParams>& info) {
        const auto& [zp, with_gate] = info.param;
        const char* zp_name = zp == Zp::PerGroup    ? "ZeroPointPerGroup"
                              : zp == Zp::MidScalar ? "MidScalarZeroPoint"
                                                    : "NoZeroPoint";
        return std::string(zp_name) + (with_gate ? "_WriteGate" : "_NoWriteGate");
    }
};

TEST_P(GatedResidualFusionCompressedSourceTests, Read_WeightsStayDecompressionSubgraphs) {
    const auto& [zp, with_gate] = GetParam();
    ReadOptions opt;
    opt.compressed = true;
    opt.zp = zp;
    opt.with_gate = with_gate;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

INSTANTIATE_TEST_SUITE_P(GatedResidualFusion,
                         GatedResidualFusionCompressedSourceTests,
                         ::testing::Combine(::testing::Values(Zp::PerGroup, Zp::MidScalar, Zp::None),
                                            ::testing::Bool()),
                         GatedResidualFusionCompressedSourceTests::getTestCaseName);

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

// A rank-1 residual [C*D] has no rows; the op needs at least two dimensions.
TEST_F(GatedResidualFusionTests, Read_RankOneResidualIsNotFused) {
    ReadOptions opt;
    opt.leading = 0;
    model = getReadModel(opt);
}

// The op has one friendly name, the one of the mix output; a gate that is itself a model output would lose its name.
TEST_F(GatedResidualFusionTests, Read_WriteGateThatIsAModelOutputStaysDecomposed) {
    ReadOptions opt;
    opt.with_gate = true;
    opt.gate_to_result = true;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt, /*keep_gate_branch=*/true);
}

// The op applies 1/C itself, so a scale that is not 1/C is model semantics the fusion cannot absorb.
TEST_F(GatedResidualFusionTests, Read_DownScaleOfAnotherValueIsNotFused) {
    ReadOptions opt;
    opt.low_scale_factor = 1.5F;
    model = getReadModel(opt);
}

TEST_F(GatedResidualFusionTests, Read_DownScaleFoldedIntoWeightsOfAnotherValueIsNotFused) {
    ReadOptions opt;
    opt.low_scale = Scale::Folded;
    opt.low_scale_factor = 1.5F;
    model = getReadModel(opt);
}

// The gate stays decomposed on the normalization, which the fused READ still computes.
TEST_F(GatedResidualFusionTests, Read_WriteGateScaleOfAnotherValueStaysDecomposed) {
    ReadOptions opt;
    opt.with_gate = true;
    opt.gate_scale_factor = 1.5F;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt, /*keep_gate_branch=*/true);
}

// The gate's Sigmoid is also read elsewhere: the branch is not private to the fused op.
TEST_F(GatedResidualFusionTests, Read_WriteGateWithAnotherConsumerStaysDecomposed) {
    ReadOptions opt;
    opt.with_gate = true;
    opt.share_gate = true;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt, /*keep_gate_branch=*/true);
}

// The op takes the expanded [C*D] gain; a [1] one would broadcast in the graph but is not that.
TEST_F(GatedResidualFusionTests, Read_GainOfWrongSizeIsNotFused) {
    ReadOptions opt;
    opt.gain_size = 1;
    model = getReadModel(opt);
}

// The scales are stored in a narrower type than the compute one, where 1/C is rounded.
TEST_F(GatedResidualFusionTests, Read_ScalesStoredInBf16AreFused) {
    ReadOptions opt;
    opt.with_gate = true;
    opt.scale_type = element::bf16;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
}

TEST_F(GatedResidualFusionTests, Read_ScalesStoredInBf16OfAnotherValueAreNotFused) {
    ReadOptions opt;
    opt.scale_type = element::bf16;
    opt.low_scale_factor = 1.5F;
    model = getReadModel(opt);
}

// W_inject must be [C, C*D]: the op produces one gate per branch.
TEST_F(GatedResidualFusionTests, Read_WriteGateOfWrongWidthStaysDecomposed) {
    ReadOptions opt;
    opt.with_gate = true;
    opt.inject_rows = C + 1;
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt, /*keep_gate_branch=*/true);
}

// An RMS over another axis (RMSFusion's feature-axis form) is not the normalization over D.
TEST_F(GatedResidualFusionTests, Read_FusedRmsOverAnotherAxisIsNotFused) {
    ReadOptions opt;
    opt.norm = Norm::FusedRms;
    opt.rms_axis = 1;
    model = getReadModel(opt);
}

TEST_F(GatedResidualFusionTests, Read_GammalessRmsOverAnotherAxisIsNotFused) {
    ReadOptions opt;
    opt.norm = Norm::GammalessRms;
    opt.rms_axis = 1;
    model = getReadModel(opt);
}

TEST_F(GatedResidualFusionTests, Read_AppliedTwiceGivesTheSameGraph) {
    ReadOptions opt;
    opt.with_gate = true;
    manager.register_pass<pass::GatedResidualFusion>();
    model = getReadModel(opt);
    model_ref = getReadModelRef(opt);
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

// The row shapes the sink accepts: dynamic or static, any number of them, and rows of Y that may broadcast (the rewrite
// only appends the [C, D] split to them, which keeps the broadcasting as it is).
struct WriteRowsCase {
    std::string name;
    PartialShape rows;
    PartialShape y_rows;
};

class GatedResidualWriteSinkReshapeRowsTests : public testing::WithParamInterface<WriteRowsCase>,
                                               public GatedResidualWriteSinkReshapeTests {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<WriteRowsCase>& info) {
        return info.param.name;
    }
};

TEST_P(GatedResidualWriteSinkReshapeRowsTests, AddMovesNextToTheMultiply) {
    const auto& [name, rows, y_rows] = GetParam();
    model = getWriteModel(rows, y_rows);
    model_ref = getWriteSplitModel(rows, y_rows);
}

INSTANTIATE_TEST_SUITE_P(
    GatedResidualWriteSinkReshape,
    GatedResidualWriteSinkReshapeRowsTests,
    ::testing::Values(WriteRowsCase{"DynamicRows", PartialShape::dynamic(2), PartialShape::dynamic(2)},
                      WriteRowsCase{"PagedAttentionRows", PartialShape::dynamic(1), PartialShape::dynamic(1)},
                      WriteRowsCase{"ThreeRowDimensions", PartialShape::dynamic(3), PartialShape::dynamic(3)},
                      WriteRowsCase{"StaticRows", PartialShape{2, 3}, PartialShape{2, 3}},
                      WriteRowsCase{"DynamicRowsBroadcastY", PartialShape::dynamic(2), PartialShape{1, 1}},
                      WriteRowsCase{"StaticRowsBroadcastY", PartialShape{2, 3}, PartialShape{1, 3}}),
    GatedResidualWriteSinkReshapeRowsTests::getTestCaseName);

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

// The rewrite moves the Multiply under the Add: another reader of it or of its flattened form would see a changed
// graph.
TEST_F(GatedResidualWriteSinkReshapeTests, Write_ScaledValueWithAnotherConsumerIsNotRewritten) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2), Shared::Scaled);
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_FlattenedValueWithAnotherConsumerIsNotRewritten) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2), Shared::ScaledFlat);
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_AppliedTwiceGivesTheSameGraph) {
    manager.register_pass<pass::GatedResidualWriteSinkReshape>();
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    model_ref = getWriteSplitModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
}

TEST_F(GatedResidualWriteSinkReshapeTests, Write_CallbackKeepsTheOriginalGraph) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    manager.get_pass_config()->set_callback<pass::GatedResidualWriteSinkReshape>(
        [](const std::shared_ptr<const Node>&) {
            return true;
        });
}

TEST_F(GatedResidualFusionTests, Write_ExporterFormIsFusedByDefault) {
    const auto rows = symbolic_rows(2);
    model = getWriteModel(rows, rows);
    model_ref = getWriteRef(rows);
}

TEST_F(GatedResidualFusionTests, Write_SplitFormIsFusedByDefault) {
    const auto rows = symbolic_rows(2);
    model = getWriteSplitModel(rows, rows);
    model_ref = getWriteRef(rows);
}

// The sink leaves a Multiply that another output reads, but the op still replaces the Add: that Multiply stays for its
// reader.
TEST_F(GatedResidualFusionTests, Write_ScaledValueWithAnotherConsumerIsFusedWithoutTheSink) {
    const auto rows = symbolic_rows(2);
    model = getWriteModel(rows, rows, Shared::Scaled);
    model_ref = getWriteRef(rows, Shared::Scaled);
}

TEST_F(GatedResidualFusionTests, Write_FlattenedValueWithAnotherConsumerIsFusedWithoutTheSink) {
    const auto rows = symbolic_rows(2);
    model = getWriteModel(rows, rows, Shared::ScaledFlat);
    model_ref = getWriteRef(rows, Shared::ScaledFlat);
}

TEST_F(GatedResidualFusionTests, Write_CallbackKeepsTheSplitForm) {
    const auto rows = symbolic_rows(2);
    model = getWriteModel(rows, rows);
    model_ref = getWriteSplitModel(rows, rows);
    manager.get_pass_config()->set_callback<pass::GatedResidualWriteFusion>([](const std::shared_ptr<const Node>&) {
        return true;
    });
}

// Dynamic rows of the residual, Y and the gate without a known relation may broadcast at runtime, which the fused op
// does not model; the sink still moves the Add.
TEST_F(GatedResidualFusionTests, Write_DynamicRowsWithoutSharedSymbolsAreNotFused) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    model_ref = getWriteSplitModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
}

TEST_F(GatedResidualFusionTests, Write_RowsThatMayBroadcastAreNotFused) {
    // The fused op indexes Y and the gate per residual row, so rows that may broadcast stay a plain graph; the sink
    // still moves the Add.
    const auto rows = symbolic_rows(2);
    model = getWriteModel(rows, PartialShape{1, 1});
    model_ref = getWriteSplitModel(rows, PartialShape{1, 1});
}

// The gate and Y reach the Multiply through Reshapes that change their rank: [2, 3, C] -> [6, C, 1]. The op needs one
// rank for the residual, Y and the gate, so the graph is left alone instead of failing the op's validation.
TEST_F(GatedResidualFusionTests, Write_RankChangingGateAndYReshapesAreNotFused) {
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, static_cast<int64_t>(CD)});
    auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, static_cast<int64_t>(D)});
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, static_cast<int64_t>(C)});
    auto s3 = reshape_to(gate, {6, static_cast<int64_t>(C), 1});
    auto y3 = reshape_to(y, {6, 1, static_cast<int64_t>(D)});
    auto flat = reshape_to(std::make_shared<op::v1::Multiply>(s3, y3), {6, static_cast<int64_t>(CD)});
    auto out = std::make_shared<op::v1::Add>(residual, flat);
    model = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                    ParameterVector{residual, y, gate});
    manager.get_pass_config()->disable<pass::GatedResidualWriteSinkReshape>();
}

TEST_F(GatedResidualFusionTests, Write_RankChangingResidualSplitIsNotFused) {
    // The split of the residual [2, 3, C*D] -> [6, C, D] drops a leading dimension, in the split-domain form.
    auto residual = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, 3, static_cast<int64_t>(CD)});
    auto y = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, static_cast<int64_t>(D)});
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{6, static_cast<int64_t>(C)});
    auto r3 = reshape_to(residual, {6, static_cast<int64_t>(C), static_cast<int64_t>(D)});
    auto s3 = std::make_shared<op::v0::Unsqueeze>(gate, op::v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto y3 = std::make_shared<op::v0::Unsqueeze>(y, op::v0::Constant::create(element::i64, Shape{1}, {-2}));
    auto updated = std::make_shared<op::v1::Add>(r3, std::make_shared<op::v1::Multiply>(s3, y3));
    auto out = reshape_to(updated, {2, 3, static_cast<int64_t>(CD)});
    model = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(out)},
                                    ParameterVector{residual, y, gate});
}

TEST_F(GatedResidualFusionTests, Write_BothWritePassesDisabledLeavesTheGraph) {
    model = getWriteModel(PartialShape::dynamic(2), PartialShape::dynamic(2));
    manager.get_pass_config()->disable<pass::GatedResidualWriteFusion>();
    manager.get_pass_config()->disable<pass::GatedResidualWriteSinkReshape>();
}

// A rank-1 residual [C*D] has no rows; the op needs at least two dimensions, so the op is not built (the sink still
// moves the Add).
TEST_F(GatedResidualFusionTests, Write_RankOneResidualIsNotFused) {
    model = getWriteModel(PartialShape{}, PartialShape{});
    model_ref = getWriteSplitModel(PartialShape{}, PartialShape{});
}

// The op takes floating-point activations; an integer graph that has the shape of the pattern is left alone (the sink
// still moves the Add).
TEST_F(GatedResidualFusionTests, Write_IntegerActivationsAreNotFused) {
    const PartialShape rows{4};
    model = getWriteModel(rows, rows, Shared::None, element::i32);
    model_ref = getWriteSplitModel(rows, rows, element::i32);
}
