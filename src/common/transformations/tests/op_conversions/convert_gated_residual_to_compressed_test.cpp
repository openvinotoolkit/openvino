// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gated_residual_to_compressed.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov;

namespace {

constexpr size_t C = 4;
constexpr size_t D = 8;
constexpr size_t L = 6;
constexpr size_t CD = C * D;
constexpr float EPS = 1e-6F;

using Gr = op::internal::GatedResidual;

// How the zero point of a compressed weight is stored: per group, a scalar, or none.
enum class Zp { PerGroup, Scalar, None };

struct ModelOptions {
    element::Type weights_type = element::u4;
    element::Type data_type = element::f32;  // the activations, the gain and the decompressed weights
    size_t groups = 2;                       // scale columns G; 1 is per-channel
    Zp zp = Zp::PerGroup;
    Zp up_zp = Zp::PerGroup;  // the zero point form of W_up when `different_forms`
    bool different_forms = false;
    bool compressed = true;  // false: the weights are plain constants
    // The stored type of the scales; anything but `data_type` is converted to it in the graph.
    element::Type scale_type = element::dynamic;
    // The stored type of the zero points; the weights type if dynamic, anything but `data_type` is converted to it.
    element::Type zp_type = element::dynamic;
    bool per_input_channel_scale = false;  // a [1, K] scale (`groups` == 1 only)
    bool per_tensor_scale = false;         // a [1, 1] scale instead of one per channel (`groups` == 1 only)
    bool transposed = false;               // the weights are stored [K, N] and transposed back (`groups` == 1 only)
    bool with_gate = false;
    std::optional<Zp> inject_zp;  // the zero point form of W_inject when it differs from the others
    bool weightless = false;      // the stored weights are linked to their file for weightless caching
};

// A constant of `type` whose every byte is `byte`.
std::shared_ptr<op::v0::Constant> bytes_constant(const element::Type& type, const Shape& shape, uint8_t byte) {
    Tensor tensor(type, shape);
    std::memset(tensor.data(), byte, tensor.get_byte_size());
    return std::make_shared<op::v0::Constant>(tensor);
}

std::shared_ptr<op::v0::Constant> float_constant(const element::Type& type, const Shape& shape, float value) {
    return op::v0::Constant::create(type, shape, std::vector<float>(shape_size(shape), value));
}

uint8_t weights_byte(const element::Type& type) {
    return type.bitwidth() == 4 ? 0x5A : 0x7B;
}

// The link of the stored weights to their file: the attribute that has to follow the weights to the op.
void set_weightless_attr(const std::shared_ptr<Node>& weights, size_t n, size_t k, const ModelOptions& opt) {
    if (opt.weightless) {
        weights->get_rt_info()[WeightlessCacheAttribute::get_type_info_static()] =
            WeightlessCacheAttribute(n * k, n, opt.weights_type);
    }
}

// NNCF-style compressed weight [N, G, K/G] -> Convert [-> Subtract(zp)] -> Multiply(scale) [-> Reshape [N, K]]; per
// channel (G = 1) the weights are [N, K] and there is no Reshape.
Output<Node> compressed_weight(size_t n, size_t k, const ModelOptions& opt, Zp zp) {
    if (!opt.compressed) {
        return float_constant(opt.data_type, Shape{n, k}, 0.01F);
    }
    const bool grouped = opt.groups > 1;
    Shape weights_shape = grouped ? Shape{n, opt.groups, k / opt.groups} : Shape{n, k};
    Shape scale_shape = grouped ? Shape{n, opt.groups, 1} : Shape{n, 1};
    if (opt.transposed) {
        weights_shape = Shape{k, n};
        scale_shape = Shape{1, n};
    } else if (opt.per_tensor_scale) {
        scale_shape = Shape{1, 1};
    } else if (opt.per_input_channel_scale) {
        scale_shape = Shape{1, k};
    }
    auto stored = bytes_constant(opt.weights_type, weights_shape, weights_byte(opt.weights_type));
    set_weightless_attr(stored, n, k, opt);
    Output<Node> value = std::make_shared<op::v0::Convert>(stored, opt.data_type);
    if (zp != Zp::None) {
        const Shape zp_shape = zp == Zp::PerGroup ? scale_shape : Shape{1};
        const bool float_zp = opt.zp_type != element::dynamic;
        std::shared_ptr<Node> zp_const = float_zp ? std::shared_ptr<Node>(float_constant(opt.zp_type, zp_shape, 2.0F))
                                                  : bytes_constant(opt.weights_type, zp_shape, 0x33);
        if (zp_const->get_output_element_type(0) != opt.data_type) {
            zp_const = std::make_shared<op::v0::Convert>(zp_const, opt.data_type);
        }
        value = std::make_shared<op::v1::Subtract>(value, zp_const);
    }
    Output<Node> scale = float_constant(opt.data_type, scale_shape, 0.02F);
    if (opt.scale_type != element::dynamic) {
        scale = std::make_shared<op::v0::Convert>(float_constant(opt.scale_type, scale_shape, 2.0F), opt.data_type);
    }
    Output<Node> result = std::make_shared<op::v1::Multiply>(value, scale);
    if (opt.transposed) {
        result = std::make_shared<op::v1::Transpose>(result, op::v0::Constant::create(element::i64, Shape{2}, {1, 0}));
    }
    if (grouped) {
        result = std::make_shared<op::v1::Reshape>(
            result,
            op::v0::Constant::create(element::i64, Shape{2}, {static_cast<int64_t>(n), static_cast<int64_t>(k)}),
            false);
    }
    return result;
}

// The op as the fusion leaves it: the weights are decompression subgraphs.
std::shared_ptr<Model> getModel(const ModelOptions& opt) {
    auto residual = std::make_shared<op::v0::Parameter>(opt.data_type, PartialShape{-1, static_cast<int64_t>(CD)});
    auto gain = std::make_shared<op::v0::Parameter>(opt.data_type, Shape{CD});
    const auto up_zp = opt.different_forms ? opt.up_zp : opt.zp;
    OutputVector inputs{residual, gain, compressed_weight(L, CD, opt, opt.zp), compressed_weight(CD, L, opt, up_zp)};
    if (opt.with_gate) {
        inputs.push_back(compressed_weight(C, CD, opt, opt.inject_zp.value_or(opt.zp)));
    }
    auto gr = std::make_shared<Gr>(inputs, Gr::Config{Gr::Mode::READ, C, D, L, EPS, opt.with_gate});
    ResultVector results{std::make_shared<op::v0::Result>(gr->output(0))};
    if (opt.with_gate) {
        results.push_back(std::make_shared<op::v0::Result>(gr->output(1)));
    }
    return std::make_shared<Model>(results, ParameterVector{residual, gain});
}

// The model the pass must produce: the stored constants, scales and zero points ([N, G] or the stored single element).
std::shared_ptr<Model> getModelRef(const ModelOptions& opt) {
    auto residual = std::make_shared<op::v0::Parameter>(opt.data_type, PartialShape{-1, static_cast<int64_t>(CD)});
    auto gain = std::make_shared<op::v0::Parameter>(opt.data_type, Shape{CD});
    const element::Type type = opt.weights_type;
    const uint8_t byte = weights_byte(type);
    const bool with_zp = opt.zp != Zp::None;

    struct Parts {
        Output<Node> weights;
        Output<Node> scale;
        Output<Node> zp;
    };
    const auto parts = [&](size_t n, size_t k) {
        Parts p;
        p.weights = bytes_constant(type, Shape{n, k}, byte);
        set_weightless_attr(p.weights.get_node_shared_ptr(), n, k, opt);
        p.scale = float_constant(opt.data_type, opt.per_tensor_scale ? Shape{1, 1} : Shape{n, opt.groups}, 0.02F);
        if (with_zp) {
            p.zp = bytes_constant(opt.weights_type, opt.zp == Zp::Scalar ? Shape{1} : Shape{n, opt.groups}, 0x33);
        }
        return p;
    };
    const auto down = parts(L, CD);
    const auto up = parts(CD, L);
    OutputVector inputs{residual, gain, down.weights, up.weights, down.scale, up.scale};
    if (with_zp) {
        inputs.insert(inputs.end(), {down.zp, up.zp});
    }
    if (opt.with_gate) {
        const auto inject = parts(C, CD);
        inputs.insert(inputs.end(), {inject.weights, inject.scale});
        if (with_zp) {
            inputs.push_back(inject.zp);
        }
    }
    auto gr = std::make_shared<Gr>(inputs, Gr::Config{Gr::Mode::READ, C, D, L, EPS, opt.with_gate});
    ResultVector results{std::make_shared<op::v0::Result>(gr->output(0))};
    if (opt.with_gate) {
        results.push_back(std::make_shared<op::v0::Result>(gr->output(1)));
    }
    return std::make_shared<Model>(results, ParameterVector{residual, gain});
}

}  // namespace

class ConvertGatedResidualToCompressedTests : public TransformationTestsF {
public:
    ConvertGatedResidualToCompressedTests() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }

protected:
    virtual element::TypeVector supported_weights_types() const {
        return {element::u8, element::i8, element::u4, element::i4};
    }
    virtual pass::ConvertGatedResidualToCompressed::SupportsPredicate supports_config() const {
        return nullptr;
    }

    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<pass::ConvertGatedResidualToCompressed>(supported_weights_types(), supports_config());
    }
};

using ConvertParams = std::tuple<element::Type, size_t, Zp, bool>;  // weights type, groups, zero point, write gate

class ConvertGatedResidualToCompressedParamTests : public testing::WithParamInterface<ConvertParams>,
                                                   public ConvertGatedResidualToCompressedTests {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertParams>& info) {
        const auto& [type, groups, zp, with_gate] = info.param;
        const char* zp_name = zp == Zp::PerGroup ? "ZeroPoint" : zp == Zp::Scalar ? "ScalarZeroPoint" : "NoZeroPoint";
        return type.to_string() + "_G" + std::to_string(groups) + "_" + zp_name + (with_gate ? "_WriteGate" : "");
    }
};

TEST_P(ConvertGatedResidualToCompressedParamTests, WeightsAreFedCompressed) {
    const auto& [type, groups, zp, with_gate] = GetParam();
    ModelOptions opt;
    opt.weights_type = type;
    opt.groups = groups;
    opt.zp = zp;
    opt.with_gate = with_gate;
    model = getModel(opt);
    model_ref = getModelRef(opt);
}

INSTANTIATE_TEST_SUITE_P(ConvertGatedResidualToCompressed,
                         ConvertGatedResidualToCompressedParamTests,
                         ::testing::Combine(::testing::Values(element::u4, element::u8),
                                            ::testing::Values(size_t{1}, size_t{2}),
                                            ::testing::Values(Zp::PerGroup, Zp::Scalar, Zp::None),
                                            ::testing::Bool()),
                         ConvertGatedResidualToCompressedParamTests::getTestCaseName);

// The link of grouped weights to their file survives the regrouping: the weights of the op are new constants.
TEST_F(ConvertGatedResidualToCompressedTests, WeightlessCacheAttributeFollowsTheWeights) {
    ModelOptions opt;
    opt.weightless = true;
    opt.with_gate = true;
    model = getModel(opt);
    model_ref = getModelRef(opt);
}

// The element types of the activations, the gain and the scales follow the model.
TEST_F(ConvertGatedResidualToCompressedTests, F16ActivationsKeepTheirType) {
    ModelOptions opt;
    opt.data_type = element::f16;
    opt.with_gate = true;
    model = getModel(opt);
    model_ref = getModelRef(opt);
}

TEST_F(ConvertGatedResidualToCompressedTests, AppliedTwiceGivesTheSameGraph) {
    ModelOptions opt;
    opt.with_gate = true;
    manager.register_pass<pass::ConvertGatedResidualToCompressed>(supported_weights_types(), supports_config());
    model = getModel(opt);
    model_ref = getModelRef(opt);
}

TEST_F(ConvertGatedResidualToCompressedTests, FloatWeightsAreNotConverted) {
    ModelOptions opt;
    opt.compressed = false;
    model = getModel(opt);
}

// W_down with zero points and W_up without them: the zero points of the op are all-or-nothing.
TEST_F(ConvertGatedResidualToCompressedTests, WeightsOfDifferentFormsAreNotConverted) {
    ModelOptions opt;
    opt.different_forms = true;
    opt.up_zp = Zp::None;
    model = getModel(opt);
}

// W_inject with other zero points than the rest: the form is shared by all weights of the op.
TEST_F(ConvertGatedResidualToCompressedTests, WriteGateWeightsOfAnotherFormAreNotConverted) {
    ModelOptions opt;
    opt.with_gate = true;
    opt.inject_zp = Zp::None;
    model = getModel(opt);
}

// The op takes floating-point scales; an integer one kept behind a Convert is not a compressed form it can run.
TEST_F(ConvertGatedResidualToCompressedTests, IntegerScalesAreNotConverted) {
    ModelOptions opt;
    opt.scale_type = element::i8;
    model = getModel(opt);
}

// The zero points of the op have the weight type: a floating-point one, behind a Convert or used as it is, is not
// converted.
TEST_F(ConvertGatedResidualToCompressedTests, FloatZeroPointBehindConvertIsNotConverted) {
    ModelOptions opt;
    opt.zp_type = element::f16;
    model = getModel(opt);
}

TEST_F(ConvertGatedResidualToCompressedTests, FloatZeroPointWithoutConvertIsNotConverted) {
    ModelOptions opt;
    opt.zp_type = element::f32;
    opt.with_gate = true;
    model = getModel(opt);
}

// The scales of the op are per output channel [N, G]: a per input channel [1, K] one is not converted.
TEST_F(ConvertGatedResidualToCompressedTests, PerInputChannelScaleIsNotConverted) {
    ModelOptions opt;
    opt.groups = 1;
    opt.zp = Zp::None;
    opt.per_input_channel_scale = true;
    model = getModel(opt);
}

// A single-element zero point (per-tensor, e.g. the legacy symmetric u4 with zp = 8) is kept as it is, as by the
// FullyConnected conversion.
TEST_F(ConvertGatedResidualToCompressedTests, SingleElementZeroPointIsKept) {
    ModelOptions opt;
    opt.zp = Zp::Scalar;
    opt.with_gate = true;
    model = getModel(opt);
    model_ref = getModelRef(opt);
}

// The stored weights are consumed as [N, K]: a Transpose in the decompression is not expected (the plugin fuses it into
// the MatMul first) and has no counterpart in the op.
TEST_F(ConvertGatedResidualToCompressedTests, TransposedWeightsAreNotConverted) {
    ModelOptions opt;
    opt.groups = 1;
    opt.zp = Zp::None;
    opt.transposed = true;
    model = getModel(opt);
}

// One scale for the whole tensor is kept as it is.
TEST_F(ConvertGatedResidualToCompressedTests, PerTensorScaleIsKept) {
    ModelOptions opt;
    opt.groups = 1;
    opt.zp = Zp::Scalar;
    opt.per_tensor_scale = true;
    model = getModel(opt);
    model_ref = getModelRef(opt);
}

class ConvertGatedResidualToCompressedUnsupportedTypeTests : public ConvertGatedResidualToCompressedTests {
protected:
    element::TypeVector supported_weights_types() const override {
        return {element::u8, element::i8};
    }
};

TEST_F(ConvertGatedResidualToCompressedUnsupportedTypeTests, WeightsOfAnotherTypeAreNotConverted) {
    model = getModel(ModelOptions{});
}

class ConvertGatedResidualToCompressedRejectedTests : public ConvertGatedResidualToCompressedTests {
protected:
    pass::ConvertGatedResidualToCompressed::SupportsPredicate supports_config() const override {
        return [](const std::shared_ptr<Gr>&) {
            return false;
        };
    }
};

TEST_F(ConvertGatedResidualToCompressedRejectedTests, BackendRejectsTheCompressedOp) {
    model = getModel(ModelOptions{});
}
