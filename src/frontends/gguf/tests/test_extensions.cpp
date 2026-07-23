// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for FrontEnd::add_extension (the extension-passing path in frontend.cpp).
//
// A ConversionExtension registers a custom translator for a ggml op name; the frontend
// merges it into the op table (overriding a built-in translator on name collision, or
// adding a translator for an otherwise unsupported op). The converter receives an
// ov::frontend::NodeContext, which the gguf NodeContext derives from.

#include <openvino/op/abs.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>

#include "op_test_utils.hpp"
#include "openvino/frontend/extension/conversion.hpp"

using namespace ov_gguf_test;

namespace {

// A ConversionExtension whose converter emits Negative(in0) for whatever op it is
// registered against.
std::shared_ptr<ov::frontend::ConversionExtension> make_negate_ext(const std::string& op_type) {
    return std::make_shared<ov::frontend::ConversionExtension>(
        op_type,
        [](const ov::frontend::NodeContext& context) -> ov::OutputVector {
            return {std::make_shared<ov::op::v0::Negative>(context.get_input(0))};
        });
}

}  // namespace

// A ConversionExtension registered for a built-in op name overrides the built-in
// translator: GGML_OP_SCALE normally does in*scale+bias, but here it must negate.
TEST(GGUFExtensions, ConversionExtensionOverridesBuiltin) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_SCALE")
                     .input("x", ov::element::f32, {2, 4})
                     .output("out", ov::element::f32, {2, 4})
                     .attr<float>("scale", 2.0f)
                     .attr<float>("bias", 0.0f)
                     .build_with_extensions({make_negate_ext("GGML_OP_SCALE")});

    std::vector<float> x{1, -2, 3, -4, 5, -6, 7, -8};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({2, 4}, x)}});

    std::vector<float> expected(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        expected[i] = -x[i];  // Negative, not the built-in scale
    expect_near(out, expected);
}

// A ConversionExtension can add a translator for an op the frontend does not support
// out of the box (here a made-up "GGML_OP_CUSTOM_NEGATE").
TEST(GGUFExtensions, ConversionExtensionAddsNewOp) {
    auto model = SingleOpBuilder()
                     .op("GGML_OP_CUSTOM_NEGATE")
                     .input("x", ov::element::f32, {3})
                     .output("out", ov::element::f32, {3})
                     .build_with_extensions({make_negate_ext("GGML_OP_CUSTOM_NEGATE")});

    std::vector<float> x{1, -2, 3};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({3}, x)}});
    expect_near(out, {-1, 2, -3});
}

// Two extensions registered together are both applied.
TEST(GGUFExtensions, MultipleConversionExtensions) {
    auto abs_ext = std::make_shared<ov::frontend::ConversionExtension>(
        "GGML_OP_CUSTOM_ABS",
        [](const ov::frontend::NodeContext& context) -> ov::OutputVector {
            return {std::make_shared<ov::op::v0::Abs>(context.get_input(0))};
        });

    auto model = SingleOpBuilder()
                     .op("GGML_OP_CUSTOM_ABS")
                     .input("x", ov::element::f32, {4})
                     .output("out", ov::element::f32, {4})
                     .build_with_extensions({make_negate_ext("GGML_OP_CUSTOM_NEGATE"), abs_ext});

    std::vector<float> x{1, -2, 3, -4};
    auto out = run_on_cpu(model, {{"x", make_f32_tensor({4}, x)}});
    expect_near(out, {1, 2, 3, 4});  // Abs applied
}

// Without the extension, an unsupported op fails to convert -- confirming the op is not
// already known and that the extension in the test above is what enables it.
TEST(GGUFExtensions, UnsupportedOpWithoutExtensionThrows) {
    auto builder = SingleOpBuilder()
                       .op("GGML_OP_CUSTOM_NEGATE")
                       .input("x", ov::element::f32, {3})
                       .output("out", ov::element::f32, {3});
    EXPECT_ANY_THROW(builder.build());
}
