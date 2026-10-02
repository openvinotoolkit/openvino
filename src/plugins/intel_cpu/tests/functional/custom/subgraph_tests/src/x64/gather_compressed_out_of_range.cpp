// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace test {
namespace {

// Gather of u2 rows along axis 0, i.e. afterAxisSize == kHidden > 1. That is what makes the
// out-of-range zero fill observable: the zeroing loop is per row, and a loop that does not walk
// the row leaves kHidden - 1 elements holding whatever the previous inference wrote there.
constexpr size_t kVocab = 8;
constexpr size_t kHidden = 6;
constexpr size_t kIndices = 3;

// The four combinations below select all four dequantization branches of the u2 kernel:
//   scalar scale + scalar/absent zp -> the scalar fast path
//   per-row scale + per-row/absent zp -> the grouped path
//   scalar scale + per-row zp -> neither, i.e. the generic fallback loop
struct GatherU2Form {
    bool scalar_scale;
    bool use_zp;
    bool scalar_zp;
    const char* name;
};

// Weights live in {2, 3} and zero points in {0, 1}, so every dequantized element is at least one
// quantization step away from zero. A stale value can therefore never be mistaken for a
// legitimately gathered 0.
int32_t weight_at(size_t v, size_t h) {
    return static_cast<int32_t>(((v + h) % 2) + 2);
}

float zp_at(const GatherU2Form& form, size_t v) {
    if (!form.use_zp) {
        return 0.0F;
    }
    return form.scalar_zp ? 1.0F : static_cast<float>(v % 2);
}

float scale_at(const GatherU2Form& form, size_t v) {
    return form.scalar_scale ? 0.5F : 0.25F * static_cast<float>(v + 1);
}

float expected_at(const GatherU2Form& form, size_t v, size_t h) {
    return (static_cast<float>(weight_at(v, h)) - zp_at(form, v)) * scale_at(form, v);
}

std::shared_ptr<ov::Model> make_model(const GatherU2Form& form) {
    std::vector<int32_t> weight_values(kVocab * kHidden);
    for (size_t v = 0; v < kVocab; ++v) {
        for (size_t h = 0; h < kHidden; ++h) {
            weight_values[v * kHidden + h] = weight_at(v, h);
        }
    }
    auto weights = ov::op::v0::Constant::create(ov::element::u2, ov::Shape{kVocab, kHidden}, weight_values);
    weights->set_friendly_name("Compressed_weights");

    std::shared_ptr<ov::Node> dict = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);

    if (form.use_zp) {
        const ov::Shape zp_shape = form.scalar_zp ? ov::Shape{1} : ov::Shape{kVocab, 1};
        std::vector<int32_t> zp_values(ov::shape_size(zp_shape));
        for (size_t i = 0; i < zp_values.size(); ++i) {
            zp_values[i] = static_cast<int32_t>(zp_at(form, i));
        }
        auto zp = ov::op::v0::Constant::create(ov::element::u2, zp_shape, zp_values);
        dict = std::make_shared<ov::op::v1::Subtract>(dict,
                                                     std::make_shared<ov::op::v0::Convert>(zp, ov::element::f32));
    }

    const ov::Shape scale_shape = form.scalar_scale ? ov::Shape{1} : ov::Shape{kVocab, 1};
    std::vector<float> scale_values(ov::shape_size(scale_shape));
    for (size_t i = 0; i < scale_values.size(); ++i) {
        scale_values[i] = scale_at(form, i);
    }
    dict = std::make_shared<ov::op::v1::Multiply>(
        dict,
        ov::op::v0::Constant::create(ov::element::f32, scale_shape, scale_values));

    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{kIndices});
    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    auto gather = std::make_shared<ov::op::v8::Gather>(dict, indices, axis);
    gather->set_friendly_name("gather_node");

    return std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{indices}, "GatherCompressedU2");
}

void expect_compressed_u2_gather(const ov::CompiledModel& compiled) {
    bool found = false;
    for (const auto& node : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type = rt_info.find(ov::exec_model_info::LAYER_TYPE);
        if (layer_type == rt_info.end() || layer_type->second.as<std::string>() != "Gather") {
            continue;
        }
        // 4 inputs (data, indices, axis, scale) is how a GatherCompressed shows up in the runtime model.
        if (node->get_input_size() < 4) {
            continue;
        }
        found = true;
        EXPECT_EQ(node->get_input_element_type(0), ov::element::u2);
    }
    ASSERT_TRUE(found) << "no compressed Gather in the runtime model: the u2 dictionary was decompressed "
                          "away, so this test would not reach the u2 gather kernel at all";
}

class GatherCompressedU2OutOfRange : public testing::WithParamInterface<GatherU2Form>,
                                     public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherU2Form>& obj) {
        return obj.param.name;
    }
};

TEST_P(GatherCompressedU2OutOfRange, out_of_range_index_zeroes_the_whole_row) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    const auto& form = GetParam();

    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled = core->compile_model(make_model(form),
                                        ov::test::utils::DEVICE_CPU,
                                        {ov::hint::inference_precision(ov::element::f32)});
    expect_compressed_u2_gather(compiled);

    auto request = compiled.create_infer_request();
    // The indices are written in place instead of handing over a fresh tensor per inference: the
    // point of this test is that consecutive inferences share one output buffer.
    auto indices = request.get_input_tensor(0);

    auto set_indices = [&](int32_t i0, int32_t i1, int32_t i2) {
        auto* data = indices.data<int32_t>();
        data[0] = i0;
        data[1] = i1;
        data[2] = i2;
    };
    auto check_gathered = [&](size_t row, size_t source_row) {
        const auto* out = request.get_output_tensor(0).data<float>();
        for (size_t h = 0; h < kHidden; ++h) {
            EXPECT_NEAR(out[row * kHidden + h], expected_at(form, source_row, h), 1e-6F)
                << "row " << row << " (index " << source_row << ") element " << h;
        }
    };
    auto check_zeroed = [&](size_t row) {
        const auto* out = request.get_output_tensor(0).data<float>();
        for (size_t h = 0; h < kHidden; ++h) {
            EXPECT_EQ(out[row * kHidden + h], 0.0F) << "row " << row << " element " << h << " was not zeroed";
        }
    };
    // Fill the output with non-zero values first. On a freshly allocated (zeroed) buffer an
    // incomplete zero fill is invisible, so each out-of-range check below is preceded by this.
    auto fill_output_with_non_zeros = [&]() {
        set_indices(1, 2, 3);
        request.infer();
        check_gathered(0, 1);
        check_gathered(1, 2);
        check_gathered(2, 3);
    };

    fill_output_with_non_zeros();

    // kVocab is one past the last valid row; Gather-8 defines such a row as all zeros.
    set_indices(1, static_cast<int32_t>(kVocab), 3);
    request.infer();
    check_gathered(0, 1);
    check_zeroed(1);
    check_gathered(2, 3);

    fill_output_with_non_zeros();

    // Same row, reached the other way: reverse indexing adds axisDim to a negative index, so
    // -kVocab - 1 stays negative, wraps around when cast to an unsigned offset, and lands in the
    // same out-of-range branch.
    set_indices(1, -static_cast<int32_t>(kVocab) - 1, 3);
    request.infer();
    check_gathered(0, 1);
    check_zeroed(1);
    check_gathered(2, 3);
}

const std::vector<GatherU2Form> forms = {
    {false, false, false, "per_row_scale_no_zp"},
    {false, true, false, "per_row_scale_per_row_zp"},
    {true, true, true, "scalar_scale_scalar_zp"},
    {true, false, false, "scalar_scale_no_zp"},
    {true, true, false, "scalar_scale_per_row_zp"},
};

INSTANTIATE_TEST_SUITE_P(smoke_GatherCompressedU2OutOfRange,
                         GatherCompressedU2OutOfRange,
                         ::testing::ValuesIn(forms),
                         GatherCompressedU2OutOfRange::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
