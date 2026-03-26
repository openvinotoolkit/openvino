// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Dynamic shape kernel selection regression test.
//
// Checks: after execute(), get_impl()->get_kernel_name() must NOT contain
//         "ocl:ref" for the deconv primitive on oneDNN-capable HW (DG2+).
//
// Why: Dynamic 5D deconv + quantize(u8, scale_shift_opt=true) causes oneDNN
//      to receive f16→u8 output, which its backward-data path cannot handle
//      with an optimized kernel. Without the fix, it falls back to ocl:ref
//      (~30x slower than jit:ir).
//
// Note: Skipped on non-immad HW where oneDNN deconv path is not used.

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/deconvolution.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "primitive_inst.h"

#include <algorithm>
#include <string>

using namespace cldnn;
using namespace ::tests;
using namespace testing;

namespace {

// Helper: check if the engine supports oneDNN (immad).
// The ocl:ref regression only occurs on oneDNN-capable HW (e.g. DG2).
bool engine_supports_immad(const engine& eng) {
    return eng.get_device_info().supports_immad;
}

TEST(dynamic_deconv_quantize_kernel, no_ocl_ref_on_onednn_5d) {
    // Skip when oneDNN is not available — the bug only affects oneDNN path.
    auto& engine = get_test_engine();
    if (!engine_supports_immad(engine)) {
        GTEST_SKIP() << "oneDNN not supported on this device; skipping ocl:ref regression test";
    }

    // 5D dynamic input: [1, 320, ?, ?, ?]
    auto input_layout_dynamic = layout{ov::PartialShape{1, 320, -1, -1, -1}, data_types::f16, format::bfzyx};

    // Weights: deconv 320->320, kernel 3x3x3
    const int ic = 320, oc = 320;
    layout weight_layout{ov::PartialShape{ic, oc, 3, 3, 3}, data_types::f16, format::bfzyx};
    auto weights_mem = engine.allocate_memory(weight_layout);
    // Zero-fill is fine — we test kernel selection, not accuracy.
    set_values(weights_mem, std::vector<ov::float16>(weight_layout.count(), ov::float16(0.01f)));

    // Quantize parameters: input_low=0, input_high=1, output_low=0, output_high=255
    // This makes scale_shift_opt=true with out_dt=u8 — the exact pattern that triggers ocl:ref.
    auto in_lo_mem  = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::f32, format::bfyx});
    auto in_hi_mem  = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::f32, format::bfyx});
    auto out_lo_mem = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::f32, format::bfyx});
    auto out_hi_mem = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::f32, format::bfyx});
    set_values<float>(in_lo_mem,  {0.0f});
    set_values<float>(in_hi_mem,  {1.0f});
    set_values<float>(out_lo_mem, {0.0f});
    set_values<float>(out_hi_mem, {255.0f});

    topology topo;
    topo.add(input_layout("input", input_layout_dynamic));
    topo.add(data("weights", weights_mem));
    topo.add(data("in_lo", in_lo_mem));
    topo.add(data("in_hi", in_hi_mem));
    topo.add(data("out_lo", out_lo_mem));
    topo.add(data("out_hi", out_hi_mem));
    topo.add(deconvolution("deconv", input_info("input"), "weights",
                           ov::Strides{2, 2, 2},
                           ov::CoordinateDiff{0, 0, 0},
                           ov::Strides{1, 1, 1}));
    topo.add(quantize("quant", input_info("deconv"),
                      input_info("in_lo"), input_info("in_hi"),
                      input_info("out_lo"), input_info("out_hi"),
                      256, data_types::u8));
    topo.add(reorder("output", input_info("quant"), format::bfzyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network net(engine, topo, config);

    // Allocate concrete input: [1, 320, 10, 10, 18]
    layout input_actual{ov::PartialShape{1, 320, 10, 10, 18}, data_types::f16, format::bfzyx};
    auto input_mem = engine.allocate_memory(input_actual);
    set_values(input_mem, std::vector<ov::float16>(input_actual.count(), ov::float16(0.5f)));

    net.set_input_data("input", input_mem);
    auto outputs = net.execute();

    // After execute(), the dynamic impl is selected and we can inspect it.
    auto deconv_inst = net.get_primitive("deconv");
    ASSERT_NE(deconv_inst, nullptr) << "deconv primitive instance not found";
    auto* impl = deconv_inst->get_impl();
    ASSERT_NE(impl, nullptr) << "deconv has no implementation after execute";

    auto kernel_name = impl->get_kernel_name();

    // The fix ensures deconv selects jit:ir (fast) instead of ocl:ref (slow, 30x+ regression).
    EXPECT_TRUE(kernel_name.find("jit:ir") != std::string::npos)
        << "Regression: deconv did not select jit:ir kernel. "
        << "kernel_name=" << kernel_name;
}

}  // namespace
