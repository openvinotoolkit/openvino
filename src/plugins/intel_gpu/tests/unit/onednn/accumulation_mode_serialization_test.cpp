// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "convolution_inst.h"
#include "graph/impls/onednn/primitive_onednn_base.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

TEST(onednn_convolution_serialization, preserves_accumulation_mode_in_cache) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    tests::random_generator rg(GET_SUITE_NAME);

    const int batch = 1, ifm = 64, ofm = 64, spatial = 16, ksize = 3, pad = 1;

    auto input_size = tensor(batch, ifm, spatial, spatial);
    auto weights_size = tensor(ofm, ifm, ksize, ksize);

    auto input_data = rg.generate_random_4d<ov::float16>(batch, ifm, spatial, spatial, -1, 1);
    auto weights_data = rg.generate_random_4d<ov::float16>(ofm, ifm, ksize, ksize, -1, 1);

    auto input_mem = engine.allocate_memory({data_types::f16, format::bfyx, input_size});
    auto weights_mem = engine.allocate_memory({data_types::f16, format::bfyx, weights_size});
    set_values(input_mem, flatten_4d(format::bfyx, input_data));
    set_values(weights_mem, flatten_4d(format::bfyx, weights_data));

    topology topology(input_layout("input", input_mem->get_layout()),
                      data("weights", weights_mem),
                      reorder("input_fsv", input_info("input"), format::b_fs_yx_fsv16, data_types::f16),
                      convolution("conv", input_info("input_fsv"), "weights", "", 1, {1, 1}, {1, 1}, {pad, pad}, {pad, pad}, false),
                      reorder("output", input_info("conv"), format::bfyx, data_types::f32));

    // Default execution_mode is PERFORMANCE, so accumulation_mode::any is set on the
    // f16 conv and therefore must round-trip through serialization.
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    ov::intel_gpu::ImplementationDesc conv_impl = {format::b_fs_yx_fsv16, std::string(""), impl_types::onednn};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv", conv_impl}}));

    auto stream = get_test_stream_ptr();

    using return_type = std::pair<dnnl::accumulation_mode, std::vector<float>>;
    auto run = [&](bool is_caching_test) -> return_type {
        auto net = get_network(engine, topology, config, stream, is_caching_test);

        auto conv_inst = net->get_primitive("conv");
        auto impl = conv_inst->get_impl();
        auto onednn_impl = dynamic_cast<cldnn::onednn::typed_primitive_onednn_impl<cldnn::convolution>*>(impl);
        if (!onednn_impl)
            throw std::runtime_error("conv is not a oneDNN primitive implementation");

        auto acc_mode = (onednn_impl->_attrs && onednn_impl->_attrs->get()) ? onednn_impl->_attrs->get_accumulation_mode() : dnnl::accumulation_mode::strict;

        net->set_input_data("input", input_mem);
        auto outputs = net->execute();
        auto out_mem = outputs.at("output").get_memory();
        mem_lock<float, mem_lock_type::read> ptr(out_mem, get_test_stream());
        std::vector<float> vals(ptr.size());
        for (size_t i = 0; i < ptr.size(); ++i)
            vals[i] = ptr[i];
        return std::make_pair(acc_mode, vals);
    };

    auto ref = run(false);
    auto cached = run(true);

    ASSERT_EQ(ref.first, dnnl::accumulation_mode::any);
    ASSERT_EQ(cached.first, dnnl::accumulation_mode::any);

    ASSERT_EQ(ref.second.size(), cached.second.size());
    for (size_t i = 0; i < ref.second.size(); ++i) {
        ASSERT_EQ(cached.second[i], ref.second[i]);
    }
}
