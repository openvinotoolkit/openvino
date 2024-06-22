// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/partial_shape.hpp"
#include "test_utils.h"
#include "random_generator.hpp"
#include "network_test.h"
#include <intel_gpu/runtime/utils.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include "intel_gpu/primitives/dynamic_quantize.hpp"
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "intel_gpu/runtime/compilation_context.hpp"
#include "fully_connected_inst.h"

#include <cmath>

using namespace cldnn;
using namespace ::tests;

class dynamic_quantization_gpu_tests: public ::testing::Test {
public:

    void test_dynamic_quantization(bool is_caching_test, bool is_dynamic, int batch = 1, int ifm = 1024) {
        tests::random_generator rg(GET_SUITE_NAME);
        auto& engine = get_test_engine();

        if (engine.get_device_info().dev_type == device_type::discrete_gpu)
            GTEST_SKIP();

        long int batch_num = batch;
        long int ifm_num = ifm;

        bool is_4d = true;

        auto input_ps = is_4d ?  ov::PartialShape{ batch_num, 1, 1, ifm_num } : ov::PartialShape{ batch_num, ifm_num};
        auto dyn_input_ps = is_4d ?  ov::PartialShape{ -1, 1, 1, ifm_num } : ov::PartialShape{ -1, ifm_num};
        auto input_mem = engine.allocate_memory({ input_ps, data_types::f16, format::bfyx });

        auto input_data = rg.generate_random_1d<ov::float16>(batch_num * ifm_num, -16.0f, 16.0f);
        set_values(input_mem, input_data);

        auto in_layout = is_dynamic ? layout{ dyn_input_ps, data_types::f16, format::bfyx }
                                    : layout{ input_ps, data_types::f16, format::bfyx };

        auto dyn_quan_prim = dynamic_quantize("dyn_quan_prim", input_info("input"), 32, data_types::i8);

        // Implemented dynamic quantize kernel
        auto get_ref_results = [&]() {
            topology topology(
                input_layout("input", in_layout),
                dyn_quan_prim
            );

            auto config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            config.set_property(ov::intel_gpu::optimize_data(true));

            ov::intel_gpu::ImplementationDesc dyn_quan_impl_desc = { format::bfyx, "dynamic_quantize_gpu_ref", impl_types::ocl };
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"dyn_quan_prim", dyn_quan_impl_desc} }));

            network network(engine, topology, config);
            network.set_input_data("input", input_mem);

            auto outputs = network.execute();
            OPENVINO_ASSERT(outputs.begin()->first == "dyn_quan_prim");

            auto output_layout = outputs.begin()->second.get_layout();
            auto output_mem = outputs.begin()->second.get_memory();

            return engine.reinterpret_buffer(*output_mem, output_layout);
        };

        topology topology(
            input_layout("input", in_layout),
            dyn_quan_prim
        );

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        config.set_property(ov::intel_gpu::optimize_data(true));

        network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        // if (is_dynamic && !engine.get_device_info().supports_immad) {
        //     auto inst = network->get_primitive("dyn_quan_prim");
        //     auto impl = inst->get_impl();
        //     ASSERT_TRUE(impl != NULL);
        //     ASSERT_EQ(impl->get_kernels().size(), size_t((is_dynamic ? 3 : 2))); // shape-agnostic kernels
        // }

        network->set_input_data("input", input_mem);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.begin()->first, "dyn_quan_prim");

        auto output_mem = outputs.begin()->second.get_memory();
        cldnn::mem_lock<ov::float16> output_ptr (output_mem, get_test_stream());

        auto ref_output_mem = get_ref_results();
        cldnn::mem_lock<ov::float16> output_ptr_ref (ref_output_mem, get_test_stream());

        size_t count = 0;
        float max_diff = 0.f;
        float avg = 0.f;
        for (size_t i = 0; i < output_ptr_ref.size(); ++i) {
            auto abs_diff = std::abs(output_ptr_ref[i] - output_ptr[i]);
            if (max_diff < abs_diff)
                max_diff = abs_diff;
            avg = abs_diff;
            count++;
            // OPENVINO_ASSERT(abs_diff < 256);
        }
        /*GPU_DEBUG_LOG*/std::cout << "---> count: " << count << ", max_diff:" << max_diff << ", avg_diff: " << (avg/count) << std::endl;
    }
};

TEST_F(dynamic_quantization_gpu_tests, simple_quantizing_large_size) {
    this->test_dynamic_quantization(false, false, 2048, 4096);
}