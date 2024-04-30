// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/layout.hpp"
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "eltwise_inst.h"

using namespace cldnn;
using namespace ::tests;

struct eltwise_perf_test_params {
    data_types  input_type;
    tensor      first_input_size;
    tensor      second_input_size;

    format::type in_format;
    format::type in_format_second;  // For testing 1x1x1x1 bfyx
    format::type out_format;
    eltwise_mode mode;
    impl_types   impl_type;
    bool         is_caching_test;
};

struct eltwise_perf_test : testing::TestWithParam<eltwise_perf_test_params>
{
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    static std::string PrintToString(const eltwise_perf_test_params& params) {
        std::string res = " data (" + ov::element::Type(params.input_type).get_type_name() + "), ";
        res += " format (" + format::traits(params.in_format).str + ") input1 : ";
        res += params.first_input_size.to_string() + " / input2 : ";
        res += params.second_input_size.to_string() + "\n";

        return res;
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);

        auto data = rg.generate_random_4d<T>(b, f, y, x, min, max, k);
        mem_lock<T> ptr{mem, get_test_stream()};
        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t fi = 0; fi < f; ++fi) {
                for (size_t yi = 0; yi < y; ++yi) {
                    for (size_t xi = 0; xi < x; ++xi) {
                        auto coords = tensor(batch(bi), feature(fi), spatial(xi, yi, 0, 0));
                        auto offset = mem->get_layout().get_linear_offset(coords);
                        ptr[offset] = data[bi][fi][yi][xi];
                    }
                }
            }
        }
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<ov::float16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    void execute_perf(const eltwise_perf_test_params& params, bool check_result) {
        auto& engine = get_test_engine();

        auto in_layout1 = layout(params.input_type, params.in_format, params.first_input_size);
        auto in_layout2 = layout(params.input_type, params.in_format_second, params.second_input_size);
        // bool is_caching_test = params.is_caching_test;
        auto input1 = engine.allocate_memory(in_layout1);
        auto input2 = engine.allocate_memory(in_layout2);
        fill_random(input1);
        fill_random(input2);

        cldnn::topology topo;
        topo.add(input_layout("input1", input1->get_layout()));
        topo.add(input_layout("input2", input2->get_layout()));
        auto prim = eltwise("eltwise", { input_info("input1"), input_info("input2") }, params.mode);
        topo.add(prim);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::enable_profiling(true));

        cldnn::network net(engine, topo, config);
        net.set_input_data("input1", input1);
        net.set_input_data("input2", input2);

        std::map<primitive_id, network_output> result;
        auto r = 10000;
        double exectime = 0.f;
        for (int i = 0; i < r; ++i) {
            result = net.execute();
            // print_profiling_all_exectimes(result);
            exectime += get_profiling_exectime(result, "eltwise");
        }
        exectime /= r;
        std::string input_type = ov::element::Type(params.input_type).get_type_name();

        std::cout << "Exectued time" << " input1(" << params.first_input_size.to_string() << ") input2(" << params.second_input_size.to_string() << ")"
                  << ") " << params.in_format << " " << input_type << " " << exectime << std::endl;
    }
};

TEST_P(eltwise_perf_test, perf) {
    auto param = GetParam();
    execute_perf(param, true);
}

INSTANTIATE_TEST_SUITE_P(eltwise_perf,
                         eltwise_perf_test,
                         testing::ValuesIn(
                            std::vector<eltwise_perf_test_params>{
                                { data_types::f16, {1, 1, 1, 11008}, {1, 1, 1, 1}, cldnn::format::bfyx, cldnn::format::bfyx, cldnn::format::bfyx, eltwise_mode::sum, impl_types::ocl, false },
                                { data_types::f16, {1, 1, 1, 35}, {1, 1, 1, 1}, cldnn::format::bfyx, cldnn::format::bfyx, cldnn::format::bfyx, eltwise_mode::sum, impl_types::ocl, false },
                                { data_types::f16, {1, 1, 32, 128}, {1, 1, 1, 1}, cldnn::format::bfyx, cldnn::format::bfyx, cldnn::format::bfyx, eltwise_mode::sum, impl_types::ocl, false },
                                { data_types::f16, {1, 1, 32, 64}, {1, 1, 1, 1}, cldnn::format::bfyx, cldnn::format::bfyx, cldnn::format::bfyx, eltwise_mode::sum, impl_types::ocl, false },
                                { data_types::f16, {1, 1, 1, 4096}, {1, 1, 1, 1}, cldnn::format::bfyx, cldnn::format::bfyx, cldnn::format::bfyx, eltwise_mode::sum, impl_types::ocl, false },
                                { data_types::f16, {1, 32, 1, 128}, {1, 1, 1, 1}, cldnn::format::bfyx, cldnn::format::bfyx, cldnn::format::bfyx, eltwise_mode::sum, impl_types::ocl, false },
                            }
                        ));