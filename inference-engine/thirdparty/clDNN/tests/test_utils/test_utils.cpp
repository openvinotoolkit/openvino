/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "api/memory.hpp"
#include <api/primitive.hpp>
#include <api/input_layout.hpp>
#include <api/data.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils.h"
#include "float16.h"
#include "instrumentation.h"
#include <iostream>

using namespace cldnn;

namespace tests
{
    const std::string graph_dump_dir = DUMP_DIRECTORY;

    generic_test::generic_test() : generic_params(std::get<0>(GetParam())), layer_params(std::get<1>(GetParam())), max_ulps_diff_allowed(4), random_values(true), dump_graphs(false), dump_memory(false)
    {
    }
    void generic_test::run_single_test()
    {
        assert((generic_params->data_type == data_types::f32) || (generic_params->data_type == data_types::f16));
        if (dump_graphs)
        {
            generic_params->network_build_options.set_option(cldnn::build_option::graph_dumps_dir(DUMP_DIRECTORY));
        }
        topology topology;
        topology.add_primitive(layer_params);

        std::vector<memory> input_mems;
        std::vector<std::string> input_layouts_names = {};

        size_t multipler = 0;
        for (size_t i = 0 ; i < generic_params->input_layouts.size() ; i++)
        {
            input_mems.push_back( memory::allocate(engine, generic_params->input_layouts[i]) );

            if (random_values)
            {
                if (generic_params->data_type == data_types::f32)
                {
                    tests::set_random_values<float>(input_mems[i], true, 7, 10);
                }
                else
                {
                    tests::set_random_values<FLOAT16>(input_mems[i], true, 5, 10);
                }
            }
            else
            {
                size_t size = generic_params->input_layouts[i].size.batch[0] * generic_params->input_layouts[i].size.feature[0];

                if (generic_params->data_type == data_types::f32)
                {
                    std::vector<float> values;
                    for (size_t j = 1; j <= size; j++)
                    {
                        values.push_back(static_cast<float>(multipler + j));
                    }
                    tests::set_values_per_batch_and_feature<float>(input_mems[i], values);
                    multipler = values.size();
                }
                else
                {
                    std::vector<FLOAT16> values;
                    for (size_t j = 1; j <= size; j++)
                    {
                        values.push_back(FLOAT16(static_cast<float>(multipler + j)));
                    }
                    tests::set_values_per_batch_and_feature<FLOAT16>(input_mems[i], values);
                    multipler = values.size();
                }
            }
            std::string input_name = "input" + std::to_string(i);
            if ( (i == 0) && generic_params->network_build_options.get<cldnn::build_option_type::optimize_data>()->enabled() )
            {
                // Add reorder after the first input in case of optimize data flag since it might change the input layout.
                input_name = "input0_init";
            }

            // First input is provided to the network as input_layout.
            // Other inputs are provided as input_layout if optimize data flag is off. Otherwise they are provided as data.
            if ( (i == 0) || !generic_params->network_build_options.get<cldnn::build_option_type::optimize_data>()->enabled())
            {
                topology.add(input_layout(input_name, input_mems[i].get_layout()));
                input_layouts_names.push_back(input_name);
            }
            else
            {
                topology.add(data(input_name, input_mems[i]));
            }

            if (!is_format_supported(generic_params->fmt))
            {
                ASSERT_THROW(network bad(engine, topology), std::exception);
                return;
            }
        }

        if (generic_params->network_build_options.get<cldnn::build_option_type::optimize_data>()->enabled())
        {
            // Add reorder after the first input in case of optimize data flag since it might change the input layout.
            topology.add(reorder("input0", "input0_init", input_mems[0].get_layout()));
        }

        if (layer_params->input[0] == "reorder0")
        {
            // Add reorder layer with output padding as input to the tested layer.
            topology.add(reorder("reorder0", "input0", input_mems[0].get_layout().with_padding(padding{ { 0, 0, 1, 3 },{ 0, 0, 5, 2 } })));
        }

        prepare_input_for_test(input_mems);

        network network(engine, topology, generic_params->network_build_options);

        for (size_t i = 0 ; i < input_layouts_names.size() ; i++)
        {
            network.set_input_data(input_layouts_names[i], input_mems[i]);
        }

        auto outputs = network.execute();
        EXPECT_EQ(outputs.size(), size_t(1));

        auto output = outputs.begin()->second.get_memory();

        auto output_ref = generate_reference(input_mems);


        if (dump_memory)
        {
            std::string prefix = test_info.name();
            for (size_t i = 0; i < generic_params->input_layouts.size(); i++)
            {
                ::instrumentation::logger::log_memory_to_file(input_mems[i], prefix + "input" + std::to_string(i));
            }
            for (size_t i = 0; i < outputs.size(); i++)
            {
                ::instrumentation::logger::log_memory_to_file(output, prefix + "output" + std::to_string(i));
            }
        }

        if (output.get_layout().data_type == data_types::f32)
        {
            compare_buffers<float>(output, output_ref);
        }
        else
        {
            compare_buffers<FLOAT16>(output, output_ref);
        }
    }

    template<typename Type>
    void generic_test::compare_buffers(const memory& out, const memory& ref)
    {
        auto out_layout = out.get_layout();
        auto ref_layout = ref.get_layout();

        EXPECT_EQ(out_layout.size, ref_layout.size);
        EXPECT_EQ(out_layout.data_type, ref_layout.data_type);
        EXPECT_EQ(get_expected_output_tensor(), out_layout.size);
        EXPECT_EQ(out_layout.get_linear_size(), ref_layout.get_linear_size());
        EXPECT_EQ(out_layout.data_padding, ref_layout.data_padding);

        auto output_size = out_layout.size;

        int batch_size = output_size.batch[0];
        int feature_size = output_size.feature[0];
        int y_size = output_size.spatial[1];
        int x_size = output_size.spatial[0];

        auto res_data = out.pointer<Type>();
        auto ref_data = ref.pointer<Type>();

        const auto out_desc = get_linear_memory_desc(out_layout);
        const auto ref_desc = get_linear_memory_desc(ref_layout);

        for (int b = 0; b < batch_size; b++)
        {
            for (int f = 0; f < feature_size; f++)
            {
                for (int y = 0; y < y_size; y++)
                {
                    for (int x = 0; x < x_size; x++)
                    {
                        size_t res_index = get_linear_index(out_layout, b, f, y, x, out_desc);
                        size_t ref_index = get_linear_index(ref_layout, b, f, y, x, ref_desc);

                        EXPECT_TRUE(floating_point_equal(res_data[res_index], ref_data[ref_index], max_ulps_diff_allowed))
                            << "Expected " << (float)res_data[res_index] << " to be almost equal (within " << max_ulps_diff_allowed << " ULP's) to " << (float)ref_data[ref_index]
                            << " (ref index = " << ref_index << ", B " << b << ", F "<< f << ", Y " << y << ", X " << x << ")!";

                        if (HasFailure())
                        {
                            return;
                        }
                    }
                }
            }
        }
    }

    static size_t calc_offfset(const layout & layout, const pitches& p)
    {
        auto lower_padding = layout.data_padding.lower_size();
        if (layout.format == format::bfzyx) {
            return
                p.b * lower_padding.batch[0] +
                p.f * lower_padding.feature[0] +
                p.z * lower_padding.spatial[2] +
                p.y * lower_padding.spatial[1] +
                p.x * lower_padding.spatial[0];
        }
        else {
            return
                p.b * lower_padding.batch[0] +
                p.f * lower_padding.feature[0] +
                p.y * lower_padding.spatial[1] +
                p.x * lower_padding.spatial[0];
        }
    }

    memory_desc generic_test::get_linear_memory_desc(const layout & layout)
    {
        pitches p;

        switch (layout.format)
        {
            case format::bfyx:
            {
                p.x = 1;
                p.y = layout.get_buffer_size().sizes(format::bfyx)[3] * p.x;
                p.f = layout.get_buffer_size().sizes(format::bfyx)[2] * p.y;
                p.b = layout.get_buffer_size().sizes(format::bfyx)[1] * p.f;
                break;
            }
            case format::yxfb:
            {
                p.b = 1;
                p.f = layout.get_buffer_size().sizes(format::yxfb)[3] * p.b;
                p.x = layout.get_buffer_size().sizes(format::yxfb)[2] * p.f;
                p.y = layout.get_buffer_size().sizes(format::yxfb)[1] * p.x;
                break;
            }
            case format::fyxb:
            {
                p.b = 1;
                p.x = layout.get_buffer_size().sizes(format::fyxb)[3] * p.b;
                p.y = layout.get_buffer_size().sizes(format::fyxb)[2] * p.x;
                p.f = layout.get_buffer_size().sizes(format::fyxb)[1] * p.y;
                break;
            }
            case format::byxf:
            {
                p.f = 1;
                p.x = layout.get_buffer_size().sizes(format::byxf)[3] * p.f;
                p.y = layout.get_buffer_size().sizes(format::byxf)[2] * p.x;
                p.b = layout.get_buffer_size().sizes(format::byxf)[1] * p.y;
                break;
            }
            case format::bfzyx:
            {
                p.x = 1;
                p.y = layout.get_buffer_size().sizes(format::bfzyx)[4] * p.x;
                p.z = layout.get_buffer_size().sizes(format::bfzyx)[3] * p.y;
                p.f = layout.get_buffer_size().sizes(format::bfzyx)[2] * p.z;
                p.b = layout.get_buffer_size().sizes(format::bfzyx)[1] * p.f;
                break;
            }
            default:
            {
                throw std::runtime_error("Format not supported yet.");
            }
        }

        return{ p, calc_offfset(layout, p) };
    }

    size_t generic_test::get_linear_index(const layout&, size_t b, size_t f, size_t y, size_t x, const memory_desc& desc)
    {
        return
            desc.offset +
            b*desc.pitch.b +
            f*desc.pitch.f +
            y*desc.pitch.y +
            x*desc.pitch.x;
    }

    size_t generic_test::get_linear_index(const layout&, size_t b, size_t f, size_t z, size_t y, size_t x, const memory_desc& desc)
    {
        return
            desc.offset +
            b*desc.pitch.b +
            f*desc.pitch.f +
            z*desc.pitch.z +
            y*desc.pitch.y +
            x*desc.pitch.x;
    }

    size_t generic_test::get_linear_index_with_broadcast(const layout& in_layout, size_t b, size_t f, size_t y, size_t x, const memory_desc& desc)
    {
        return
            desc.offset +
            (b % in_layout.size.batch[0]) * desc.pitch.b +
            (f % in_layout.size.feature[0]) * desc.pitch.f +
            (y % in_layout.size.spatial[1]) * desc.pitch.y +
            (x % in_layout.size.spatial[0]) * desc.pitch.x;
    }

    //Default implementation. Should be overridden in derived class otherwise.
    cldnn::tensor generic_test::get_expected_output_tensor()
    {
        return generic_params->input_layouts[0].size;
    }

    std::vector<std::shared_ptr<test_params>> generic_test::generate_generic_test_params(std::vector<std::shared_ptr<test_params>>& all_generic_params)
    {
        // , { format::yx,{ 531,777 } } , { format::yx,{ 4096,1980 } } ,
        //{ format::bfyx,{ 1,1,1,1 } } , { format::bfyx,{ 1,1,2,2 } } , { format::yx,{ 3,3 } } , { format::yx,{ 4,4 } } , { format::bfyx,{ 1,1,5,5 } } , { format::yx,{ 6,6 } } , { format::yx,{ 7,7 } } ,
        //{ format::yx,{ 8,8 } } , { format::yx,{ 9,9 } } , { format::yx,{ 10,10 } } , { format::yx,{ 11,11 } } , { format::yx,{ 12,12 } } , { format::yx,{ 13,13 } } ,
        //{ format::yx,{ 14,14 } } , { format::yx,{ 15,15 } } , { format::yx,{ 16,16 } } };

        auto data_types = test_data_types();

        for (cldnn::data_types data_type : data_types)
        {
            for (cldnn::format fmt : test_input_formats)
            {
                for (int batch_size : test_batch_sizes)
                {
                    for (int feature_size : test_feature_sizes)
                    {
                        for (tensor input_size : test_input_sizes)
                        {
                            all_generic_params.emplace_back(new test_params(data_type, fmt, batch_size, feature_size, input_size));
                        }
                    }
                }
            }
        }

        return all_generic_params;
    }

    const cldnn::engine & get_test_engine()
    {
        static const cldnn::engine engine;
        return engine;
    }

    const std::string test_dump::name() const
    {
        std::string temp = name_str;
        std::replace(temp.begin(), temp.end(), '/', '_');
        return temp;
    }

    const std::string test_dump::test_case_name() const
    {
        size_t pos = test_case_name_str.find("/");
        if (pos > test_case_name_str.length())
        {
            pos = 0;
        }
        std::string temp = test_case_name_str.substr(pos);
        return temp;
    }

    std::string test_params::print_tensor(cldnn::tensor t)
    {
        std::stringstream str;
        for (size_t i = 0; i < t.sizes(format::bfyx).size(); i++)
        {
            str << t.sizes(format::bfyx)[i] << " ";
        }
        str << "]";
        return str.str();
    }

    std::string test_params::print()
    {
        std::stringstream str;
        str << "Data type: " << data_type_traits::name(data_type) << std::endl;

        for (int j = 0 ; j < (int)input_layouts.size(); j++)
        {
            const cldnn::tensor& t = input_layouts[j].size;

            str << "Input " << j << ": " << print_tensor(t) << std::endl;
        }
        return str.str();
    }

    std::vector<cldnn::data_types> generic_test::test_data_types()
    {
        std::vector<cldnn::data_types> result;
        result.push_back(cldnn::data_types::f32);

        if(get_test_engine().get_info().supports_fp16)
        {
            result.push_back(cldnn::data_types::f16);
        }
        return result;
    }

    std::vector<cldnn::format> generic_test::test_input_formats = { cldnn::format::bfyx , cldnn::format::yxfb, cldnn::format::fyxb, cldnn::format::byxf };
    std::vector<int32_t> generic_test::test_batch_sizes = { 1, 2 };// 4, 8, 16};
    std::vector<int32_t> generic_test::test_feature_sizes = { 1, 2 };// , 3, 15};
    std::vector<tensor> generic_test::test_input_sizes = { { 1, 1, 100, 100 } ,{ 1, 1, 277, 277 } ,{ 1, 1, 400, 600 } };

}
