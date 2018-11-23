/*
// Copyright (c) 2016 Intel Corporation
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

#include <gtest/gtest.h>

#include <api/CPP/engine.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/index_select.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>

#include "test_utils/test_utils.h"

#include <vector>
#include <algorithm>

using namespace cldnn;
using namespace tests;

std::vector<float> generate_reference_bfyx(const std::vector<float>& input, const std::vector<int32_t>& indices, index_select_axis_name axis, const size_t b_size, const size_t f_size,
    const size_t y_size, const size_t x_size)
{

    auto unique_indices = std::set<int32_t>(indices.begin(), indices.end());
    std::map<uint32_t, std::vector<float>> mapped_axises;

    auto append_to_vector = [&](std::vector<float>& vec)
    {
        for (auto const& id : indices)
        {
            vec.insert(vec.end(), mapped_axises.at(id).begin(), mapped_axises.at(id).end());
        }
    };

    std::vector<float> ret;
    size_t offset = 0;    
    switch (axis)
    {
    case index_select_axis_name::along_b:
        offset = f_size * y_size * x_size; 
        for (auto const& ui : unique_indices)
        {
            mapped_axises[ui] = std::vector<float>(input.begin() + ui * offset, input.begin() + ui * offset + offset);
        }
        append_to_vector(ret);
        return ret;
    case index_select_axis_name::along_f:
        offset = y_size * x_size;   
        for (size_t i = 0; i < b_size; i++)
        {
            size_t batch_index = i * f_size * y_size * x_size;
            mapped_axises.clear();
            for (auto const& ui : unique_indices)
            {
                mapped_axises[ui] = std::vector<float>(input.begin() + batch_index + ui * offset, input.begin() + batch_index + ui * offset + offset);
            }
            append_to_vector(ret);
        }
        return ret;
    case index_select_axis_name::along_x:
        offset = y_size;
        ret.resize(b_size * f_size * y_size * indices.size());
        for (size_t i = 0; i < b_size; i++)
        {
            size_t batch_index = i * f_size * y_size * x_size;
            for (size_t j = 0; j < f_size; j++)
            {
                size_t feature_index = j * y_size * x_size;
                size_t start_idx = batch_index + feature_index;
                mapped_axises.clear();
                for (auto const& ui : unique_indices)
                {
                    std::vector<float> values = {};
                    for (size_t k = 0; k < offset; k++)
                    {
                        values.push_back(input.at(start_idx + k * x_size +  ui));
                    }
                    mapped_axises[ui] = values;
                }

                for (size_t idx = 0; idx < indices.size(); idx++)
                {
                    auto const id = indices.at(idx);
                    //ret.insert(ret.end(), mapped_axises.at(id).begin(), mapped_axises.at(id).end());
                    auto out_idx = i * f_size * y_size * indices.size() + j * y_size * indices.size() + idx;
                    for (size_t y = 0; y < y_size; y++)
                    {
                        ret.at(out_idx + y * indices.size()) = mapped_axises.at(id).at(y);
                    }
                }
            }
        }
        return ret;
        break;
    case index_select_axis_name::along_y:
        offset = x_size;
        for (size_t i = 0; i < b_size; i++)
        {
            size_t batch_index = i * f_size * y_size * x_size;
            for (size_t j = 0; j < f_size; j++)
            {
                size_t feature_index = j * y_size * x_size;
                size_t start_idx = batch_index + feature_index;
                mapped_axises.clear();
                for (auto const& ui : unique_indices)
                {
                    mapped_axises[ui] = std::vector<float>(input.begin() + start_idx + ui * offset, input.begin() + start_idx + ui * offset + offset);
                }
                append_to_vector(ret);
            }
        }
        return ret;
        break;
    default:
        throw std::runtime_error("Unknown index_select axis!");
        break;
    }
}


std::vector<float> generate_reference_yxfb(const std::vector<float>& input, const std::vector<int32_t>& indices, index_select_axis_name axis, const cldnn::layout& input_lay)
{
    auto memory_desc_inp = generic_test::get_linear_memory_desc(input_lay);

    std::vector<float> ret;
    switch (axis)
    {
    case index_select_axis_name::along_b:
 
        for (auto y = 0; y < input_lay.size.spatial[1]; y++)
        {
            for (auto x = 0; x < input_lay.size.spatial[0]; x++)
            {
                for (auto f = 0; f < input_lay.size.feature[0]; f++)
                {
                    for (auto const& ind : indices)
                    {

                        size_t index = generic_test::get_linear_index(input_lay, ind, f, y, x, memory_desc_inp);
                        ret.push_back(input.at(index));
                    }

                }
            }
        }
        return ret;
    case index_select_axis_name::along_f:
        for (auto y = 0; y < input_lay.size.spatial[1]; y++)
        {
            for (auto x = 0; x < input_lay.size.spatial[0]; x++)
            {
                for (auto const& ind : indices)
                {
                    for (auto b = 0; b < input_lay.size.batch[0]; b++)
                    {
                    size_t index = generic_test::get_linear_index(input_lay, b, ind, y, x, memory_desc_inp);
                    ret.push_back(input.at(index));
                    }
                }
            }
        }
        return ret;
    case index_select_axis_name::along_x:
        for (auto y = 0; y < input_lay.size.spatial[1]; y++)
        {
            for (auto const& ind : indices)
            {
                for (auto f = 0; f < input_lay.size.feature[0]; f++)
                {
                    for (auto b = 0; b < input_lay.size.batch[0]; b++)
                    {
                        size_t index = generic_test::get_linear_index(input_lay, b, f, y, ind, memory_desc_inp);
                        ret.push_back(input.at(index));
                    }
                }
            }
        }
        return ret;
    case index_select_axis_name::along_y:

        for (auto const& ind : indices)
        {
            for (auto x = 0; x < input_lay.size.spatial[0]; x++)
            {
                for (auto f = 0; f < input_lay.size.feature[0]; f++)
                {
                    for (auto b = 0; b < input_lay.size.batch[0]; b++)
                    {
                        size_t index = generic_test::get_linear_index(input_lay, b, f, ind, x, memory_desc_inp);
                        ret.push_back(input.at(index));
                    }
                }
            }
        }
        return ret;
    default:
        throw std::runtime_error("Unknown index_select axis!");
        break;
    }
}


TEST(index_select_gpu, basic_along_b_3_executes_bfyx)
{
    /*
    input: {5, 2, 3, 4}
    indices: {1, 1, 4, 1}
    output: {4, 2, 3, 4}
    */
    engine engine;
    constexpr auto in_size_b = 5;
    constexpr auto in_size_f = 2;
    constexpr auto in_size_x = 3;
    constexpr auto in_size_y = 4;
    constexpr auto count = in_size_b * in_size_f * in_size_x * in_size_y;
    constexpr auto new_indicies_size = 4;
    constexpr auto axis = index_select_axis_name::along_b;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { in_size_b, in_size_f, in_size_x, in_size_y } });
    auto indices = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 1, new_indicies_size, 1 } });
    
    auto input_data = generate_random_1d<float>(count, 0, 10);
    set_values(input, input_data);

    /*
    Network will be executed 3 times (for 3 different indicies_data).
    */
    std::vector<std::vector<int32_t>> indices_data =
    {
        {0, 1, 1, 1}, //for run: 0
        {0, 1, 2, 3}, //for run: 1
        {4, 3, 2, 1} // for run: 2
    };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        input_layout("indices", indices.get_layout())
    );
    topo.add(
        index_select("index_select", "input", "indices", axis)
    );

    network net(engine, topo);
    net.set_input_data("input", input);
    for (auto const& id : indices_data)
    {
        set_values(indices, id);
        net.set_input_data("indices", indices);
        auto outputs = net.execute();

		ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "index_select");

        auto output_mem = outputs.at("index_select").get_memory();
        auto output_layout = output_mem.get_layout();

        int b_size = output_layout.size.batch[0];
        int f_size = output_layout.size.feature[0];
        int x_size = output_layout.size.spatial[0];
        int y_size = output_layout.size.spatial[1];
        EXPECT_EQ(output_layout.format, format::bfyx);
        EXPECT_EQ(b_size, new_indicies_size);
        EXPECT_EQ(f_size, in_size_f);
        EXPECT_EQ(x_size, in_size_x);
        EXPECT_EQ(y_size, in_size_y);
        
        auto ref = generate_reference_bfyx(input_data, id, axis, in_size_b, in_size_f, in_size_y, in_size_x);

        auto output_ptr = output_mem.pointer<float>();
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            EXPECT_EQ(output_ptr[i], ref[i]);
        }
    }
}

TEST(index_select_gpu, basic_along_f_3_executes_bfyx)
{
    /*
    input: {2, 5, 3, 3}
    indices: {1, 1, 10, 1}
    output: {2, 10, 3, 3}
    */
    engine engine;
    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 5;
    constexpr auto in_size_x = 3;
    constexpr auto in_size_y = 3;
    constexpr auto count = in_size_b * in_size_f * in_size_x * in_size_y;
    constexpr auto new_indicies_size = 10;
    constexpr auto axis = index_select_axis_name::along_f;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { in_size_b, in_size_f, in_size_x, in_size_y } });
    auto indices = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 1, new_indicies_size, 1 } });

    auto input_data = generate_random_1d<float>(count, 0, 10);
    set_values(input, input_data);

    /*
    Network will be executed 3 times (for 3 different indicies_data).
    */
    std::vector<std::vector<int32_t>> indices_data =
    {
        { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 }, //for run: 0
        { 1, 1, 3, 3, 2, 2, 4, 4, 0, 0 }, //for run: 1
        { 0, 0, 0, 0, 0, 4, 3, 2, 1, 0 } // for run: 2
    };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        input_layout("indices", indices.get_layout())
    );
    topo.add(
        index_select("index_select", "input", "indices", axis)
    );

    network net(engine, topo);
    net.set_input_data("input", input);
    for (auto const& id : indices_data)
    {
        set_values(indices, id);
        net.set_input_data("indices", indices);
        auto outputs = net.execute();

		ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "index_select");

        auto output_mem = outputs.at("index_select").get_memory();
        auto output_layout = output_mem.get_layout();

        int b_size = output_layout.size.batch[0];
        int f_size = output_layout.size.feature[0];
        int x_size = output_layout.size.spatial[0];
        int y_size = output_layout.size.spatial[1];
        EXPECT_EQ(output_layout.format, format::bfyx);
        EXPECT_EQ(b_size, in_size_b);
        EXPECT_EQ(f_size, new_indicies_size);
        EXPECT_EQ(x_size, in_size_x);
        EXPECT_EQ(y_size, in_size_y);

        auto ref = generate_reference_bfyx(input_data, id, axis, in_size_b, in_size_f, in_size_y, in_size_x);

        auto output_ptr = output_mem.pointer<float>();
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            EXPECT_EQ(output_ptr[i], ref[i]);
        }
    }
}

TEST(index_select_gpu, basic_along_x_3_executes_bfyx)
{
    /*
    input: {3, 4, 6, 5}
    indices: {1, 1, 3, 1}
    output: {3, 4, 3, 5}
    */
    engine engine;
    constexpr auto in_size_b = 3;
    constexpr auto in_size_f = 4;
    constexpr auto in_size_x = 6;
    constexpr auto in_size_y = 5;
    constexpr auto count = in_size_b * in_size_f * in_size_x * in_size_y;
    constexpr auto new_indicies_size = 3;
    constexpr auto axis = index_select_axis_name::along_x;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx, { in_size_b, in_size_f, in_size_x, in_size_y } });
    auto indices = memory::allocate(engine, { data_types::i32, format::bfyx, { 1, 1, new_indicies_size, 1 } });

    auto input_data = generate_random_1d<float>(count, 0, 10);
    set_values(input, input_data);

    /*
    Network will be executed 3 times (for 3 different indicies_data).
    */
    std::vector<std::vector<int32_t>> indices_data =
    {
        { 2, 1, 0 }, //for run: 0
        { 0, 0, 0 }, //for run: 1
        { 1, 1, 0 } // for run: 2
    };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        input_layout("indices", indices.get_layout())
    );
    topo.add(
        index_select("index_select", "input", "indices", axis)
    );

    network net(engine, topo);
    net.set_input_data("input", input);
    for (auto const& id : indices_data)
    {
        set_values(indices, id);
        net.set_input_data("indices", indices);
        auto outputs = net.execute();

		ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "index_select");

        auto output_mem = outputs.at("index_select").get_memory();
        auto output_layout = output_mem.get_layout();

        int b_size = output_layout.size.batch[0];
        int f_size = output_layout.size.feature[0];
        int x_size = output_layout.size.spatial[0];
        int y_size = output_layout.size.spatial[1];
        EXPECT_EQ(output_layout.format, format::bfyx);
        EXPECT_EQ(b_size, in_size_b);
        EXPECT_EQ(f_size, in_size_f);
        EXPECT_EQ(x_size, new_indicies_size);
        EXPECT_EQ(y_size, in_size_y);

        auto ref = generate_reference_bfyx(input_data, id, axis, in_size_b, in_size_f, in_size_y, in_size_x);

        auto output_ptr = output_mem.pointer<float>();
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            EXPECT_EQ(output_ptr[i], ref[i]);
        }
    }
}

TEST(index_select_gpu, basic_along_y_3_executes_bfyx)
{
    /*
    input: {2, 4, 4, 3}
    indices: {1, 1, 5, 1}
    output: {2, 4, 4, 5}
    */
    engine engine;
    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 4;
    constexpr auto in_size_x = 4;
    constexpr auto in_size_y = 3;
    constexpr auto count = in_size_b * in_size_f * in_size_x * in_size_y;
    constexpr auto new_indicies_size = 5;
    constexpr auto axis = index_select_axis_name::along_y;
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ in_size_b, in_size_f, in_size_x, in_size_y } });
    auto indices = memory::allocate(engine, { data_types::i32, format::bfyx,{ 1, 1, new_indicies_size, 1 } });

    auto input_data = generate_random_1d<float>(count, 0, 10);
    set_values(input, input_data);

    /*
    Network will be executed 3 times (for 3 different indicies_data).
    */
    std::vector<std::vector<int32_t>> indices_data =
    {
        { 0, 1, 2, 2, 1 }, //for run: 0
        { 2, 2, 1, 0, 1 }, //for run: 1
        { 1, 1, 2, 1, 0 } // for run: 2
    };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        input_layout("indices", indices.get_layout())
    );
    topo.add(
        index_select("index_select", "input", "indices", axis)
    );

    network net(engine, topo);
    net.set_input_data("input", input);
    for (auto const& id : indices_data)
    {
        set_values(indices, id);
        net.set_input_data("indices", indices);
        auto outputs = net.execute();

		ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "index_select");

        auto output_mem = outputs.at("index_select").get_memory();
        auto output_layout = output_mem.get_layout();

        int b_size = output_layout.size.batch[0];
        int f_size = output_layout.size.feature[0];
        int x_size = output_layout.size.spatial[0];
        int y_size = output_layout.size.spatial[1];
        EXPECT_EQ(output_layout.format, format::bfyx);
        EXPECT_EQ(b_size, in_size_b);
        EXPECT_EQ(f_size, in_size_f);
        EXPECT_EQ(x_size, x_size);
        EXPECT_EQ(y_size, new_indicies_size);

        auto ref = generate_reference_bfyx(input_data, id, axis, in_size_b, in_size_f, in_size_y, in_size_x);

        auto output_ptr = output_mem.pointer<float>();
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            EXPECT_EQ(output_ptr[i], ref[i]);
        }
    }
}

TEST(index_select_gpu, basic_along_b_3_executes_yxfb)
{
    /*
    input: {5, 2, 3, 4}
    indices: {1, 1, 4, 1}
    output: {4, 2, 3, 4}
    */
    engine engine;
    constexpr auto in_size_b = 5;
    constexpr auto in_size_f = 2;
    constexpr auto in_size_x = 3;
    constexpr auto in_size_y = 4;
    constexpr auto count = in_size_b * in_size_f * in_size_x * in_size_y;
    constexpr auto new_indicies_size = 4;
    constexpr auto axis = index_select_axis_name::along_b;
    auto input_lay= cldnn::layout(data_types::f32, format::yxfb,{ in_size_b, in_size_f, in_size_x, in_size_y });
    auto input = memory::allocate(engine, input_lay);
    auto indices = memory::allocate(engine, { data_types::i32, format::yxfb, { 1, 1, new_indicies_size, 1 } });

    auto input_data = generate_random_1d<float>(count, 0, 10);
    set_values(input, input_data);

    /*
    Network will be executed 3 times (for 3 different indicies_data).
    */
    std::vector<std::vector<int32_t>> indices_data =
    {
        { 0, 1, 1, 1 }, //for run: 0
        { 0, 1, 2, 3 }, //for run: 1
        { 4, 3, 2, 1 } // for run: 2
    };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        input_layout("indices", indices.get_layout())
    );
    topo.add(
        index_select("index_select", "input", "indices", axis)
    );

    network net(engine, topo);
    net.set_input_data("input", input);
    for (auto const& id : indices_data)
    {
        set_values(indices, id);
        net.set_input_data("indices", indices);
        auto outputs = net.execute();

		ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "index_select");

        auto output_mem = outputs.at("index_select").get_memory();
        auto output_layout = output_mem.get_layout();

        int b_size = output_layout.size.batch[0];
        int f_size = output_layout.size.feature[0];
        int x_size = output_layout.size.spatial[0];
        int y_size = output_layout.size.spatial[1];
        EXPECT_EQ(output_layout.format, format::yxfb);
        EXPECT_EQ(b_size, new_indicies_size);
        EXPECT_EQ(f_size, in_size_f);
        EXPECT_EQ(x_size, in_size_x);
        EXPECT_EQ(y_size, in_size_y);

        auto ref = generate_reference_yxfb(input_data, id, axis, input_lay);

        auto output_ptr = output_mem.pointer<float>();
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            EXPECT_EQ(output_ptr[i], ref[i]);
        }
    }
}

TEST(index_select_gpu, basic_along_f_3_executes_yxfb)
{
    /*
    input: {2, 5, 3, 3}
    indices: {1, 1, 10, 1}
    output: {2, 10, 3, 3}
    */
    engine engine;
    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 5;
    constexpr auto in_size_x = 3;
    constexpr auto in_size_y = 3;
    constexpr auto count = in_size_b * in_size_f * in_size_x * in_size_y;
    constexpr auto new_indicies_size = 10;
    constexpr auto axis = index_select_axis_name::along_f;
    auto input_lay = cldnn::layout(data_types::f32, format::yxfb, { in_size_b, in_size_f, in_size_x, in_size_y });
    auto input = memory::allocate(engine, input_lay);
    auto indices = memory::allocate(engine, { data_types::i32, format::yxfb,{ 1, 1, new_indicies_size, 1 } });

    auto input_data = generate_random_1d<float>(count, 0, 10);
    set_values(input, input_data);

    /*
    Network will be executed 3 times (for 3 different indicies_data).
    */
    std::vector<std::vector<int32_t>> indices_data =
    {
        { 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 }, //for run: 0
    { 1, 1, 3, 3, 2, 2, 4, 4, 0, 0 }, //for run: 1
    { 0, 0, 0, 0, 0, 4, 3, 2, 1, 0 } // for run: 2
    };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        input_layout("indices", indices.get_layout())
    );
    topo.add(
        index_select("index_select", "input", "indices", axis)
    );

    network net(engine, topo);
    net.set_input_data("input", input);
    for (auto const& id : indices_data)
    {
        set_values(indices, id);
        net.set_input_data("indices", indices);
        auto outputs = net.execute();

		ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "index_select");

        auto output_mem = outputs.at("index_select").get_memory();
        auto output_layout = output_mem.get_layout();

        int b_size = output_layout.size.batch[0];
        int f_size = output_layout.size.feature[0];
        int x_size = output_layout.size.spatial[0];
        int y_size = output_layout.size.spatial[1];
        EXPECT_EQ(output_layout.format, format::yxfb);
        EXPECT_EQ(b_size, in_size_b);
        EXPECT_EQ(f_size, new_indicies_size);
        EXPECT_EQ(x_size, in_size_x);
        EXPECT_EQ(y_size, in_size_y);

        auto ref = generate_reference_yxfb(input_data, id, axis, input_lay);

        auto output_ptr = output_mem.pointer<float>();
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            EXPECT_EQ(output_ptr[i], ref[i]);
        }
    }
}

TEST(index_select_gpu, basic_along_x_3_executes_yxfb)
{
    /*
    input: {3, 4, 6, 5}
    indices: {1, 1, 3, 1}
    output: {3, 4, 3, 5}
    */
    engine engine;
    constexpr auto in_size_b = 3;
    constexpr auto in_size_f = 4;
    constexpr auto in_size_x = 6;
    constexpr auto in_size_y = 5;
    constexpr auto count = in_size_b * in_size_f * in_size_x * in_size_y;
    constexpr auto new_indicies_size = 3;
    constexpr auto axis = index_select_axis_name::along_x;
    auto input_lay = cldnn::layout(data_types::f32, format::yxfb, { in_size_b, in_size_f, in_size_x, in_size_y });
    auto input = memory::allocate(engine, input_lay);
    auto indices = memory::allocate(engine, { data_types::i32, format::yxfb,{ 1, 1, new_indicies_size, 1 } });

    auto input_data = generate_random_1d<float>(count, 0, 10);
    set_values(input, input_data);

    /*
    Network will be executed 3 times (for 3 different indicies_data).
    */
    std::vector<std::vector<int32_t>> indices_data =
    {
        { 2, 1, 0 }, //for run: 0
        { 0, 0, 0 }, //for run: 1
        { 1, 1, 0 } // for run: 2
    };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        input_layout("indices", indices.get_layout())
    );
    topo.add(
        index_select("index_select", "input", "indices", axis)
    );

    network net(engine, topo);
    net.set_input_data("input", input);
    for (auto const& id : indices_data)
    {
        set_values(indices, id);
        net.set_input_data("indices", indices);
        auto outputs = net.execute();

		ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "index_select");

        auto output_mem = outputs.at("index_select").get_memory();
        auto output_layout = output_mem.get_layout();

        int b_size = output_layout.size.batch[0];
        int f_size = output_layout.size.feature[0];
        int x_size = output_layout.size.spatial[0];
        int y_size = output_layout.size.spatial[1];
        EXPECT_EQ(output_layout.format, format::yxfb);
        EXPECT_EQ(b_size, in_size_b);
        EXPECT_EQ(f_size, in_size_f);
        EXPECT_EQ(x_size, new_indicies_size);
        EXPECT_EQ(y_size, in_size_y);

        auto ref = generate_reference_yxfb(input_data, id, axis, input_lay);

        auto output_ptr = output_mem.pointer<float>();
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            EXPECT_EQ(output_ptr[i], ref[i]);
        }
    }
}
TEST(index_select_gpu, basic_along_y_3_executes_yxfb)
{
    /*
    input: {2, 4, 4, 3}
    indices: {1, 1, 5, 1}
    output: {2, 4, 4, 5}
    */
    engine engine;
    constexpr auto in_size_b = 2;
    constexpr auto in_size_f = 4;
    constexpr auto in_size_x = 4;
    constexpr auto in_size_y = 3;
    constexpr auto count = in_size_b * in_size_f * in_size_x * in_size_y;
    constexpr auto new_indicies_size = 5;
    constexpr auto axis = index_select_axis_name::along_y;
    auto input_lay = cldnn::layout(data_types::f32, format::yxfb, { in_size_b, in_size_f, in_size_x, in_size_y });
    auto input = memory::allocate(engine, input_lay);
    auto indices = memory::allocate(engine, { data_types::i32, format::yxfb,{ 1, 1, new_indicies_size, 1 } });

    auto input_data = generate_random_1d<float>(count, 0, 10);
    set_values(input, input_data);

    /*
    Network will be executed 3 times (for 3 different indicies_data).
    */
    std::vector<std::vector<int32_t>> indices_data =
    {
        { 0, 1, 2, 2, 1 }, //for run: 0
        { 2, 2, 1, 0, 1 }, //for run: 1
        { 1, 1, 2, 1, 0 } // for run: 2
    };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        input_layout("indices", indices.get_layout())
    );
    topo.add(
        index_select("index_select", "input", "indices", axis)
    );

    network net(engine, topo);
    net.set_input_data("input", input);
    for (auto const& id : indices_data)
    {
        set_values(indices, id);
        net.set_input_data("indices", indices);
        auto outputs = net.execute();

		ASSERT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "index_select");

        auto output_mem = outputs.at("index_select").get_memory();
        auto output_layout = output_mem.get_layout();

        int b_size = output_layout.size.batch[0];
        int f_size = output_layout.size.feature[0];
        int x_size = output_layout.size.spatial[0];
        int y_size = output_layout.size.spatial[1];
        EXPECT_EQ(output_layout.format, format::yxfb);
        EXPECT_EQ(b_size, in_size_b);
        EXPECT_EQ(f_size, in_size_f);
        EXPECT_EQ(x_size, x_size);
        EXPECT_EQ(y_size, new_indicies_size);

        auto ref = generate_reference_yxfb(input_data, id, axis, input_lay);

        auto output_ptr = output_mem.pointer<float>();
        for (size_t i = 0; i < output_ptr.size(); i++)
        {
            EXPECT_EQ(output_ptr[i], ref[i]);
        }
    }
}
