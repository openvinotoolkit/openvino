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
    const auto& engine = get_test_engine();
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
    const auto& engine = get_test_engine();
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
    const auto& engine = get_test_engine();
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
    const auto& engine = get_test_engine();
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
    const auto& engine = get_test_engine();
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
    const auto& engine = get_test_engine();
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
    const auto& engine = get_test_engine();
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
    const auto& engine = get_test_engine();
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

TEST(index_select_gpu, reverse_along_b_bfyx)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 4, 2 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,

        8.f,  9.f, 10.f, 11.f,
        12.f, 13.f, 14.f, 15.f,



        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,

        24.f, 25.f, 26.f, 27.f,
        28.f, 29.f, 30.f, 31.f,
    };

    std::vector<float> out_data = {
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,

        24.f, 25.f, 26.f, 27.f,
        28.f, 29.f, 30.f, 31.f,

        

        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,

        8.f,  9.f, 10.f, 11.f,
        12.f, 13.f, 14.f, 15.f,
    };

    constexpr auto axis = index_select_axis_name::along_b;

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);
    
    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_f_bfyx)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 3, 4 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f
    };

    std::vector<float> out_data = {
        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,

        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f
    };

    constexpr auto axis = index_select_axis_name::along_f;

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_y_bfyx)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 4, 3 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f
    };

    std::vector<float> out_data = {
        8.f,  9.f, 10.f, 11.f,
        4.f,  5.f,  6.f,  7.f,
        0.f,  1.f,  2.f,  3.f,
        
        20.f, 21.f, 22.f, 23.f,
        16.f, 17.f, 18.f, 19.f,
        12.f, 13.f, 14.f, 15.f
    };

    constexpr auto axis = index_select_axis_name::along_y;

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_x_bfyx)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 4, 3 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f
    };

    std::vector<float> out_data = {
        3.f,  2.f,  1.f,  0.f,
        7.f,  6.f,  5.f,  4.f,
        11.f,  10.f, 9.f, 8.f,

        15.f, 14.f, 13.f, 12.f,
        19.f, 18.f, 17.f, 16.f,
        23.f, 22.f, 21.f, 20.f
    };

    constexpr auto axis = index_select_axis_name::along_x;

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}


TEST(index_select_gpu, reverse_along_y_yxfb)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 4, 2, 2, 2 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,

        8.f,  9.f, 10.f, 11.f,
        12.f, 13.f, 14.f, 15.f,



        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,

        24.f, 25.f, 26.f, 27.f,
        28.f, 29.f, 30.f, 31.f,
    };

    std::vector<float> out_data = {
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,

        24.f, 25.f, 26.f, 27.f,
        28.f, 29.f, 30.f, 31.f,



        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,

        8.f,  9.f, 10.f, 11.f,
        12.f, 13.f, 14.f, 15.f,
    };

    constexpr auto axis = index_select_axis_name::along_y;

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_x_yxfb)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 3, 4, 2, 1 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f
    };

    std::vector<float> out_data = {
        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,

        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f
    };

    constexpr auto axis = index_select_axis_name::along_x;

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_f_yxfb)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 4, 3, 2, 1 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f
    };

    std::vector<float> out_data = {
        8.f,  9.f, 10.f, 11.f,
        4.f,  5.f,  6.f,  7.f,
        0.f,  1.f,  2.f,  3.f,

        20.f, 21.f, 22.f, 23.f,
        16.f, 17.f, 18.f, 19.f,
        12.f, 13.f, 14.f, 15.f
    };

    constexpr auto axis = index_select_axis_name::along_f;

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_b_yxfb)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 4, 3, 2, 1 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f
    };

    std::vector<float> out_data = {
        3.f,  2.f,  1.f,  0.f,
        7.f,  6.f,  5.f,  4.f,
        11.f,  10.f, 9.f, 8.f,

        15.f, 14.f, 13.f, 12.f,
        19.f, 18.f, 17.f, 16.f,
        23.f, 22.f, 21.f, 20.f
    };

    constexpr auto axis = index_select_axis_name::along_b;

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}


TEST(index_select_gpu, reverse_along_yx_bfyx)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 4, 3 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f
    };

    std::vector<float> out_data = {
        11.f,  10.f, 9.f, 8.f,
        7.f,  6.f,  5.f,  4.f,
        3.f,  2.f,  1.f,  0.f,

        23.f, 22.f, 21.f, 20.f,
        19.f, 18.f, 17.f, 16.f,
        15.f, 14.f, 13.f, 12.f
    };

    std::vector<index_select_axis_name> axis = { index_select_axis_name::along_y, index_select_axis_name::along_x };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_fyx_bfyx)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 2, 4, 3 } });

    std::vector<float> input_data = {
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f
    };

    std::vector<float> out_data = {
        23.f, 22.f, 21.f, 20.f,
        19.f, 18.f, 17.f, 16.f,
        15.f, 14.f, 13.f, 12.f,

        11.f,  10.f, 9.f, 8.f,
        7.f,  6.f,  5.f,  4.f,
        3.f,  2.f,  1.f,  0.f
    };

    std::vector<index_select_axis_name> axis = { index_select_axis_name::along_f, index_select_axis_name::along_y, index_select_axis_name::along_x };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_bfyx_bfyx)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 3, 3, 4, 3 } });

    std::vector<float> input_data = {
        // b0f0
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,
        // f1
        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,
        // f2
        24.f, 25.f, 26.f, 27.f,
        28.f, 29.f, 30.f, 31.f,
        32.f, 33.f, 34.f, 35.f,

        // b1f0
        36.f, 37.f, 38.f, 39.f,
        40.f, 41.f, 42.f, 43.f,
        44.f, 45.f, 46.f, 47.f,
        // f1
        48.f, 49.f, 50.f, 51.f,
        52.f, 53.f, 54.f, 55.f,
        56.f, 57.f, 58.f, 59.f,
        // f2
        60.f, 61.f, 62.f, 63.f,
        64.f, 65.f, 66.f, 67.f,
        68.f, 69.f, 70.f, 71.f,

        // b2f0
        72.f, 73.f, 74.f, 75.f,
        76.f, 77.f, 78.f, 79.f,
        80.f, 81.f, 82.f, 83.f,
        // f1
        84.f, 85.f, 86.f, 87.f,
        88.f, 89.f, 90.f, 91.f,
        92.f, 93.f, 94.f, 95.f,
        // f2
        96.f, 97.f, 98.f, 99.f,
        100.f, 101.f, 102.f, 103.f,
        104.f, 105.f, 106.f, 107.f
    };

    std::vector<float> out_data = {
        107.f, 106.f, 105.f, 104.f,
        103.f, 102.f, 101.f, 100.f,
        99.f, 98.f, 97.f, 96.f,

        95.f, 94.f, 93.f, 92.f,
        91.f, 90.f, 89.f, 88.f,
        87.f, 86.f, 85.f, 84.f,

        83.f, 82.f, 81.f, 80.f,
        79.f, 78.f, 77.f, 76.f,
        75.f, 74.f, 73.f, 72.f,


        71.f, 70.f, 69.f, 68.f,
        67.f, 66.f, 65.f, 64.f,
        63.f, 62.f, 61.f, 60.f,

        59.f, 58.f, 57.f, 56.f,
        55.f, 54.f, 53.f, 52.f,
        51.f, 50.f, 49.f, 48.f,

        47.f, 46.f, 45.f, 44.f,
        43.f, 42.f, 41.f, 40.f,
        39.f, 38.f, 37.f, 36.f,

        
        35.f, 34.f, 33.f, 32.f,
        31.f, 30.f, 29.f, 28.f,
        27.f, 26.f, 25.f, 24.f,
        
        23.f, 22.f, 21.f, 20.f,
        19.f, 18.f, 17.f, 16.f,
        15.f, 14.f, 13.f, 12.f,

        11.f,  10.f, 9.f, 8.f,
        7.f,  6.f,  5.f,  4.f,
        3.f,  2.f,  1.f,  0.f
    };

    std::vector<index_select_axis_name> axis = { index_select_axis_name::along_b, index_select_axis_name::along_f, index_select_axis_name::along_y, index_select_axis_name::along_x };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_bfx_yxfb)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 4, 3, 3, 3 } });

    std::vector<float> input_data = {
        // y0x0
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,
        // x1
        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,
        // x2
        24.f, 25.f, 26.f, 27.f,
        28.f, 29.f, 30.f, 31.f,
        32.f, 33.f, 34.f, 35.f,

        // y1x0
        36.f, 37.f, 38.f, 39.f,
        40.f, 41.f, 42.f, 43.f,
        44.f, 45.f, 46.f, 47.f,
        // x1
        48.f, 49.f, 50.f, 51.f,
        52.f, 53.f, 54.f, 55.f,
        56.f, 57.f, 58.f, 59.f,
        // x2
        60.f, 61.f, 62.f, 63.f,
        64.f, 65.f, 66.f, 67.f,
        68.f, 69.f, 70.f, 71.f,

        // y2x0
        72.f, 73.f, 74.f, 75.f,
        76.f, 77.f, 78.f, 79.f,
        80.f, 81.f, 82.f, 83.f,
        // x1
        84.f, 85.f, 86.f, 87.f,
        88.f, 89.f, 90.f, 91.f,
        92.f, 93.f, 94.f, 95.f,
        // x2
        96.f, 97.f, 98.f, 99.f,
        100.f, 101.f, 102.f, 103.f,
        104.f, 105.f, 106.f, 107.f
    };

    std::vector<float> out_data = {
        35.f, 34.f, 33.f, 32.f,
        31.f, 30.f, 29.f, 28.f,
        27.f, 26.f, 25.f, 24.f,

        23.f, 22.f, 21.f, 20.f,
        19.f, 18.f, 17.f, 16.f,
        15.f, 14.f, 13.f, 12.f,

        11.f,  10.f, 9.f, 8.f,
        7.f,  6.f,  5.f,  4.f,
        3.f,  2.f,  1.f,  0.f,


        71.f, 70.f, 69.f, 68.f,
        67.f, 66.f, 65.f, 64.f,
        63.f, 62.f, 61.f, 60.f,

        59.f, 58.f, 57.f, 56.f,
        55.f, 54.f, 53.f, 52.f,
        51.f, 50.f, 49.f, 48.f,

        47.f, 46.f, 45.f, 44.f,
        43.f, 42.f, 41.f, 40.f,
        39.f, 38.f, 37.f, 36.f,


        107.f, 106.f, 105.f, 104.f,
        103.f, 102.f, 101.f, 100.f,
        99.f, 98.f, 97.f, 96.f,

        95.f, 94.f, 93.f, 92.f,
        91.f, 90.f, 89.f, 88.f,
        87.f, 86.f, 85.f, 84.f,

        83.f, 82.f, 81.f, 80.f,
        79.f, 78.f, 77.f, 76.f,
        75.f, 74.f, 73.f, 72.f
    };

    std::vector<index_select_axis_name> axis = { index_select_axis_name::along_f, index_select_axis_name::along_b, index_select_axis_name::along_x };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}

TEST(index_select_gpu, reverse_along_bfyx_yxfb)
{
    const auto& engine = get_test_engine();
    auto input = memory::allocate(engine, { data_types::f32, format::yxfb,{ 4, 3, 3, 3 } });

    std::vector<float> input_data = {
        // y0x0
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,
        // x1
        12.f, 13.f, 14.f, 15.f,
        16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f,
        // x2
        24.f, 25.f, 26.f, 27.f,
        28.f, 29.f, 30.f, 31.f,
        32.f, 33.f, 34.f, 35.f,

        // y1x0
        36.f, 37.f, 38.f, 39.f,
        40.f, 41.f, 42.f, 43.f,
        44.f, 45.f, 46.f, 47.f,
        // x1
        48.f, 49.f, 50.f, 51.f,
        52.f, 53.f, 54.f, 55.f,
        56.f, 57.f, 58.f, 59.f,
        // x2
        60.f, 61.f, 62.f, 63.f,
        64.f, 65.f, 66.f, 67.f,
        68.f, 69.f, 70.f, 71.f,

        // y2x0
        72.f, 73.f, 74.f, 75.f,
        76.f, 77.f, 78.f, 79.f,
        80.f, 81.f, 82.f, 83.f,
        // x1
        84.f, 85.f, 86.f, 87.f,
        88.f, 89.f, 90.f, 91.f,
        92.f, 93.f, 94.f, 95.f,
        // x2
        96.f, 97.f, 98.f, 99.f,
        100.f, 101.f, 102.f, 103.f,
        104.f, 105.f, 106.f, 107.f
    };

    std::vector<float> out_data = {
        107.f, 106.f, 105.f, 104.f,
        103.f, 102.f, 101.f, 100.f,
        99.f, 98.f, 97.f, 96.f,

        95.f, 94.f, 93.f, 92.f,
        91.f, 90.f, 89.f, 88.f,
        87.f, 86.f, 85.f, 84.f,

        83.f, 82.f, 81.f, 80.f,
        79.f, 78.f, 77.f, 76.f,
        75.f, 74.f, 73.f, 72.f,


        71.f, 70.f, 69.f, 68.f,
        67.f, 66.f, 65.f, 64.f,
        63.f, 62.f, 61.f, 60.f,

        59.f, 58.f, 57.f, 56.f,
        55.f, 54.f, 53.f, 52.f,
        51.f, 50.f, 49.f, 48.f,

        47.f, 46.f, 45.f, 44.f,
        43.f, 42.f, 41.f, 40.f,
        39.f, 38.f, 37.f, 36.f,


        35.f, 34.f, 33.f, 32.f,
        31.f, 30.f, 29.f, 28.f,
        27.f, 26.f, 25.f, 24.f,

        23.f, 22.f, 21.f, 20.f,
        19.f, 18.f, 17.f, 16.f,
        15.f, 14.f, 13.f, 12.f,

        11.f,  10.f, 9.f, 8.f,
        7.f,  6.f,  5.f,  4.f,
        3.f,  2.f,  1.f,  0.f
    };

    std::vector<index_select_axis_name> axis = { index_select_axis_name::along_b, index_select_axis_name::along_f, index_select_axis_name::along_y, index_select_axis_name::along_x };

    topology topo;
    topo.add(
        input_layout("input", input.get_layout())
    );
    topo.add(
        index_select("index_select", "input", axis)
    );

    network net(engine, topo);

    set_values(input, input_data);
    net.set_input_data("input", input);

    auto outputs = net.execute();
    auto output_mem = outputs.at("index_select").get_memory();
    auto output_ptr = output_mem.pointer<float>();

    for (size_t i = 0; i < output_ptr.size(); i++)
    {
        EXPECT_EQ(output_ptr[i], out_data[i]);
    }
}