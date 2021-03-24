// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <initializer_list>

#include <../api/memory.hpp>



using namespace cldnn;
template<typename T>
void set_values(const cldnn::memory& mem, std::initializer_list<T> args) 
{
    auto ptr = mem.pointer<T>();
    auto it = ptr.begin();
    for (auto x : args)
    {
        *it++ = x;
    }
}

template<typename T>
void set_values(const cldnn::memory& mem, std::vector<T>&& args)
{
    auto ptr = mem.pointer<T>();
    auto it = ptr.begin();
    for (auto x : args)
    {
        *it++ = x;
    }
}

template <typename T>
std::vector<T> get_simple_data(const memory& m)
{
    std::vector<T> data(m.get_layout().get_linear_size());
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = static_cast<T>(i);
    }

    return std::move(data);
}
