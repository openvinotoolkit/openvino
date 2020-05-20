//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <deque>
#include <forward_list>
#include <iomanip>
#include <map>
#include <numeric>
#include <unordered_set>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

#include <iostream>

using namespace std;
using namespace ngraph;

std::string ngraph::to_cplusplus_sourcecode_literal(bool val)
{
    return val ? "true" : "false";
}

void ngraph::dump(ostream& out, const void* _data, size_t _size)
{
    auto flags = out.flags();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(_data);
    size_t len = _size;
    size_t index = 0;
    while (index < len)
    {
        out << std::hex << std::setw(8) << std::setfill('0') << index;
        for (int i = 0; i < 8; i++)
        {
            if (index + i < len)
            {
                out << " " << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<uint32_t>(data[i]);
            }
            else
            {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 8; i < 16; i++)
        {
            if (index + i < len)
            {
                out << " " << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<uint32_t>(data[i]);
            }
            else
            {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 0; i < 16; i++)
        {
            char ch = (index + i < len ? data[i] : ' ');
            out << ((ch < 32) ? '.' : ch);
        }
        out << "\n";
        data += 16;
        index += 16;
    }
    out.flags(flags);
}

std::string ngraph::to_lower(const std::string& s)
{
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
    return rc;
}

std::string ngraph::to_upper(const std::string& s)
{
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::toupper);
    return rc;
}

string ngraph::trim(const string& s)
{
    string rc = s;
    // trim trailing spaces
    size_t pos = rc.find_last_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(0, pos + 1);
    }

    // trim leading spaces
    pos = rc.find_first_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(pos);
    }
    return rc;
}

vector<string> ngraph::split(const string& src, char delimiter, bool do_trim)
{
    size_t pos;
    string token;
    size_t start = 0;
    vector<string> rc;
    while ((pos = src.find(delimiter, start)) != std::string::npos)
    {
        token = src.substr(start, pos - start);
        start = pos + 1;
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    if (start <= src.size())
    {
        token = src.substr(start);
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    return rc;
}

size_t ngraph::hash_combine(const std::vector<size_t>& list)
{
    size_t seed = 0;
    for (size_t v : list)
    {
        seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

void* ngraph::ngraph_malloc(size_t size)
{
    auto ptr = malloc(size);
    if (size != 0 && !ptr)
    {
        NGRAPH_ERR << "malloc failed to allocate memory of size " << size;
        throw std::bad_alloc();
    }
    return ptr;
}

void ngraph::ngraph_free(void* ptr)
{
    if (ptr)
    {
        free(ptr);
    }
}

size_t ngraph::round_up(size_t size, size_t alignment)
{
    if (alignment == 0)
    {
        return size;
    }

    size_t remainder = size % alignment;
    if (remainder == 0)
    {
        return size;
    }

    return size + alignment - remainder;
}

ngraph::FpropCache ngraph::cache_fprop(std::shared_ptr<ngraph::Function> fprop,
                                       std::shared_ptr<ngraph::Function> bprop)
{
    using namespace ngraph;

    // Create a fprop_cache object to store the results of this analysis
    FpropCache fprop_cache;

    // Traverse bprop to find all of the nodes in the bprop graph
    std::set<Output<Node>> in_bprop;
    ngraph::traverse_nodes(bprop, [&in_bprop](std::shared_ptr<Node> node) {
        for (auto value : node->outputs())
        {
            in_bprop.insert(value);
        }
    });

    // Traverse fprop to make a map that stores parameters with the same
    // shape and element type as the nodes in fprop iff they are in bprop
    // and aren't inputs to bprop
    vector<Output<Node>> bprop_inputs;
    for (auto param : bprop->get_parameters())
    {
        bprop_inputs.push_back(param);
    }
    ngraph::traverse_nodes(
        fprop, [&fprop_cache, &in_bprop, &bprop_inputs](std::shared_ptr<Node> node) {
            for (auto value : node->outputs())
            {
                if (in_bprop.count(value) != 0 &&
                    std::find(bprop_inputs.begin(), bprop_inputs.end(), value) ==
                        bprop_inputs.end())
                {
                    fprop_cache.node_param_map[value] = std::make_shared<op::Parameter>(
                        value.get_element_type(), value.get_shape());
                }
            }
        });

    // clone the nodes in bprop, replacing fprop-related nodes with the
    // intermediate parameters from fprop_cache. This breaks connections in the
    // bprop graph such that only intermediate values from fprop needed by bprop
    // are still connected to the bprop graph as parameters
    ngraph::clone_nodes(bprop->get_ops(), fprop_cache.node_param_map);

    // invert the fprop_cache cloned node map for easy back and for acces.
    std::map<Output<Node>, RawNodeOutput> inverted_node_map;
    for (auto kv : fprop_cache.node_param_map)
    {
        inverted_node_map[kv.second] = kv.first;
    }

    // get cloned bprop results
    ResultVector cloned_results;
    NodeVector result_nodes;
    for (auto node : bprop->get_results())
    {
        auto result = as_type_ptr<op::Result>(
            fprop_cache.node_param_map.at(Output<Node>(node)).get_node_shared_ptr());
        if (!result)
        {
            throw ngraph_error("Expected op::Result values for op::Result keys in node_param_map");
        }
        cloned_results.push_back(result);
        result_nodes.push_back(result);
    }

    // Utility for getting bprop parameters with fprop cache.
    auto get_bprop_params = [&bprop_inputs, &fprop_cache]() {
        // get cloned bprop parameters
        ParameterVector bprop_input_params;
        for (auto param : bprop_inputs)
        {
            bprop_input_params.push_back(as_type_ptr<op::Parameter>(
                fprop_cache.node_param_map.at(Output<Node>(param)).get_node_shared_ptr()));
        }

        // add the cached fprop nodes as inputs to bprop
        for (auto x : fprop_cache.fprop_output_nodes)
        {
            bprop_input_params.push_back(
                as_type_ptr<op::Parameter>(fprop_cache.node_param_map.at(x).get_node_shared_ptr()));
        }
        return bprop_input_params;
    };

    // Traverse the graph from the cloned results of bprop. If we find a parameter
    // that's not an original input of bprop, this is an intermediate value of
    // fprop that needs to be returned from fprop and send to bprop
    auto cloned_bprop_inputs = get_bprop_params();
    ngraph::traverse_nodes(
        result_nodes,
        [&cloned_bprop_inputs, &fprop_cache, &inverted_node_map](std::shared_ptr<Node> node) {
            auto pnode = as_type_ptr<op::Parameter>(node);
            if (pnode &&
                std::find(cloned_bprop_inputs.begin(), cloned_bprop_inputs.end(), pnode) ==
                    cloned_bprop_inputs.end())
            {
                fprop_cache.fprop_output_nodes.push_back(inverted_node_map.at(Output<Node>(node)));
            }
        });

    // create the new outputs for fprop and the new fprop function
    ResultVector fprop_outputs = fprop->get_results();

    for (auto fpirn : fprop_cache.fprop_output_nodes)
    {
        if (as_type_ptr<op::Result>(fpirn.node->shared_from_this()))
        {
            throw ngraph_error("Unexpected op::Result in fprop->get_results()");
        }
        fprop_outputs.push_back(std::make_shared<op::Result>(fpirn));
    }

    fprop_cache.fprop = std::make_shared<Function>(fprop_outputs, fprop->get_parameters());

    // Create the new bprop function with cloned results and cached parameters.
    fprop_cache.bprop = std::make_shared<Function>(cloned_results, get_bprop_params());

    return fprop_cache;
}

size_t stopwatch::get_call_count() const
{
    return m_total_count;
}

size_t stopwatch::get_seconds() const
{
    return chrono::duration_cast<chrono::seconds>(get_timer_value()).count();
}

size_t stopwatch::get_milliseconds() const
{
    return chrono::duration_cast<chrono::milliseconds>(get_timer_value()).count();
}

size_t stopwatch::get_microseconds() const
{
    return chrono::duration_cast<chrono::microseconds>(get_timer_value()).count();
}

size_t stopwatch::get_nanoseconds() const
{
    return get_timer_value().count();
}

chrono::nanoseconds stopwatch::get_timer_value() const
{
    if (m_active)
    {
        return (m_clock.now() - m_start_time);
    }
    else
    {
        return m_last_time;
    }
}

size_t stopwatch::get_total_seconds() const
{
    return chrono::duration_cast<chrono::seconds>(m_total_time).count();
}

size_t stopwatch::get_total_milliseconds() const
{
    return chrono::duration_cast<chrono::milliseconds>(m_total_time).count();
}

size_t stopwatch::get_total_microseconds() const
{
    return chrono::duration_cast<chrono::microseconds>(m_total_time).count();
}

size_t stopwatch::get_total_nanoseconds() const
{
    return m_total_time.count();
}

namespace ngraph
{
    template <>
    float parse_string<float>(const std::string& s)
    {
        const char* tmp = s.c_str();
        char* end;
        float result = strtof(tmp, &end);
        if (*end != 0)
        {
            throw std::runtime_error("Could not parse literal '" + s + "'");
        }
        return result;
    }

    template <>
    double parse_string<double>(const std::string& s)
    {
        const char* tmp = s.c_str();
        char* end;
        double result = strtod(tmp, &end);
        if (*end != 0)
        {
            throw std::runtime_error("Could not parse literal '" + s + "'");
        }
        return result;
    }

    template <>
    int8_t parse_string<int8_t>(const std::string& s)
    {
        char* err;
        int8_t result = strtol(s.c_str(), &err, 10);

        // Check that (1) parsing succeeded and (2) the entire string was used.
        if (*err != 0)
        {
            throw std::runtime_error("Could not parse literal '" + s + "'");
        }

        return result;
    }

    template <>
    uint8_t parse_string<uint8_t>(const std::string& s)
    {
        char* err;
        uint8_t result = strtol(s.c_str(), &err, 10);

        // Check that (1) parsing succeeded and (2) the entire string was used.
        if (*err != 0)
        {
            throw std::runtime_error("Could not parse literal '" + s + "'");
        }

        return result;
    }
}

std::ostream& operator<<(std::ostream& os, const ngraph::NodeVector& nv)
{
    std::vector<std::string> names;
    for (auto n : nv)
    {
        names.push_back(n->get_name());
    }
    os << vector_to_string(names);
    return os;
}

void ngraph::check_fp_values_isinf(const char* name, const float* array, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (std::isinf(array[i]))
        {
            throw std::runtime_error("Discovered Inf in '" + string(name) + "'");
        }
    }
}

void ngraph::check_fp_values_isinf(const char* name, const double* array, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (std::isinf(array[i]))
        {
            throw std::runtime_error("Discovered Inf in '" + string(name) + "'");
        }
    }
}

void ngraph::check_fp_values_isnan(const char* name, const float* array, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (std::isnan(array[i]))
        {
            throw std::runtime_error("Discovered NaN in '" + string(name) + "'");
        }
    }
}

void ngraph::check_fp_values_isnan(const char* name, const double* array, size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        if (std::isnan(array[i]))
        {
            throw std::runtime_error("Discovered NaN in '" + string(name) + "'");
        }
    }
}

bool ngraph::is_valid_permutation(ngraph::AxisVector permutation, ngraph::Rank rank)
{
    std::vector<bool> axis_occurs(permutation.size(), false);

    // Check bounds if rank is static
    if (rank.is_static())
    {
        auto bound = rank.get_length();
        for (auto axis : permutation)
        {
            if (static_cast<decltype(bound)>(axis) >= bound)
            {
                return false;
            }
        }
    }

    for (auto& axis : permutation)
    {
        axis_occurs[axis] = true;
    }

    for (size_t axis = 0; axis < permutation.size(); axis++)
    {
        if (!axis_occurs[axis])
        {
            return false;
        }
    }

    return (rank.is_dynamic() || permutation.size() == rank.get_length());
}

template <typename T>
T ngraph::apply_permutation(T input, AxisVector order)
{
    NGRAPH_CHECK(is_valid_permutation(order, input.size()),
                 "Permutation ",
                 order,
                 " is not valid for ",
                 input);

    T output(input.size());

    for (size_t i = 0; i < order.size(); i++)
    {
        output[i] = input.at(order.at(i));
    }

    return output;
}

template AxisVector ngraph::apply_permutation<AxisVector>(AxisVector input, AxisVector order);
template Shape ngraph::apply_permutation<Shape>(Shape input, AxisVector order);
template ngraph::Coordinate ngraph::apply_permutation<ngraph::Coordinate>(ngraph::Coordinate input,
                                                                          ngraph::AxisVector order);
template ngraph::CoordinateDiff
    ngraph::apply_permutation<ngraph::CoordinateDiff>(ngraph::CoordinateDiff input,
                                                      ngraph::AxisVector order);
template ngraph::Strides ngraph::apply_permutation<ngraph::Strides>(ngraph::Strides input,
                                                                    ngraph::AxisVector order);

namespace ngraph
{
    template <>
    PartialShape apply_permutation(PartialShape input, AxisVector order)
    {
        NGRAPH_CHECK(is_valid_permutation(order, input.rank()),
                     "Permutation ",
                     order,
                     " is not valid for ",
                     input);

        // Here's the special part: if AxisVector is a viable permutation of _some_ rank, and input
        // has dynamic rank, we just stick with dynamic rank.
        if (input.rank().is_dynamic())
        {
            return input;
        }

        PartialShape output{PartialShape::dynamic(order.size())};

        for (size_t i = 0; i < order.size(); i++)
        {
            output[i] = input[order.at(i)];
        }

        return output;
    }
}

AxisVector ngraph::get_default_order(const Shape& shape)
{
    return get_default_order(shape.size());
}

AxisVector ngraph::get_default_order(size_t rank)
{
    AxisVector default_order(rank);
    std::iota(begin(default_order), end(default_order), 0);
    return default_order;
}

AxisVector ngraph::get_permutation_to_default_order(const AxisVector& axis_order)
{
    AxisVector out(axis_order.size());
    for (size_t i = 0; i < axis_order.size(); i++)
    {
        out.at(axis_order[i]) = i;
    }
    return out;
}

void ngraph::parse_version_string(
    std::string version, size_t& major, size_t& minor, size_t& patch, string& extra)
{
    // Since regex is broken in gcc 4.8 I will just manually parse the version string
    // Version strings look like `0.25.0-rc.0+7c32240` or `v0.25.0-rc.0+7c32240`
    size_t start;
    size_t end;
    extra = "";
    start = (version[0] == 'v' ? 1 : 0);
    end = version.find_first_of('.', start);
    string major_str = version.substr(start, end - start);
    start = end + 1;

    end = version.find_first_of('.', start);
    string minor_str = version.substr(start, end - start);
    start = end + 1;

    end = version.find_first_of("-+", start);
    string patch_str = version.substr(start, end - start);
    start = end;

    if (start != string::npos)
    {
        extra = version.substr(start);
    }

    size_t err;
    bool error = false;
    try
    {
        major = stoi(major_str, &err);
        if (err != major_str.size())
        {
            error = true;
        }
        minor = stoi(minor_str, &err);
        if (err != minor_str.size())
        {
            error = true;
        }
        patch = stoi(patch_str, &err);
        if (err != patch_str.size())
        {
            error = true;
        }
    }
    catch (...)
    {
        error = true;
    }
    if (error)
    {
        throw runtime_error("Error parsing version string '" + version + "'");
    }
}

vector<float> read_float_vector(shared_ptr<runtime::Tensor> tv)
{
    vector<float> float_vec;
    element::Type element_type = tv->get_tensor_layout()->get_element_type();

    if (element_type == element::boolean)
    {
        vector<char> vec = read_vector<char>(tv);
        // Changed from vector ctor to explicit for loop to add static_cast
        // This silences MSVC warnings
        for (char value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::bf16)
    {
        vector<bfloat16> vec = read_vector<bfloat16>(tv);
        float_vec = bfloat16::to_float_vector(vec);
    }
    else if (element_type == element::f16)
    {
        vector<float16> vec = read_vector<float16>(tv);
        for (float16 value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::f32)
    {
        vector<float> vec = read_vector<float>(tv);
        for (float value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::f64)
    {
        vector<double> vec = read_vector<double>(tv);
        for (double value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::i8)
    {
        vector<int8_t> vec = read_vector<int8_t>(tv);
        for (int8_t value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::i16)
    {
        vector<int16_t> vec = read_vector<int16_t>(tv);
        for (int16_t value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::i32)
    {
        vector<int32_t> vec = read_vector<int32_t>(tv);
        for (int32_t value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::i64)
    {
        vector<int64_t> vec = read_vector<int64_t>(tv);
        for (int64_t value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::u8)
    {
        vector<uint8_t> vec = read_vector<uint8_t>(tv);
        for (uint8_t value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::u16)
    {
        vector<uint16_t> vec = read_vector<uint16_t>(tv);
        for (uint16_t value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::u32)
    {
        vector<uint32_t> vec = read_vector<uint32_t>(tv);
        for (uint32_t value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else if (element_type == element::u64)
    {
        vector<uint64_t> vec = read_vector<uint64_t>(tv);
        for (uint64_t value : vec)
        {
            float_vec.push_back(static_cast<float>(value));
        }
    }
    else
    {
        throw ngraph_error("Unsupported nGraph element type.");
    }

    return float_vec;
}

vector<int64_t> read_index_vector(shared_ptr<runtime::Tensor> tv)
{
    vector<int64_t> index_vec;
    element::Type element_type = tv->get_tensor_layout()->get_element_type();

    if (element_type == element::boolean)
    {
        vector<char> vec = read_vector<char>(tv);
        // Changed from vector ctor to explicit for loop to add static_cast
        // This silences MSVC warnings
        for (char value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::bf16)
    {
        vector<bfloat16> vec = read_vector<bfloat16>(tv);
        vector<float> float_vec = bfloat16::to_float_vector(vec);
        for (float value : float_vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::f16)
    {
        vector<float16> vec = read_vector<float16>(tv);
        for (float16 value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(static_cast<float>(value)));
        }
    }
    else if (element_type == element::f32)
    {
        vector<float> vec = read_vector<float>(tv);
        for (float value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::f64)
    {
        vector<double> vec = read_vector<double>(tv);
        for (double value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::i8)
    {
        vector<int8_t> vec = read_vector<int8_t>(tv);
        for (int8_t value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::i16)
    {
        vector<int16_t> vec = read_vector<int16_t>(tv);
        for (int16_t value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::i32)
    {
        vector<int32_t> vec = read_vector<int32_t>(tv);
        for (int32_t value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::i64)
    {
        index_vec = read_vector<int64_t>(tv);
    }
    else if (element_type == element::u8)
    {
        vector<uint8_t> vec = read_vector<uint8_t>(tv);
        for (uint8_t value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::u16)
    {
        vector<uint16_t> vec = read_vector<uint16_t>(tv);
        for (uint16_t value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::u32)
    {
        vector<uint32_t> vec = read_vector<uint32_t>(tv);
        for (uint32_t value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else if (element_type == element::u64)
    {
        vector<uint64_t> vec = read_vector<uint64_t>(tv);
        for (uint64_t value : vec)
        {
            index_vec.push_back(static_cast<int64_t>(value));
        }
    }
    else
    {
        throw ngraph_error("Unsupported nGraph element type.");
    }

    return index_vec;
}
