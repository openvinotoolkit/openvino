//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"
#include "test_tools.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

// This function traverses the vector of ops and verifies that each op's dependencies (its inputs)
// is located earlier in the vector. That is enough to be valid
bool validate_list(const vector<shared_ptr<Node>>& nodes)
{
    bool rc = true;
    for (auto it = nodes.rbegin(); it != nodes.rend(); it++)
    {
        auto node_tmp = *it;
        NodeVector dependencies_tmp;
        for (auto& val : node_tmp->input_values())
            dependencies_tmp.emplace_back(val.get_node_shared_ptr());
        vector<Node*> dependencies;

        for (shared_ptr<Node> n : dependencies_tmp)
        {
            dependencies.push_back(n.get());
        }
        auto tmp = it;
        for (tmp++; tmp != nodes.rend(); tmp++)
        {
            auto dep_tmp = *tmp;
            auto found = find(dependencies.begin(), dependencies.end(), dep_tmp.get());
            if (found != dependencies.end())
            {
                dependencies.erase(found);
            }
        }
        if (dependencies.size() > 0)
        {
            rc = false;
        }
    }
    return rc;
}

shared_ptr<Function> make_test_graph()
{
    auto arg_0 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto arg_1 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto arg_2 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto arg_3 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto arg_4 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    auto arg_5 = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    auto t0 = make_shared<op::v1::Add>(arg_0, arg_1);
    auto t1 = make_shared<op::MatMul>(t0, arg_2);
    auto t2 = make_shared<op::v1::Multiply>(t0, arg_3);

    auto t3 = make_shared<op::v1::Add>(t1, arg_4);
    auto t4 = make_shared<op::v1::Add>(t2, arg_5);

    auto r0 = make_shared<op::v1::Add>(t3, t4);

    auto f0 = make_shared<Function>(r0, ParameterVector{arg_0, arg_1, arg_2, arg_3, arg_4, arg_5});

    return f0;
}

template <>
void copy_data<bool>(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<bool>& data)
{
    std::vector<char> data_char(data.begin(), data.end());
    copy_data(tv, data_char);
}

template <>
void init_int_tv<char>(ngraph::runtime::Tensor* tv,
                       std::default_random_engine& engine,
                       char min,
                       char max)
{
    size_t size = tv->get_element_count();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<char> vec(size);
    for (char& element : vec)
    {
        element = static_cast<char>(dist(engine));
    }
    tv->write(vec.data(), vec.size() * sizeof(char));
}

template <>
void init_int_tv<int8_t>(ngraph::runtime::Tensor* tv,
                         std::default_random_engine& engine,
                         int8_t min,
                         int8_t max)
{
    size_t size = tv->get_element_count();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<int8_t> vec(size);
    for (int8_t& element : vec)
    {
        element = static_cast<int8_t>(dist(engine));
    }
    tv->write(vec.data(), vec.size() * sizeof(int8_t));
}

template <>
void init_int_tv<uint8_t>(ngraph::runtime::Tensor* tv,
                          std::default_random_engine& engine,
                          uint8_t min,
                          uint8_t max)
{
    size_t size = tv->get_element_count();
    std::uniform_int_distribution<int16_t> dist(static_cast<short>(min), static_cast<short>(max));
    std::vector<uint8_t> vec(size);
    for (uint8_t& element : vec)
    {
        element = static_cast<uint8_t>(dist(engine));
    }
    tv->write(vec.data(), vec.size() * sizeof(uint8_t));
}

void random_init(ngraph::runtime::Tensor* tv, std::default_random_engine& engine)
{
    element::Type et = tv->get_element_type();
    if (et == element::boolean)
    {
        init_int_tv<char>(tv, engine, 0, 1);
    }
    else if (et == element::f32)
    {
        init_real_tv<float>(tv, engine, numeric_limits<float>::min(), 1.0f);
    }
    else if (et == element::f64)
    {
        init_real_tv<double>(tv, engine, numeric_limits<double>::min(), 1.0);
    }
    else if (et == element::i8)
    {
        init_int_tv<int8_t>(tv, engine, -1, 1);
    }
    else if (et == element::i16)
    {
        init_int_tv<int16_t>(tv, engine, -1, 1);
    }
    else if (et == element::i32)
    {
        init_int_tv<int32_t>(tv, engine, 0, 1);
    }
    else if (et == element::i64)
    {
        init_int_tv<int64_t>(tv, engine, 0, 1);
    }
    else if (et == element::u8)
    {
        init_int_tv<uint8_t>(tv, engine, 0, 1);
    }
    else if (et == element::u16)
    {
        init_int_tv<uint16_t>(tv, engine, 0, 1);
    }
    else if (et == element::u32)
    {
        init_int_tv<uint32_t>(tv, engine, 0, 1);
    }
    else if (et == element::u64)
    {
        init_int_tv<uint64_t>(tv, engine, 0, 1);
    }
    else
    {
        throw runtime_error("unsupported type");
    }
}

template <>
string get_results_str(const std::vector<char>& ref_data,
                       const std::vector<char>& actual_data,
                       size_t max_results)
{
    stringstream ss;
    size_t num_results = std::min(static_cast<size_t>(max_results), ref_data.size());
    ss << "First " << num_results << " results";
    for (size_t i = 0; i < num_results; ++i)
    {
        ss << std::endl
           << std::setw(4) << i << " ref: " << std::setw(16) << std::left
           << static_cast<int>(ref_data[i]) << "  actual: " << std::setw(16) << std::left
           << static_cast<int>(actual_data[i]);
    }
    ss << std::endl;

    return ss.str();
}

::testing::AssertionResult test_ordered_ops(shared_ptr<Function> f, const NodeVector& required_ops)
{
    unordered_set<Node*> seen;
    for (auto& node_ptr : f->get_ordered_ops())
    {
        Node* node = node_ptr.get();
        if (seen.count(node) > 0)
        {
            return ::testing::AssertionFailure() << "Duplication in ordered ops";
        }
        size_t arg_count = node->get_input_size();
        for (size_t i = 0; i < arg_count; ++i)
        {
            Node* dep = node->get_input_node_ptr(i);
            if (seen.count(dep) == 0)
            {
                return ::testing::AssertionFailure()
                       << "Argument " << *dep << " does not occur before op" << *node;
            }
        }
        for (auto& dep_ptr : node->get_control_dependencies())
        {
            if (seen.count(dep_ptr.get()) == 0)
            {
                return ::testing::AssertionFailure()
                       << "Control dependency " << *dep_ptr << " does not occur before op" << *node;
            }
        }
        seen.insert(node);
    }
    for (auto& node_ptr : required_ops)
    {
        if (seen.count(node_ptr.get()) == 0)
        {
            return ::testing::AssertionFailure()
                   << "Required op " << *node_ptr << "does not occur in ordered ops";
        }
    }
    return ::testing::AssertionSuccess();
}

constexpr NodeTypeInfo ngraph::TestOpMultiOut::type_info;

bool ngraph::TestOpMultiOut::evaluate(const HostTensorVector& outputs,
                                      const HostTensorVector& inputs) const
{
    inputs[0]->read(outputs[0]->get_data_ptr(), inputs[0]->get_size_in_bytes());
    inputs[1]->read(outputs[1]->get_data_ptr(), inputs[1]->get_size_in_bytes());
    return true;
}
