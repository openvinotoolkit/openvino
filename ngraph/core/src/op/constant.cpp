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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
string to_cpp_string(T value)
{
    string rc;
    if (std::isnan(value))
    {
        rc = "NAN";
    }
    else if (std::isinf(value))
    {
        rc = (value > 0 ? "INFINITY" : "-INFINITY");
    }
    else
    {
        stringstream ss;
        ss << value;
        rc = ss.str();
    }
    return rc;
}

constexpr NodeTypeInfo op::Constant::type_info;

op::Constant::Constant(const shared_ptr<runtime::Tensor>& tensor)
    : Constant(tensor->get_element_type(), tensor->get_shape())
{
    tensor->read(get_data_ptr_nc(), tensor->get_size_in_bytes());
    m_all_elements_bitwise_identical = are_all_data_elements_bitwise_identical();
}

op::Constant::Constant(const element::Type& type,
                       const Shape& shape,
                       const std::vector<std::string>& values)
    : Constant(type, shape)
{
    NODE_VALIDATION_CHECK(this,
                          values.size() == shape_size(m_shape) || values.size() == 1,
                          "Did not get the expected number of literals for a constant of shape ",
                          m_shape,
                          " (got ",
                          values.size(),
                          ", expected ",
                          shape_size(m_shape),
                          ".");

    constructor_validate_and_infer_types();

    if (values.size() == 1 && shape_size(m_shape) != 1)
    {
        // broadcast single value
        switch (m_element_type)
        {
        case element::Type_t::boolean:
        {
            bool value = stoi(values[0]) != 0;
            auto target = get_data_ptr_nc<element::Type_t::boolean>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::bf16:
        {
            bfloat16 value = parse_string<float>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::bf16>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::f16:
        {
            float16 value = parse_string<float>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::f16>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::f32:
        {
            float value = parse_string<float>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::f32>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::f64:
        {
            double value = parse_string<double>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::f64>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::i8:
        {
            int8_t value = parse_string<int64_t>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::i8>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::i16:
        {
            int16_t value = parse_string<int64_t>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::i16>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::i32:
        {
            int32_t value = parse_string<int64_t>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::i32>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::i64:
        {
            int64_t value = parse_string<int64_t>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::i64>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::u8:
        {
            uint8_t value = parse_string<uint64_t>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::u8>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::u16:
        {
            uint16_t value = parse_string<uint64_t>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::u16>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::u32:
        {
            uint32_t value = parse_string<uint64_t>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::u32>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::u64:
        {
            uint64_t value = parse_string<uint64_t>(values[0]);
            auto target = get_data_ptr_nc<element::Type_t::u64>();
            std::fill(target, target + shape_size(m_shape), value);
            break;
        }
        case element::Type_t::undefined:
        {
            throw std::runtime_error("deserialize unsupported type undefined");
        }
        case element::Type_t::dynamic:
        {
            throw std::runtime_error("deserialize unsupported type dynamic");
        }
        case element::Type_t::u1:
        {
            throw std::runtime_error("deserialize unsupported type u1");
        }
        }
        m_all_elements_bitwise_identical = true;
    }
    else
    {
        switch (m_element_type)
        {
        case element::Type_t::boolean:
        {
            vector<uint8_t> value = parse_string<uint8_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::boolean>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::bf16:
        {
            vector<float> value = parse_string<float>(values);
            auto target = get_data_ptr_nc<element::Type_t::bf16>();
            for (size_t i = 0; i < value.size(); i++)
            {
                target[i] = value[i];
            }
            break;
        }
        case element::Type_t::f16:
        {
            vector<float> value = parse_string<float>(values);
            auto target = get_data_ptr_nc<element::Type_t::f16>();
            for (size_t i = 0; i < value.size(); i++)
            {
                target[i] = value[i];
            }
            break;
        }
        case element::Type_t::f32:
        {
            vector<float> value = parse_string<float>(values);
            auto target = get_data_ptr_nc<element::Type_t::f32>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::f64:
        {
            vector<double> value = parse_string<double>(values);
            auto target = get_data_ptr_nc<element::Type_t::f64>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::i8:
        {
            vector<int8_t> value = parse_string<int8_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::i8>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::i16:
        {
            vector<int16_t> value = parse_string<int16_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::i16>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::i32:
        {
            vector<int32_t> value = parse_string<int32_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::i32>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::i64:
        {
            vector<int64_t> value = parse_string<int64_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::i64>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::u8:
        {
            vector<uint8_t> value = parse_string<uint8_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::u8>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::u16:
        {
            vector<uint16_t> value = parse_string<uint16_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::u16>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::u32:
        {
            vector<uint32_t> value = parse_string<uint32_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::u32>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::u64:
        {
            vector<uint64_t> value = parse_string<uint64_t>(values);
            auto target = get_data_ptr_nc<element::Type_t::u64>();
            std::copy(value.begin(), value.end(), target);
            break;
        }
        case element::Type_t::undefined:
            throw std::runtime_error("deserialize unsupported type undefined");
        case element::Type_t::dynamic:
            throw std::runtime_error("deserialize unsupported type dynamic");
        case element::Type_t::u1: throw std::runtime_error("deserialize unsupported type u1");
        }
        m_all_elements_bitwise_identical = are_all_data_elements_bitwise_identical();
    }
}

op::Constant::Constant(const element::Type& type, const Shape& shape)
    : m_element_type(type)
    , m_shape(shape)
{
    allocate_buffer();
    constructor_validate_and_infer_types();
}

void op::Constant::allocate_buffer()
{
    m_data = make_shared<runtime::AlignedBuffer>(shape_size(m_shape) * m_element_type.size(),
                                                 host_alignment());
    std::memset(m_data->get_ptr(), 0, m_data->size());
}

op::Constant::Constant(const element::Type& type, const Shape& shape, const void* data)
    : Constant(type, shape)
{
    size_t size = ceil(shape_size(m_shape) * m_element_type.bitwidth() / 8.f);
    std::memcpy(get_data_ptr_nc(), data, size);
    constructor_validate_and_infer_types();
    m_all_elements_bitwise_identical = are_all_data_elements_bitwise_identical();
}

op::Constant::Constant(const Constant& other)
{
    m_element_type = other.m_element_type;
    m_shape = other.m_shape;
    m_data = other.m_data;
    m_all_elements_bitwise_identical = other.m_all_elements_bitwise_identical;
    constructor_validate_and_infer_types();
}

op::Constant::~Constant() {}

string op::Constant::convert_value_to_string(size_t index) const
{
    string rc;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (get_element_type())
    {
    case element::Type_t::boolean: rc = to_string(get_data_ptr<char>()[index]); break;
    case element::Type_t::bf16:
        rc = to_cpp_string(static_cast<float>(get_data_ptr<bfloat16>()[index]));
        break;
    case element::Type_t::f16:
        rc = to_cpp_string(static_cast<float>(get_data_ptr<float16>()[index]));
        break;
    case element::Type_t::f32: rc = to_cpp_string(get_data_ptr<float>()[index]); break;
    case element::Type_t::f64: rc = to_cpp_string(get_data_ptr<double>()[index]); break;
    case element::Type_t::i8: rc = to_string(get_data_ptr<int8_t>()[index]); break;
    case element::Type_t::i16: rc = to_string(get_data_ptr<int16_t>()[index]); break;
    case element::Type_t::i32: rc = to_string(get_data_ptr<int32_t>()[index]); break;
    case element::Type_t::i64: rc = to_string(get_data_ptr<int64_t>()[index]); break;
    case element::Type_t::u1:
        rc = to_string((get_data_ptr<uint8_t>()[index / 8] >> (7 - (index % 8))) & 1);
        break;
    case element::Type_t::u8: rc = to_string(get_data_ptr<uint8_t>()[index]); break;
    case element::Type_t::u16: rc = to_string(get_data_ptr<uint16_t>()[index]); break;
    case element::Type_t::u32: rc = to_string(get_data_ptr<uint32_t>()[index]); break;
    case element::Type_t::u64: rc = to_string(get_data_ptr<uint64_t>()[index]); break;
    case element::Type_t::undefined: throw runtime_error("unsupported type");
    case element::Type_t::dynamic: throw runtime_error("unsupported type");
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
    return rc;
}

vector<string> op::Constant::get_value_strings() const
{
    vector<string> rc;

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (get_element_type())
    {
    case element::Type_t::boolean:
        for (int value : get_vector<char>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::bf16:
        for (bfloat16 value : get_vector<bfloat16>())
        {
            rc.push_back(to_cpp_string(static_cast<float>(value)));
        }
        break;
    case element::Type_t::f16:
        for (float16 value : get_vector<float16>())
        {
            rc.push_back(to_cpp_string(static_cast<float>(value)));
        }
        break;
    case element::Type_t::f32:
        for (float value : get_vector<float>())
        {
            rc.push_back(to_cpp_string(value));
        }
        break;
    case element::Type_t::f64:
        for (double value : get_vector<double>())
        {
            rc.push_back(to_cpp_string(value));
        }
        break;
    case element::Type_t::i8:
        for (int value : get_vector<int8_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::i16:
        for (int value : get_vector<int16_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::i32:
        for (int32_t value : get_vector<int32_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::i64:
        for (int64_t value : get_vector<int64_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u8:
        for (uint32_t value : get_vector<uint8_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u16:
        for (uint32_t value : get_vector<uint16_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u32:
        for (uint32_t value : get_vector<uint32_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u64:
        for (uint64_t value : get_vector<uint64_t>())
        {
            rc.push_back(to_string(value));
        }
        break;
    case element::Type_t::u1: throw runtime_error("unsupported type");
    case element::Type_t::undefined: throw runtime_error("unsupported type");
    case element::Type_t::dynamic: throw runtime_error("unsupported type");
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

    return rc;
}

Shape op::Constant::get_shape_val() const
{
    NGRAPH_CHECK(m_element_type.is_integral_number());
    std::vector<int64_t> out_shape = cast_vector<int64_t>();
    Shape output_shape(shape_size(m_shape));
    std::transform(out_shape.begin(), out_shape.end(), output_shape.begin(), [&](const int64_t& v) {
        return (v > 0) ? v : 0;
    });
    return output_shape;
}

Strides op::Constant::get_strides_val() const
{
    NGRAPH_CHECK(m_element_type == element::i64);
    std::vector<int64_t> out_strides = cast_vector<int64_t>();
    Strides output_strides(shape_size(m_shape));
    std::transform(out_strides.begin(),
                   out_strides.end(),
                   output_strides.begin(),
                   [&](const int64_t& v) { return (v > 0) ? v : 0; });
    return output_strides;
}

Coordinate op::Constant::get_coordinate_val() const
{
    NGRAPH_CHECK(m_element_type == element::i64);
    std::vector<int64_t> out_coordinate = cast_vector<int64_t>();
    Coordinate output_coordinate(shape_size(m_shape));
    std::transform(out_coordinate.begin(),
                   out_coordinate.end(),
                   output_coordinate.begin(),
                   [&](const int64_t& v) { return (v > 0) ? v : 0; });
    return output_coordinate;
}

CoordinateDiff op::Constant::get_coordinate_diff_val() const
{
    NGRAPH_CHECK(m_element_type == element::i64);
    std::vector<int64_t> out_coordinate_diff = cast_vector<int64_t>();
    CoordinateDiff output_coordinate_diff(shape_size(m_shape));
    std::transform(out_coordinate_diff.begin(),
                   out_coordinate_diff.end(),
                   output_coordinate_diff.begin(),
                   [&](const int64_t& v) { return (v > 0) ? v : 0; });
    return output_coordinate_diff;
}

AxisVector op::Constant::get_axis_vector_val() const
{
    NGRAPH_CHECK(m_element_type.is_integral_number());
    std::vector<int64_t> out_axis_vector = cast_vector<int64_t>();
    AxisVector output_axis_vector(shape_size(m_shape));
    std::transform(out_axis_vector.begin(),
                   out_axis_vector.end(),
                   output_axis_vector.begin(),
                   [&](const int64_t& v) { return (v > 0) ? v : 0; });
    return output_axis_vector;
}

AxisSet op::Constant::get_axis_set_val() const
{
    NGRAPH_CHECK(m_element_type.is_integral_number());
    std::vector<int64_t> out_axis_set = cast_vector<int64_t>();
    AxisSet output_axis_set;
    for (auto& axis : out_axis_set)
    {
        output_axis_set.insert(axis > 0 ? axis : 0);
    }
    return output_axis_set;
}

void op::Constant::set_data_shape(const Shape& shape)
{
    NGRAPH_CHECK(shape_size(shape) == shape_size(m_shape));
    m_shape = shape;
}

shared_ptr<Node> op::Constant::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v0_Constant_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Constant>(*this);
}

template <typename T>
static bool test_bitwise_identical(const op::Constant* constant)
{
    const size_t size = shape_size(constant->get_shape());
    bool data_is_constant = true;
    if (size > 0)
    {
        const T* data = constant->get_data_ptr<T>();
        const T compare = data[0];
        for (size_t i = 1; i < size; i++)
        {
            if (data[i] != compare)
            {
                data_is_constant = false;
                break;
            }
        }
    }
    return data_is_constant;
}

bool op::Constant::are_all_data_elements_bitwise_identical() const
{
    bool rc = false;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (get_element_type())
    {
    case element::Type_t::boolean:
    case element::Type_t::i8:
    case element::Type_t::u8:
    {
        rc = test_bitwise_identical<uint8_t>(this);
        break;
    }
    case element::Type_t::bf16:
    case element::Type_t::f16:
    case element::Type_t::i16:
    case element::Type_t::u16:
    {
        rc = test_bitwise_identical<uint16_t>(this);
        break;
    }
    case element::Type_t::f32:
    case element::Type_t::i32:
    case element::Type_t::u32:
    {
        rc = test_bitwise_identical<uint32_t>(this);
        break;
    }
    case element::Type_t::f64:
    case element::Type_t::i64:
    case element::Type_t::u64:
    {
        rc = test_bitwise_identical<uint64_t>(this);
        break;
    }
    case element::Type_t::u1:
    case element::Type_t::undefined:
    case element::Type_t::dynamic: break;
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
    return rc;
}

bool op::v0::Constant::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v0_Constant_visit_attributes);
    Shape prev_shape = m_shape;
    element::Type prev_type = m_element_type;
    visitor.on_attribute("element_type", m_element_type);
    visitor.on_attribute("shape", m_shape);

    bool need_to_reallocate = (m_shape != prev_shape || prev_type != m_element_type);
    if (m_alloc_buffer_on_visit_attributes && need_to_reallocate)
    {
        // Filling in a fresh constant
        allocate_buffer();
    }
    visitor.on_attribute("value", m_data);
    return true;
}

bool op::v0::Constant::evaluate(const HostTensorVector& outputs,
                                const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v0_Constant_evaluate);
    auto output = outputs[0];
    output->write(get_data_ptr(), output->get_size_in_bytes());
    return true;
}

bool op::v0::Constant::evaluate_lower(const HostTensorVector& outputs) const
{
    return evaluate(outputs, {});
}
bool op::v0::Constant::evaluate_upper(const HostTensorVector& outputs) const
{
    return evaluate(outputs, {});
}

//
// We have to open up namespace blocks here to work around a problem with gcc:
//
// https://stackoverflow.com/questions/25594644/warning-specialization-of-template-in-different-namespace
//
namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            template <>
            void Constant::write_to_buffer<string>(const element::Type& /* target_type */,
                                                   const Shape& /* target_shape */,
                                                   const vector<string>& /* source */,
                                                   void* /* target */,
                                                   size_t /* target_element_count */)
            {
            }
        }
    }
}
