// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/random_uniform.hpp"
#include <ctime>
#include <ngraph/validation_util.hpp>
#include "itt.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::RandomUniform, "RandomUniform", 8);

vector<int64_t>
    get_min_max_int(ngraph::element::Type elem_type, const char* min_val, const char* max_val);

op::v8::RandomUniform::RandomUniform(const Output<Node>& out_shape,
                                     const Output<Node>& min_val,
                                     const Output<Node>& max_val,
                                     ngraph::element::Type initial_type,
                                     ngraph::element::Type out_type,
                                     int64_t seed,
                                     int64_t seed2)
    : Op({out_shape, min_val, max_val})
    , m_initial_type(initial_type)
    , m_output_type(out_type)
    , m_seed(seed)
    , m_seed2(seed2)
{
    constructor_validate_and_infer_types();
}

void op::v8::RandomUniform::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v8_RandomUniform_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_integral_number(),
                          "Output shape must be an integral number.");
    PartialShape output_shape = PartialShape::dynamic();
    const auto& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              input_shape.rank() == 1,
                              "The rank of the tensor defining output shape must be equal to 1.");
        if (const auto& const_shape = get_constant_from_source(input_value(0)))
        {
            output_shape = PartialShape(const_shape->cast_vector<int64_t>());
        }
    }

    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(1).compatible(Shape{}), "'min_val' input is not a scalar.");
    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(2).compatible(Shape{}), "'max_val' input is not a scalar.");

    element::Type min_element_type = get_input_element_type(1);
    element::Type max_element_type = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          min_element_type == max_element_type,
                          "'min_val' should have the same type as 'max_val'.");
    NODE_VALIDATION_CHECK(
        this,
        min_element_type == get_out_type(),
        "'min_val' and 'max_val' should have the same type as 'out_type' attribute.");

    if (const auto& const_min = get_constant_from_source(input_value(1)))
    {
        if (const auto& const_max = get_constant_from_source(input_value(2)))
        {
            if (get_out_type() == ngraph::element::Type_t::i64 ||
                get_out_type() == ngraph::element::Type_t::i32)
            {
                int64_t min_val = const_min->cast_vector<int64_t>()[0];
                int64_t max_val = const_max->cast_vector<int64_t>()[0];

                NODE_VALIDATION_CHECK(this,
                                      min_val < max_val,
                                      "Min value must be less than max value. Got "
                                      "min value: ",
                                      min_val,
                                      ", max value: ",
                                      max_val);
            }
            else
            {
                double min_val = const_min->cast_vector<double>()[0];
                double max_val = const_max->cast_vector<double>()[0];

                NODE_VALIDATION_CHECK(this,
                                      min_val < max_val,
                                      "Min value must be less than max value. Got "
                                      "min value: ",
                                      min_val,
                                      ", max value: ",
                                      max_val);
            }
        }
    }

    set_output_type(0, get_out_type(), output_shape);
}

bool op::v8::RandomUniform::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v8_RandomUniform_visit_attributes);
    visitor.on_attribute("initial_type", m_initial_type);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("seed", m_seed);
    visitor.on_attribute("seed2", m_seed2);
    return true;
}

shared_ptr<Node> op::v8::RandomUniform::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v8_Roll_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v8::RandomUniform>(
        new_args[0], new_args[1], new_args[2], m_initial_type, m_output_type, m_seed, m_seed2);
}

std::pair<uint32_t, uint32_t> split_high_low(uint64_t value)
{
    uint32_t low = static_cast<uint32_t>(value);
    uint32_t high = static_cast<uint32_t>(value >> 32);
    return {low, high};
}

uint64_t unite_high_low(uint32_t high, uint32_t low)
{
    return (static_cast<uint64_t>(high) << 32) + low;
}

void calculate_round(const uint64_t& key, uint64_t& counter, uint64_t& n)
{
    auto counter_lr = split_high_low(counter);
    auto key_lr = split_high_low(key);
    auto n_lr = split_high_low(n);
    auto prod0 = split_high_low(static_cast<uint64_t>(0xD2511F53) * n_lr.first);
    auto prod1 = split_high_low(static_cast<uint64_t>(0xCD9E8D57) * counter_lr.first);

    n_lr.first = prod1.second ^ n_lr.second ^ key_lr.first;
    n_lr.second = prod1.first;
    counter_lr.first = prod0.second ^ counter_lr.second ^ key_lr.second;
    counter_lr.second = prod0.first;

    counter = unite_high_low(counter_lr.second, counter_lr.first);
    n = unite_high_low(n_lr.second, n_lr.first);
}

void raise_key(uint64_t& key)
{
    auto key_lr = split_high_low(key);
    key_lr.first += 0x9E3779B9;
    key_lr.second += 0xBB67AE85;
    key = unite_high_low(key_lr.second, key_lr.first);
}

float uint32_to_float(uint32_t x)
{
    uint32_t x_uint32 = (static_cast<uint32_t>(127) << 23) | (x & 0x7fffffu);

    float x_float;
    memcpy(&x_float, &x_uint32, sizeof(x_uint32));
    return x_float - 1.0f;
}

float16 uint32_to_float16(uint32_t x)
{
    uint16_t x_uint16 = static_cast<uint16_t>(x);
    x_uint16 = (static_cast<uint16_t>(15) << 10) | (x_uint16 & 0x3ffu);

    float16 x_float16;
    memcpy(&x_float16, &x_uint16, sizeof(x_uint16));
    return x_float16 - static_cast<float16>(1);
}

double uint32_to_double(uint32_t x1, uint32_t x2)
{
    uint64_t mantissa = ((static_cast<uint64_t>(x1) & 0xfffffu) << 32) | static_cast<uint64_t>(x2);
    uint64_t x_uint64 = ((static_cast<uint64_t>(1023) << 52) | mantissa);

    double x_double;
    memcpy(&x_double, &x_uint64, sizeof(x_uint64));
    return x_double - 1.0;
}

uint64_t uint32_to_uint64(uint32_t x1, uint32_t x2)
{
    return (static_cast<uint64_t>(x2) << 32) | static_cast<uint64_t>(x1);
}

vector<double> generate_float(const vector<uint32_t>& rnd_values,
                              const vector<double>& min_max,
                              ngraph::element::Type initial_type)
{
    vector<double> res(4);
    switch (initial_type)
    {
    case ngraph::element::Type_t::f32:
    {
        auto min_val = static_cast<float>(min_max[0]);
        auto max_val = static_cast<float>(min_max[1]);
        std::transform(rnd_values.begin(),
                       rnd_values.end(),
                       res.begin(),
                       [&min_val, &max_val](const uint32_t& elem) {
                           return uint32_to_float(elem) * (max_val - min_val) + min_val;
                       });
        break;
    }
    case ngraph::element::Type_t::f16:
    {
        auto min_val = static_cast<float16>(min_max[0]);
        auto max_val = static_cast<float16>(min_max[1]);
        std::transform(rnd_values.begin(),
                       rnd_values.end(),
                       res.begin(),
                       [&min_val, &max_val](const uint32_t& elem) {
                           return uint32_to_float16(elem) * (max_val - min_val) + min_val;
                       });
        break;
    }
    case ngraph::element::Type_t::f64:
    {
        res[0] =
            uint32_to_double(rnd_values[0], rnd_values[1]) * (min_max[1] - min_max[0]) + min_max[0];
        res[1] =
            uint32_to_double(rnd_values[2], rnd_values[3]) * (min_max[1] - min_max[0]) + min_max[0];
        break;
    }
    default:
        throw ngraph_error("Unsupported type of RandomUniform: " + initial_type.get_type_name());
    }

    return res;
}

vector<double>
    get_min_max_float(ngraph::element::Type elem_type, const char* min_val, const char* max_val)
{
    if (elem_type == ngraph::element::Type_t::i32 || elem_type == ngraph::element::Type_t::i64)
    {
        vector<int64_t> min_max = get_min_max_int(elem_type, min_val, max_val);
        vector<double> min_max_double(2);
        min_max_double[0] = static_cast<double>(min_max[0]);
        min_max_double[1] = static_cast<double>(min_max[1]);
        return min_max_double;
    }

    vector<double> res(2);
    switch (elem_type)
    {
    case ngraph::element::Type_t::f32:
    {
        float mn[1];
        float mx[1];
        memcpy(mn, min_val, elem_type.size());
        memcpy(mx, max_val, elem_type.size());
        res[0] = static_cast<double>(mn[0]);
        res[1] = static_cast<double>(mx[0]);
        break;
    }
    case ngraph::element::Type_t::f16:
    {
        float16 mn[1];
        float16 mx[1];
        memcpy(mn, min_val, elem_type.size());
        memcpy(mx, max_val, elem_type.size());
        res[0] = static_cast<double>(mn[0]);
        res[1] = static_cast<double>(mx[0]);
        break;
    }
    case ngraph::element::Type_t::bf16:
    {
        bfloat16 mn[1];
        bfloat16 mx[1];
        memcpy(mn, min_val, elem_type.size());
        memcpy(mx, max_val, elem_type.size());
        res[0] = static_cast<double>(mn[0]);
        res[1] = static_cast<double>(mx[0]);
        break;
    }
    case ngraph::element::Type_t::f64:
    {
        memcpy(res.data(), min_val, elem_type.size());
        memcpy(res.data() + 1, max_val, elem_type.size());
        break;
    }
    default: throw ngraph_error("Unsupported type of RandomUniform: " + elem_type.get_type_name());
    }
    return res;
}

vector<int64_t>
    get_min_max_int(ngraph::element::Type elem_type, const char* min_val, const char* max_val)
{
    if (elem_type != ngraph::element::Type_t::i32 && elem_type != ngraph::element::Type_t::i64)
    {
        vector<double> min_max = get_min_max_float(elem_type, min_val, max_val);
        vector<int64_t> min_max_int64(2);
        min_max_int64[0] = static_cast<int64_t>(min_max[0]);
        min_max_int64[1] = static_cast<int64_t>(min_max[1]);
        return min_max_int64;
    }

    vector<int64_t> res(2);
    switch (elem_type)
    {
    case ngraph::element::Type_t::i32:
    {
        int32_t mn[1];
        int32_t mx[1];
        memcpy(mn, min_val, elem_type.size());
        memcpy(mx, max_val, elem_type.size());
        res[0] = mn[0];
        res[1] = mx[0];
        break;
    }
    case ngraph::element::Type_t::i64:
    {
        int64_t mn[1];
        int64_t mx[1];
        memcpy(mn, min_val, elem_type.size());
        memcpy(mx, max_val, elem_type.size());
        res[0] = mn[0];
        res[1] = mx[0];
        break;
    }
    default: throw ngraph_error("Unsupported type of RandomUniform: " + elem_type.get_type_name());
    }
    return res;
}

vector<int64_t> generate_int(const vector<uint32_t>& rnd_values,
                             const vector<int64_t>& min_max,
                             ngraph::element::Type initial_type)
{
    vector<int64_t> res(4);
    switch (initial_type)
    {
    case ngraph::element::Type_t::i32:
    {
        auto min_val = static_cast<int32_t>(min_max[0]);
        auto max_val = static_cast<int32_t>(min_max[1]);
        std::transform(rnd_values.begin(),
                       rnd_values.end(),
                       res.begin(),
                       [&min_val, &max_val](const uint32_t& elem) {
                           return static_cast<int64_t>(elem) % (max_val - min_val) + min_val;
                       });
        break;
    }
    case ngraph::element::Type_t::i64:
    {
        res[0] =
            uint32_to_uint64(rnd_values[0], rnd_values[1]) % (min_max[1] - min_max[0]) + min_max[0];
        res[1] =
            uint32_to_uint64(rnd_values[2], rnd_values[3]) % (min_max[1] - min_max[0]) + min_max[0];
        break;
    default:
        throw ngraph_error("Unsupported type of RandomUniform: " + initial_type.get_type_name());
    }
    }
    return res;
}

void run_philox(uint64_t key, uint64_t counter, uint64_t n, size_t n_rounds, vector<uint32_t>& res)
{
    for (size_t i = 0; i < n_rounds; i++)
    {
        calculate_round(key, counter, n);
        if (i < n_rounds - 1)
            raise_key(key);
    }
    auto res1 = split_high_low(n);
    auto res2 = split_high_low(counter);
    res[0] = res1.first;
    res[1] = res1.second;
    res[2] = res2.first;
    res[3] = res2.second;
}

void cast_float_to_elem_type(const vector<double>& values,
                             size_t idx,
                             size_t step,
                             size_t elem_count,
                             ngraph::element::Type elem_type,
                             char* out)
{
    if (elem_type == ngraph::element::Type_t::f64)
    {
        memcpy(out + idx * elem_type.size(),
               values.data(),
               std::min(size_t(step), elem_count - idx) * elem_type.size());
        return;
    }
    switch (elem_type)
    {
    case ngraph::element::Type_t::bf16:
    {
        bfloat16 res[4];
        std::transform(values.data(), values.data() + step, res, [](const double& elem) {
            return static_cast<bfloat16>(elem);
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());
        break;
    }
    case ngraph::element::Type_t::f16:
    {
        float16 res[4];
        std::transform(values.data(), values.data() + step, res, [](const double& elem) {
            return static_cast<float16>(elem);
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());
        break;
    }
    case ngraph::element::Type_t::f32:
    {
        float res[4];
        std::transform(values.data(), values.data() + step, res, [](const double& elem) {
            return static_cast<float>(elem);
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());

        break;
    }
    case ngraph::element::Type_t::i32:
    {
        int32_t res[4];
        std::transform(values.data(), values.data() + step, res, [](const double& elem) {
            return static_cast<int32_t>(std::round(elem));
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());

        break;
    }
    case ngraph::element::Type_t::i64:
    {
        int64_t res[4];
        std::transform(values.data(), values.data() + step, res, [](const double& elem) {
            return static_cast<int64_t>(std::round(elem));
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());

        break;
    }
    default: throw ngraph_error("Unsupported type of RandomUniform: " + elem_type.get_type_name());
    }
}

void cast_int_to_elem_type(const vector<int64_t>& values,
                           size_t idx,
                           size_t step,
                           size_t elem_count,
                           ngraph::element::Type elem_type,
                           char* out)
{
    if (elem_type == ngraph::element::Type_t::i64)
    {
        memcpy(out + idx * elem_type.size(),
               values.data(),
               std::min(size_t(step), elem_count - idx) * elem_type.size());
        return;
    }

    switch (elem_type)
    {
    case ngraph::element::Type_t::bf16:
    {
        bfloat16 res[4];
        std::transform(values.data(), values.data() + step, res, [](const int64_t& elem) {
            return static_cast<bfloat16>(elem);
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());
        break;
    }
    case ngraph::element::Type_t::f16:
    {
        float16 res[4];
        std::transform(values.data(), values.data() + step, res, [](const int64_t& elem) {
            return static_cast<float16>(elem);
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());

        break;
    }
    case ngraph::element::Type_t::f32:
    {
        float res[4];
        std::transform(values.data(), values.data() + step, res, [](const int64_t& elem) {
            return static_cast<float>(elem);
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());

        break;
    }
    case ngraph::element::Type_t::i32:
    {
        int32_t res[4];
        std::transform(values.data(), values.data() + step, res, [](const int64_t& elem) {
            return static_cast<int32_t>(elem);
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());

        break;
    }
    case ngraph::element::Type_t::f64:
    {
        double res[4];
        std::transform(values.data(), values.data() + step, res, [](const int64_t& elem) {
            return static_cast<double>(elem);
        });
        memcpy(out + idx * elem_type.size(),
               res,
               std::min(size_t(step), elem_count - idx) * elem_type.size());

        break;
    }
    default: throw ngraph_error("Unsupported type of RandomUniform: " + elem_type.get_type_name());
    }
}

void random_uniform(const uint64_t* out_shape,
                    const char* min_val,
                    const char* max_val,
                    char* out,
                    const Shape& out_shape_shape,
                    ngraph::element::Type initial_type,
                    ngraph::element::Type elem_type,
                    uint64_t seed,
                    uint64_t seed2)
{
    if (seed == 0 && seed2 == 0)
    {
        std::srand(std::time(nullptr));
        seed = std::rand();
    }
    uint64_t key = seed;
    uint64_t counter = seed2;
    uint64_t n = 0;
    size_t shape_count = shape_size(out_shape_shape);
    size_t elem_count = 1;
    for (size_t i = 0; i < shape_count; i++)
    {
        elem_count *= out_shape[i];
    }
    size_t step = initial_type.size() > 4 ? 2 : 4;

    for (size_t k = 0; k < elem_count; k += step)
    {
        vector<uint32_t> res(4);
        run_philox(key, counter, n, 10, res);
        switch (initial_type)
        {
        case ngraph::element::Type_t::f16:
        case ngraph::element::Type_t::f32:
        case ngraph::element::Type_t::f64:
        {
            vector<double> min_max = get_min_max_float(elem_type, min_val, max_val);
            vector<double> res_double = generate_float(res, min_max, initial_type);
            cast_float_to_elem_type(res_double, k, step, elem_count, elem_type, out);
            break;
        }
        case ngraph::element::Type_t::i32:
        case ngraph::element::Type_t::i64:
        {
            vector<int64_t> min_max = get_min_max_int(elem_type, min_val, max_val);
            vector<int64_t> res_int = generate_int(res, min_max, initial_type);
            cast_int_to_elem_type(res_int, k, step, elem_count, elem_type, out);
            break;
        }
        default:
            throw ngraph_error("Unsupported type of RandomUniform: " +
                               initial_type.get_type_name());
        }

        if (++n == 0)
            ++counter;
    }
}

bool op::v8::RandomUniform::evaluate(const HostTensorVector& outputs,
                                     const HostTensorVector& inputs) const
{
    NGRAPH_OP_SCOPE(v8_Roll_evaluate);
    const uint64_t* out_shape;
    std::vector<uint64_t> out_shape_uint64;
    out_shape_uint64.resize(shape_size(inputs[0]->get_shape()));

    if (inputs[0]->get_element_type() == element::Type_t::u64)
    {
        out_shape = (uint64_t*)inputs[0]->get_data_ptr<const char>();
    }
    else if (inputs[0]->get_element_type() == element::Type_t::i32)
    {
        auto out_shape_i32 = inputs[0]->get_data_ptr<const int32_t>();
        std::transform(out_shape_i32,
                       out_shape_i32 + shape_size(inputs[0]->get_shape()),
                       out_shape_uint64.begin(),
                       [](const int32_t& elem) { return static_cast<uint64_t>(elem); });
        out_shape = out_shape_uint64.data();
    }
    else if (inputs[0]->get_element_type() == element::Type_t::i64)
    {
        auto out_shape_i64 = inputs[0]->get_data_ptr<const int64_t>();
        std::transform(out_shape_i64,
                       out_shape_i64 + shape_size(inputs[0]->get_shape()),
                       out_shape_uint64.begin(),
                       [](const int64_t& elem) { return static_cast<uint64_t>(elem); });
        out_shape = out_shape_uint64.data();
    }
    else
    {
        throw ngraph_error("Unsupported type of out shape in RandomUniform operation: " +
                           inputs[0]->get_element_type().get_type_name());
    }

    element::Type_t t_out = get_out_type();
    char* out;
    switch (t_out)
    {
    case element::Type_t::i32: out = (char*)outputs[0]->get_data_ptr<const int32_t>(); break;
    case element::Type_t::i64: out = (char*)outputs[0]->get_data_ptr<const int64_t>(); break;
    case element::Type_t::f16: out = (char*)outputs[0]->get_data_ptr<const float16>(); break;
    case element::Type_t::bf16: out = (char*)outputs[0]->get_data_ptr<const bfloat16>(); break;
    case element::Type_t::f32: out = (char*)outputs[0]->get_data_ptr<const float>(); break;
    case element::Type_t::f64: out = (char*)outputs[0]->get_data_ptr<const double>(); break;
    default:
        throw ngraph_error("Unsupported type of RandomUniform: " + get_out_type().get_type_name());
    }

    random_uniform(out_shape,
                   inputs[1]->get_data_ptr<const char>(),
                   inputs[2]->get_data_ptr<const char>(),
                   out,
                   inputs[0]->get_shape(),
                   get_initial_type(),
                   get_out_type(),
                   get_seed(),
                   get_seed2());
    return true;
}