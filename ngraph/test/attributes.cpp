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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

NGRAPH_SUPPRESS_DEPRECATED_START

TEST(attributes, value_map)
{
    ValueMap value_map;
    bool a = true;
    int8_t b = 2;
    value_map.insert("a", a);
    value_map.insert("b", b);
    bool g_a = value_map.get<bool>("a");
    int8_t g_b = value_map.get<int8_t>("b");
    EXPECT_EQ(a, g_a);
    EXPECT_EQ(b, g_b);
}

enum class TuringModel
{
    XL400,
    XL1200
};

namespace ngraph
{
    template <>
    EnumNames<TuringModel>& EnumNames<TuringModel>::get()
    {
        static auto enum_names = EnumNames<TuringModel>(
            "TuringModel", {{"XL400", TuringModel::XL400}, {"XL1200", TuringModel::XL1200}});
        return enum_names;
    }

    template <>
    class AttributeAdapter<TuringModel> : public EnumAttributeAdapterBase<TuringModel>
    {
    public:
        AttributeAdapter(TuringModel& value)
            : EnumAttributeAdapterBase<TuringModel>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<TuringModel>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    constexpr DiscreteTypeInfo AttributeAdapter<TuringModel>::type_info;

    struct Position
    {
        float x;
        float y;
        float z;
        bool operator==(const Position& p) const { return x == p.x && y == p.y && z == p.z; }
        Position& operator=(const Position& p)
        {
            x = p.x;
            y = p.y;
            z = p.z;
            return *this;
        }
    };

    template <>
    class AttributeAdapter<Position> : public VisitorAdapter
    {
    public:
        AttributeAdapter(Position& value)
            : m_ref(value)
        {
        }
        bool visit_attributes(AttributeVisitor& visitor) override
        {
            visitor.on_attribute("x", m_ref.x);
            visitor.on_attribute("y", m_ref.y);
            visitor.on_attribute("z", m_ref.z);
            return true;
        }
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<Position>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }

    protected:
        Position& m_ref;
    };

    constexpr DiscreteTypeInfo AttributeAdapter<Position>::type_info;
}

// Given a Turing machine program and data, return scalar 1 if the program would
// complete, 1 if it would not.
class Oracle : public op::Op
{
public:
    Oracle(const Output<Node>& program,
           const Output<Node>& data,
           TuringModel turing_model,
           const element::Type element_type,
           element::Type_t element_type_t,
           const string& val_string,
           bool val_bool,
           float val_float,
           double val_double,
           uint8_t val_uint8_t,
           uint16_t val_uint16_t,
           uint32_t val_uint32_t,
           uint64_t val_uint64_t,
           int8_t val_int8_t,
           int16_t val_int16_t,
           int32_t val_int32_t,
           int64_t val_int64_t,
           size_t val_size_t,
           const std::vector<std::string>& vec_string,
           const std::vector<float>& vec_float,
           const std::vector<double>& vec_double,
           const std::vector<uint8_t>& vec_uint8_t,
           const std::vector<uint16_t>& vec_uint16_t,
           const std::vector<uint32_t>& vec_uint32_t,
           const std::vector<uint64_t>& vec_uint64_t,
           const std::vector<int8_t>& vec_int8_t,
           const std::vector<int16_t>& vec_int16_t,
           const std::vector<int32_t>& vec_int32_t,
           const std::vector<int64_t>& vec_int64_t,
           const std::vector<size_t>& vec_size_t,
           const Position& position,
           const shared_ptr<Node>& node,
           const NodeVector& node_vector,
           const ParameterVector& parameter_vector,
           const ResultVector& result_vector)
        : Op({program, data})
        , m_turing_model(turing_model)
        , m_element_type(element_type)
        , m_element_type_t(element_type_t)
        , m_val_string(val_string)
        , m_val_bool(val_bool)
        , m_val_float(val_float)
        , m_val_double(val_double)
        , m_val_uint8_t(val_uint8_t)
        , m_val_uint16_t(val_uint16_t)
        , m_val_uint32_t(val_uint32_t)
        , m_val_uint64_t(val_uint64_t)
        , m_val_int8_t(val_int8_t)
        , m_val_int16_t(val_int16_t)
        , m_val_int32_t(val_int32_t)
        , m_val_int64_t(val_int64_t)
        , m_val_size_t(val_size_t)
        , m_vec_string(vec_string)
        , m_vec_float(vec_float)
        , m_vec_double(vec_double)
        , m_vec_uint8_t(vec_uint8_t)
        , m_vec_uint16_t(vec_uint16_t)
        , m_vec_uint32_t(vec_uint32_t)
        , m_vec_uint64_t(vec_uint64_t)
        , m_vec_int8_t(vec_int8_t)
        , m_vec_int16_t(vec_int16_t)
        , m_vec_int32_t(vec_int32_t)
        , m_vec_int64_t(vec_int64_t)
        , m_vec_size_t(vec_size_t)
        , m_position(position)
        , m_node(node)
        , m_node_vector(node_vector)
        , m_parameter_vector(parameter_vector)
        , m_result_vector(result_vector)
    {
    }

    static constexpr NodeTypeInfo type_info{"Oracle", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    Oracle() = default;

    TuringModel get_turing_model() const { return m_turing_model; }
    const element::Type get_element_type() const { return m_element_type; }
    const element::Type_t get_element_type_t() const { return m_element_type_t; }
    const string& get_val_string() const { return m_val_string; }
    bool get_val_bool() const { return m_val_bool; }
    bool get_val_float() const { return m_val_float; }
    bool get_val_double() const { return m_val_double; }
    uint64_t get_val_uint8_t() const { return m_val_uint8_t; }
    uint64_t get_val_uint16_t() const { return m_val_uint16_t; }
    uint64_t get_val_uint32_t() const { return m_val_uint32_t; }
    uint64_t get_val_uint64_t() const { return m_val_uint64_t; }
    int64_t get_val_int8_t() const { return m_val_int8_t; }
    int64_t get_val_int16_t() const { return m_val_int16_t; }
    int64_t get_val_int32_t() const { return m_val_int32_t; }
    int64_t get_val_int64_t() const { return m_val_int64_t; }
    size_t get_val_size_t() const { return m_val_size_t; }
    const vector<uint8_t>& get_vec_uint8_t() const { return m_vec_uint8_t; }
    const vector<uint16_t>& get_vec_uint16_t() const { return m_vec_uint16_t; }
    const vector<uint32_t>& get_vec_uint32_t() const { return m_vec_uint32_t; }
    const vector<uint64_t>& get_vec_uint64_t() const { return m_vec_uint64_t; }
    const vector<int8_t>& get_vec_int8_t() const { return m_vec_int8_t; }
    const vector<int16_t>& get_vec_int16_t() const { return m_vec_int16_t; }
    const vector<int32_t>& get_vec_int32_t() const { return m_vec_int32_t; }
    const vector<int64_t>& get_vec_int64_t() const { return m_vec_int64_t; }
    const vector<string>& get_vec_string() const { return m_vec_string; }
    const vector<float>& get_vec_float() const { return m_vec_float; }
    const vector<double>& get_vec_double() const { return m_vec_double; }
    const vector<size_t>& get_vec_size_t() const { return m_vec_size_t; }
    const Position& get_position() const { return m_position; }
    const shared_ptr<Node>& get_node() const { return m_node; }
    const NodeVector& get_node_vector() const { return m_node_vector; }
    const ParameterVector& get_parameter_vector() const { return m_parameter_vector; }
    const ResultVector& get_result_vector() const { return m_result_vector; }
    shared_ptr<Node> clone_with_new_inputs(const OutputVector& args) const override
    {
        return make_shared<Oracle>(args[0],
                                   args[1],
                                   m_turing_model,
                                   m_element_type,
                                   m_element_type_t,
                                   m_val_string,
                                   m_val_bool,
                                   m_val_float,
                                   m_val_double,
                                   m_val_uint8_t,
                                   m_val_uint16_t,
                                   m_val_uint32_t,
                                   m_val_uint64_t,
                                   m_val_int8_t,
                                   m_val_int16_t,
                                   m_val_int32_t,
                                   m_val_int64_t,
                                   m_val_size_t,
                                   m_vec_string,
                                   m_vec_float,
                                   m_vec_double,
                                   m_vec_uint8_t,
                                   m_vec_uint16_t,
                                   m_vec_uint32_t,
                                   m_vec_uint64_t,
                                   m_vec_int8_t,
                                   m_vec_int16_t,
                                   m_vec_int32_t,
                                   m_vec_int64_t,
                                   m_vec_size_t,
                                   m_position,
                                   m_node,
                                   m_node_vector,
                                   m_parameter_vector,
                                   m_result_vector);
    }

    void validate_and_infer_types() override { set_output_type(0, element::i64, {}); }
    bool visit_attributes(AttributeVisitor& visitor) override
    {
        visitor.on_attribute("turing_model", m_turing_model);
        visitor.on_attribute("element_type", m_element_type);
        visitor.on_attribute("element_type_t", m_element_type_t);
        visitor.on_attribute("val_string", m_val_string);
        visitor.on_attribute("val_bool", m_val_bool);
        visitor.on_attribute("val_float", m_val_float);
        visitor.on_attribute("val_double", m_val_double);
        visitor.on_attribute("val_uint8_t", m_val_uint8_t);
        visitor.on_attribute("val_uint16_t", m_val_uint16_t);
        visitor.on_attribute("val_uint32_t", m_val_uint32_t);
        visitor.on_attribute("val_uint64_t", m_val_uint64_t);
        visitor.on_attribute("val_int8_t", m_val_int8_t);
        visitor.on_attribute("val_int16_t", m_val_int16_t);
        visitor.on_attribute("val_int32_t", m_val_int32_t);
        visitor.on_attribute("val_int64_t", m_val_int64_t);
        visitor.on_attribute("val_size_t", m_val_size_t);
        visitor.on_attribute("vec_string", m_vec_string);
        visitor.on_attribute("vec_float", m_vec_float);
        visitor.on_attribute("vec_double", m_vec_double);
        visitor.on_attribute("vec_uint8_t", m_vec_uint8_t);
        visitor.on_attribute("vec_uint16_t", m_vec_uint16_t);
        visitor.on_attribute("vec_uint32_t", m_vec_uint32_t);
        visitor.on_attribute("vec_uint64_t", m_vec_uint64_t);
        visitor.on_attribute("vec_int8_t", m_vec_int8_t);
        visitor.on_attribute("vec_int16_t", m_vec_int16_t);
        visitor.on_attribute("vec_int32_t", m_vec_int32_t);
        visitor.on_attribute("vec_int64_t", m_vec_int64_t);
        visitor.on_attribute("vec_size_t", m_vec_size_t);
        visitor.on_attribute("position", m_position);
        visitor.on_attribute("node", m_node);
        visitor.on_attribute("node_vector", m_node_vector);
        visitor.on_attribute("parameter_vector", m_parameter_vector);
        visitor.on_attribute("result_vector", m_result_vector);
        return true;
    }

protected:
    TuringModel m_turing_model;
    element::Type m_element_type;
    element::Type_t m_element_type_t;
    string m_val_string;
    bool m_val_bool;
    float m_val_float;
    double m_val_double;
    uint8_t m_val_uint8_t;
    uint16_t m_val_uint16_t;
    uint32_t m_val_uint32_t;
    uint64_t m_val_uint64_t;
    int8_t m_val_int8_t;
    int16_t m_val_int16_t;
    int32_t m_val_int32_t;
    int64_t m_val_int64_t;
    size_t m_val_size_t{23};
    vector<string> m_vec_string;
    vector<float> m_vec_float;
    vector<double> m_vec_double;
    vector<uint8_t> m_vec_uint8_t;
    vector<uint16_t> m_vec_uint16_t;
    vector<uint32_t> m_vec_uint32_t;
    vector<uint64_t> m_vec_uint64_t;
    vector<int8_t> m_vec_int8_t;
    vector<int16_t> m_vec_int16_t;
    vector<int32_t> m_vec_int32_t;
    vector<int64_t> m_vec_int64_t;
    vector<size_t> m_vec_size_t;
    Position m_position;
    shared_ptr<Node> m_node;
    NodeVector m_node_vector;
    ParameterVector m_parameter_vector;
    ResultVector m_result_vector;
};

constexpr NodeTypeInfo Oracle::type_info;

TEST(attributes, user_op)
{
    NodeBuilder::get_ops().register_factory<Oracle>();
    auto program = make_shared<op::Parameter>(element::i32, Shape{200});
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto result = make_shared<op::Result>(data);
    auto oracle = make_shared<Oracle>(program,
                                      data,
                                      TuringModel::XL1200,
                                      element::f32,
                                      element::Type_t::i64,
                                      "12AU7",
                                      true,
                                      1.0f,
                                      1.0,
                                      2,
                                      4,
                                      8,
                                      16,
                                      -1,
                                      -2,
                                      -4,
                                      -8,
                                      34,
                                      vector<string>{"Hello", "World"},
                                      vector<float>{1.0f, 2.0f},
                                      vector<double>{1.0, 2.0},
                                      vector<uint8_t>{1, 2, 4, 8},
                                      vector<uint16_t>{1, 2, 4, 8},
                                      vector<uint32_t>{1, 2, 4, 8},
                                      vector<uint64_t>{1, 2, 4, 8},
                                      vector<int8_t>{1, 2, 4, 8},
                                      vector<int16_t>{1, 2, 4, 8},
                                      vector<int32_t>{1, 2, 4, 8},
                                      vector<int64_t>{1, 2, 4, 8},
                                      vector<size_t>{1, 3, 8, 4, 2},
                                      Position{1.3f, 5.1f, 2.3f},
                                      data,
                                      NodeVector{program, result, data},
                                      ParameterVector{data, data, program},
                                      ResultVector{result});
    NodeBuilder builder;
    AttributeVisitor& saver = builder.get_node_saver();
    AttributeVisitor& loader = builder.get_node_loader();
    loader.register_node(program, "program");
    ASSERT_EQ(loader.get_registered_node("program"), program);
    ASSERT_EQ(loader.get_registered_node_id(program), "program");
    loader.register_node(data, "data");
    loader.register_node(result, "result");
    saver.register_node(program, "program");
    saver.register_node(data, "data");
    saver.register_node(result, "result");
    builder.save_node(oracle);
    auto g_oracle = as_type_ptr<Oracle>(builder.create());

    EXPECT_EQ(g_oracle->get_turing_model(), oracle->get_turing_model());
    EXPECT_EQ(g_oracle->get_element_type(), oracle->get_element_type());
    EXPECT_EQ(g_oracle->get_element_type_t(), oracle->get_element_type_t());
    EXPECT_EQ(g_oracle->get_val_bool(), oracle->get_val_bool());
    EXPECT_EQ(g_oracle->get_val_string(), oracle->get_val_string());
    EXPECT_EQ(g_oracle->get_val_float(), oracle->get_val_float());
    EXPECT_EQ(g_oracle->get_val_double(), oracle->get_val_double());
    EXPECT_EQ(g_oracle->get_val_uint8_t(), oracle->get_val_uint8_t());
    EXPECT_EQ(g_oracle->get_val_uint16_t(), oracle->get_val_uint16_t());
    EXPECT_EQ(g_oracle->get_val_uint32_t(), oracle->get_val_uint32_t());
    EXPECT_EQ(g_oracle->get_val_uint64_t(), oracle->get_val_uint64_t());
    EXPECT_EQ(g_oracle->get_val_int8_t(), oracle->get_val_int8_t());
    EXPECT_EQ(g_oracle->get_val_int16_t(), oracle->get_val_int16_t());
    EXPECT_EQ(g_oracle->get_val_int32_t(), oracle->get_val_int32_t());
    EXPECT_EQ(g_oracle->get_val_int64_t(), oracle->get_val_int64_t());
    EXPECT_EQ(g_oracle->get_val_size_t(), oracle->get_val_size_t());
    EXPECT_EQ(g_oracle->get_vec_uint8_t(), oracle->get_vec_uint8_t());
    EXPECT_EQ(g_oracle->get_vec_uint16_t(), oracle->get_vec_uint16_t());
    EXPECT_EQ(g_oracle->get_vec_uint32_t(), oracle->get_vec_uint32_t());
    EXPECT_EQ(g_oracle->get_vec_uint64_t(), oracle->get_vec_uint64_t());
    EXPECT_EQ(g_oracle->get_vec_int8_t(), oracle->get_vec_int8_t());
    EXPECT_EQ(g_oracle->get_vec_int16_t(), oracle->get_vec_int16_t());
    EXPECT_EQ(g_oracle->get_vec_int32_t(), oracle->get_vec_int32_t());
    EXPECT_EQ(g_oracle->get_vec_int64_t(), oracle->get_vec_int64_t());
    EXPECT_EQ(g_oracle->get_vec_string(), oracle->get_vec_string());
    EXPECT_EQ(g_oracle->get_vec_float(), oracle->get_vec_float());
    EXPECT_EQ(g_oracle->get_vec_double(), oracle->get_vec_double());
    EXPECT_EQ(g_oracle->get_vec_size_t(), oracle->get_vec_size_t());
    EXPECT_EQ(g_oracle->get_position(), oracle->get_position());
    EXPECT_EQ(g_oracle->get_node(), oracle->get_node());
    EXPECT_EQ(g_oracle->get_node_vector(), oracle->get_node_vector());
    EXPECT_EQ(g_oracle->get_parameter_vector(), oracle->get_parameter_vector());
    EXPECT_EQ(g_oracle->get_result_vector(), oracle->get_result_vector());
}

TEST(attributes, matmul_op)
{
    NodeBuilder::get_ops().register_factory<opset1::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{0, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 0});

    bool transpose_a = true;
    bool transpose_b = true;

    auto matmul = make_shared<opset1::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul);
    auto g_matmul = as_type_ptr<opset1::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, partial_shape)
{
    NodeBuilder builder;
    AttributeVisitor& loader = builder.get_node_loader();
    AttributeVisitor& saver = builder.get_node_saver();

    PartialShape dyn = PartialShape::dynamic();
    saver.on_attribute("dyn", dyn);
    PartialShape g_dyn;
    loader.on_attribute("dyn", g_dyn);
    EXPECT_EQ(dyn, g_dyn);

    PartialShape scalar{};
    saver.on_attribute("scalar", scalar);
    PartialShape g_scalar;
    loader.on_attribute("scalar", g_scalar);
    EXPECT_EQ(scalar, g_scalar);

    PartialShape dyn_vector{Dimension::dynamic()};
    saver.on_attribute("dyn_vector", dyn_vector);
    PartialShape g_dyn_vector;
    loader.on_attribute("dyn_vector", g_dyn_vector);
    EXPECT_EQ(dyn_vector, g_dyn_vector);

    PartialShape stat_vector{7};
    saver.on_attribute("stat_vector", stat_vector);
    PartialShape g_stat_vector;
    loader.on_attribute("stat_vector", g_stat_vector);
    EXPECT_EQ(stat_vector, g_stat_vector);

    PartialShape general{7, Dimension::dynamic(), 2, Dimension::dynamic(), 4};
    saver.on_attribute("general", general);
    PartialShape g_general;
    loader.on_attribute("general", g_general);
    EXPECT_EQ(general, g_general);
}

TEST(attributes, max_pool_op)
{
    NodeBuilder::get_ops().register_factory<opset1::MaxPool>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});

    auto strides = Strides{2};
    auto pads_begin = Shape{1};
    auto pads_end = Shape{1};
    auto kernel = Shape{1};
    auto rounding_mode = op::RoundingType::FLOOR;
    auto auto_pad = op::PadType::EXPLICIT;

    auto max_pool = make_shared<opset1::MaxPool>(
        data, strides, pads_begin, pads_end, kernel, rounding_mode, auto_pad);
    NodeBuilder builder(max_pool);
    auto g_max_pool = as_type_ptr<opset1::MaxPool>(builder.create());

    EXPECT_EQ(g_max_pool->get_strides(), max_pool->get_strides());
    EXPECT_EQ(g_max_pool->get_pads_begin(), max_pool->get_pads_begin());
    EXPECT_EQ(g_max_pool->get_pads_end(), max_pool->get_pads_end());
    EXPECT_EQ(g_max_pool->get_kernel(), max_pool->get_kernel());
    EXPECT_EQ(g_max_pool->get_rounding_type(), max_pool->get_rounding_type());
    EXPECT_EQ(g_max_pool->get_auto_pad(), max_pool->get_auto_pad());
}

TEST(attributes, mod_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Mod>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 1});

    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    auto mod = make_shared<opset1::Mod>(A, B, auto_broadcast);
    NodeBuilder builder(mod);
    auto g_mod = as_type_ptr<opset1::Mod>(builder.create());

    EXPECT_EQ(g_mod->get_auto_broadcast(), mod->get_auto_broadcast());
}

TEST(attributes, non_max_suppression_op_custom_attributes)
{
    NodeBuilder::get_ops().register_factory<opset1::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = opset1::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;

    auto nms =
        make_shared<opset1::NonMaxSuppression>(boxes, scores, box_encoding, sort_result_descending);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_op_default_attributes)
{
    NodeBuilder::get_ops().register_factory<opset1::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<opset1::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_v3_op_custom_attributes)
{
    NodeBuilder::get_ops().register_factory<opset3::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = opset3::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;
    element::Type output_type = element::i32;

    auto nms = make_shared<opset3::NonMaxSuppression>(
        boxes, scores, box_encoding, sort_result_descending, output_type);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset3::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}

TEST(attributes, non_max_suppression_v3_op_default_attributes)
{
    NodeBuilder::get_ops().register_factory<opset3::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<opset3::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset3::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
    EXPECT_EQ(g_nms->get_output_type(), nms->get_output_type());
}

TEST(attributes, normalize_l2_op)
{
    NodeBuilder::get_ops().register_factory<opset1::NormalizeL2>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{1});
    const auto axes = make_shared<op::Constant>(element::i32, Shape{}, vector<int32_t>{0});

    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize_l2 = make_shared<opset1::NormalizeL2>(data, axes, eps, eps_mode);
    NodeBuilder builder(normalize_l2);
    auto g_normalize_l2 = as_type_ptr<opset1::NormalizeL2>(builder.create());

    EXPECT_EQ(g_normalize_l2->get_eps(), normalize_l2->get_eps());
    EXPECT_EQ(g_normalize_l2->get_eps_mode(), normalize_l2->get_eps_mode());
}

TEST(attributes, one_hot_op)
{
    NodeBuilder::get_ops().register_factory<opset1::OneHot>();
    auto indices = make_shared<op::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = op::Constant::create(element::i64, Shape{}, {4});
    auto on_value = op::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = op::Constant::create(element::f32, Shape{}, {0.0f});

    int64_t axis = 3;

    auto one_hot = make_shared<opset1::OneHot>(indices, depth, on_value, off_value, axis);
    NodeBuilder builder(one_hot);
    auto g_one_hot = as_type_ptr<opset1::OneHot>(builder.create());

    EXPECT_EQ(g_one_hot->get_axis(), one_hot->get_axis());
}

TEST(attributes, pad_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Pad>();
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    auto pad_mode = op::PadMode::EDGE;

    auto pad = make_shared<opset1::Pad>(arg, pads_begin, pads_end, pad_mode);
    NodeBuilder builder(pad);
    auto g_pad = as_type_ptr<opset1::Pad>(builder.create());

    EXPECT_EQ(g_pad->get_pad_mode(), pad->get_pad_mode());
}

TEST(attributes, psroi_pooling_op)
{
    NodeBuilder::get_ops().register_factory<opset1::PSROIPooling>();
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1024, 63, 38});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{300, 5});

    const int64_t output_dim = 64;
    const int64_t group_size = 4;
    const float spatial_scale = 0.0625;
    int spatial_bins_x = 1;
    int spatial_bins_y = 1;
    string mode = "average";

    auto psroi_pool = make_shared<opset1::PSROIPooling>(
        input, coords, output_dim, group_size, spatial_scale, spatial_bins_x, spatial_bins_y, mode);
    NodeBuilder builder(psroi_pool);
    auto g_psroi_pool = as_type_ptr<opset1::PSROIPooling>(builder.create());

    EXPECT_EQ(g_psroi_pool->get_output_dim(), psroi_pool->get_output_dim());
    EXPECT_EQ(g_psroi_pool->get_group_size(), psroi_pool->get_group_size());
    EXPECT_EQ(g_psroi_pool->get_spatial_scale(), psroi_pool->get_spatial_scale());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_x(), psroi_pool->get_spatial_bins_x());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_y(), psroi_pool->get_spatial_bins_y());
    EXPECT_EQ(g_psroi_pool->get_mode(), psroi_pool->get_mode());
}

TEST(attributes, reduce_logical_and_op)
{
    // ReduceLogicalAnd derives visit_attributes from op::util::LogicalReductionKeepDims
    NodeBuilder::get_ops().register_factory<opset1::ReduceLogicalAnd>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_logical_and = make_shared<opset1::ReduceSum>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_logical_and);
    auto g_reduce_logical_and = as_type_ptr<opset1::ReduceSum>(builder.create());

    EXPECT_EQ(g_reduce_logical_and->get_keep_dims(), reduce_logical_and->get_keep_dims());
}

TEST(attributes, reduce_logical_or_op)
{
    // ReduceLogicalOr derives visit_attributes from op::util::LogicalReductionKeepDims
    NodeBuilder::get_ops().register_factory<opset1::ReduceLogicalOr>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_logical_or = make_shared<opset1::ReduceLogicalOr>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_logical_or);
    auto g_reduce_logical_or = as_type_ptr<opset1::ReduceLogicalOr>(builder.create());

    EXPECT_EQ(g_reduce_logical_or->get_keep_dims(), reduce_logical_or->get_keep_dims());
}

TEST(attributes, reduce_max_op)
{
    // ReduceMax derives visit_attributes from op::util::ArithmeticReductionKeepDims
    NodeBuilder::get_ops().register_factory<opset1::ReduceMax>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_max = make_shared<opset1::ReduceMax>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_max);
    auto g_reduce_max = as_type_ptr<opset1::ReduceMax>(builder.create());

    EXPECT_EQ(g_reduce_max->get_keep_dims(), reduce_max->get_keep_dims());
}

TEST(attributes, reduce_mean_op)
{
    // ReduceMean derives visit_attributes from op::util::ArithmeticReductionKeepDims
    NodeBuilder::get_ops().register_factory<opset1::ReduceMean>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_mean = make_shared<opset1::ReduceMean>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_mean);
    auto g_reduce_mean = as_type_ptr<opset1::ReduceMean>(builder.create());

    EXPECT_EQ(g_reduce_mean->get_keep_dims(), reduce_mean->get_keep_dims());
}

TEST(attributes, reduce_min_op)
{
    // ReduceMin derives visit_attributes from op::util::ArithmeticReductionKeepDims
    NodeBuilder::get_ops().register_factory<opset1::ReduceMin>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_min = make_shared<opset1::ReduceMin>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_min);
    auto g_reduce_min = as_type_ptr<opset1::ReduceMin>(builder.create());

    EXPECT_EQ(g_reduce_min->get_keep_dims(), reduce_min->get_keep_dims());
}

TEST(attributes, reduce_prod_op)
{
    // ReduceProd derives visit_attributes from op::util::ArithmeticReductionKeepDims
    NodeBuilder::get_ops().register_factory<opset1::ReduceProd>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_prod = make_shared<opset1::ReduceProd>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_prod);
    auto g_reduce_prod = as_type_ptr<opset1::ReduceProd>(builder.create());

    EXPECT_EQ(g_reduce_prod->get_keep_dims(), reduce_prod->get_keep_dims());
}

TEST(attributes, reduce_sum_op)
{
    // ReduceSum derives visit_attributes from op::util::ArithmeticReductionKeepDims
    NodeBuilder::get_ops().register_factory<opset1::ReduceSum>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_sum = make_shared<opset1::ReduceSum>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_sum);
    auto g_reduce_sum = as_type_ptr<opset1::ReduceSum>(builder.create());

    EXPECT_EQ(g_reduce_sum->get_keep_dims(), reduce_sum->get_keep_dims());
}

TEST(attributes, region_yolo_op)
{
    NodeBuilder::get_ops().register_factory<opset1::RegionYolo>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 255, 26, 26});

    size_t num_coords = 4;
    size_t num_classes = 1;
    size_t num_regions = 6;
    auto do_softmax = false;
    auto mask = std::vector<int64_t>{0, 1};
    auto axis = 1;
    auto end_axis = 3;
    auto anchors = std::vector<float>{10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};

    auto region_yolo = make_shared<opset1::RegionYolo>(
        data, num_coords, num_classes, num_regions, do_softmax, mask, axis, end_axis, anchors);
    NodeBuilder builder(region_yolo);
    auto g_region_yolo = as_type_ptr<opset1::RegionYolo>(builder.create());

    EXPECT_EQ(g_region_yolo->get_num_coords(), region_yolo->get_num_coords());
    EXPECT_EQ(g_region_yolo->get_num_classes(), region_yolo->get_num_classes());
    EXPECT_EQ(g_region_yolo->get_num_regions(), region_yolo->get_num_regions());
    EXPECT_EQ(g_region_yolo->get_do_softmax(), region_yolo->get_do_softmax());
    EXPECT_EQ(g_region_yolo->get_mask(), region_yolo->get_mask());
    EXPECT_EQ(g_region_yolo->get_anchors(), region_yolo->get_anchors());
    EXPECT_EQ(g_region_yolo->get_axis(), region_yolo->get_axis());
    EXPECT_EQ(g_region_yolo->get_end_axis(), region_yolo->get_end_axis());
}

TEST(attributes, reshape_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Reshape>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4});
    auto pattern = make_shared<op::Parameter>(element::i32, Shape{2});

    bool special_zero = true;

    auto reshape = make_shared<opset1::Reshape>(data, pattern, special_zero);
    NodeBuilder builder(reshape);
    auto g_reshape = as_type_ptr<opset1::Reshape>(builder.create());

    EXPECT_EQ(g_reshape->get_special_zero(), reshape->get_special_zero());
}

TEST(attributes, reverse_op_enum_mode)
{
    NodeBuilder::get_ops().register_factory<opset1::Reverse>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto reversed_axes = make_shared<op::Parameter>(element::i32, Shape{200});

    auto reverse = make_shared<opset1::Reverse>(data, reversed_axes, opset1::Reverse::Mode::INDEX);
    NodeBuilder builder(reverse);
    auto g_reverse = as_type_ptr<opset1::Reverse>(builder.create());

    EXPECT_EQ(g_reverse->get_mode(), reverse->get_mode());
}

TEST(attributes, reverse_op_string_mode)
{
    NodeBuilder::get_ops().register_factory<opset1::Reverse>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto reversed_axes = make_shared<op::Parameter>(element::i32, Shape{200});

    std::string mode = "index";

    auto reverse = make_shared<opset1::Reverse>(data, reversed_axes, mode);
    NodeBuilder builder(reverse);
    auto g_reverse = as_type_ptr<opset1::Reverse>(builder.create());

    EXPECT_EQ(g_reverse->get_mode(), reverse->get_mode());
}

TEST(attributes, reverse_sequence_op)
{
    NodeBuilder::get_ops().register_factory<opset1::ReverseSequence>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 2});
    auto seq_indices = make_shared<op::Parameter>(element::i32, Shape{4});

    auto batch_axis = 2;
    auto seq_axis = 1;

    auto reverse_sequence =
        make_shared<opset1::ReverseSequence>(data, seq_indices, batch_axis, seq_axis);
    NodeBuilder builder(reverse_sequence);
    auto g_reverse_sequence = as_type_ptr<opset1::ReverseSequence>(builder.create());

    EXPECT_EQ(g_reverse_sequence->get_origin_batch_axis(),
              reverse_sequence->get_origin_batch_axis());
    EXPECT_EQ(g_reverse_sequence->get_origin_sequence_axis(),
              reverse_sequence->get_origin_sequence_axis());
}

TEST(attributes, rnn_cell_op_custom_attributes)
{
    NodeBuilder::get_ops().register_factory<opset1::RNNCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;
    auto activations = std::vector<std::string>{"sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    float clip = 1.0;

    auto rnn_cell = make_shared<opset1::RNNCell>(
        X, H, W, R, hidden_size, activations, activations_alpha, activations_beta, clip);

    NodeBuilder builder(rnn_cell);
    auto g_rnn_cell = as_type_ptr<opset1::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}

TEST(attributes, rnn_cell_op_default_attributes)
{
    NodeBuilder::get_ops().register_factory<opset1::RNNCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;

    auto rnn_cell = make_shared<opset1::RNNCell>(X, H, W, R, hidden_size);

    NodeBuilder builder(rnn_cell);
    auto g_rnn_cell = as_type_ptr<opset1::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}

TEST(attributes, elu_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Elu>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    double alpha = 0.1;

    const auto elu = make_shared<opset1::Elu>(data, alpha);
    NodeBuilder builder(elu);
    auto g_elu = as_type_ptr<opset1::Elu>(builder.create());

    EXPECT_EQ(g_elu->get_alpha(), elu->get_alpha());
}

TEST(attributes, fake_quantize_op)
{
    NodeBuilder::get_ops().register_factory<opset1::FakeQuantize>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    auto levels = 5;
    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    const auto fake_quantize = make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels, auto_broadcast);
    NodeBuilder builder(fake_quantize);
    auto g_fake_quantize = as_type_ptr<opset1::FakeQuantize>(builder.create());

    EXPECT_EQ(g_fake_quantize->get_levels(), fake_quantize->get_levels());
    EXPECT_EQ(g_fake_quantize->get_auto_broadcast(), fake_quantize->get_auto_broadcast());
}

TEST(attributes, broadcast_v3)
{
    NodeBuilder::get_ops().register_factory<opset3::Broadcast>();
    const auto arg = make_shared<op::Parameter>(element::i64, Shape{1, 3, 1});
    const auto shape = make_shared<op::Parameter>(element::i64, Shape{3});
    const auto broadcast_spec = op::BroadcastType::BIDIRECTIONAL;

    const auto broadcast_v3 = make_shared<op::v3::Broadcast>(arg, shape, broadcast_spec);
    NodeBuilder builder(broadcast_v3);
    auto g_broadcast_v3 = as_type_ptr<opset3::Broadcast>(builder.create());

    EXPECT_EQ(g_broadcast_v3->get_broadcast_spec(), broadcast_spec);
}

TEST(attributes, grn_op)
{
    NodeBuilder::get_ops().register_factory<opset1::GRN>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});

    float bias = 1.25f;

    auto grn = make_shared<opset1::GRN>(data, bias);
    NodeBuilder builder(grn);
    auto g_grn = as_type_ptr<opset1::GRN>(builder.create());

    EXPECT_EQ(g_grn->get_bias(), grn->get_bias());
}

TEST(attributes, group_conv_op)
{
    NodeBuilder::get_ops().register_factory<opset1::GroupConvolution>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 12, 224, 224});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{4, 1, 3, 5, 5});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{1, 2};
    auto pads_end = CoordinateDiff{1, 2};
    auto dilations = Strides{1, 1};
    auto group_conv = make_shared<opset1::GroupConvolution>(
        data, filters, strides, pads_begin, pads_end, dilations, op::PadType::VALID);
    NodeBuilder builder(group_conv);
    auto g_group_conv = as_type_ptr<opset1::GroupConvolution>(builder.create());
    EXPECT_EQ(g_group_conv->get_strides(), group_conv->get_strides());
    EXPECT_EQ(g_group_conv->get_pads_begin(), group_conv->get_pads_begin());
    EXPECT_EQ(g_group_conv->get_pads_end(), group_conv->get_pads_end());
    EXPECT_EQ(g_group_conv->get_dilations(), group_conv->get_dilations());
    EXPECT_EQ(g_group_conv->get_auto_pad(), group_conv->get_auto_pad());
}

TEST(attributes, group_conv_backprop_data_op)
{
    NodeBuilder::get_ops().register_factory<opset1::GroupConvolutionBackpropData>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 20, 224, 224});
    const auto filter = make_shared<op::Parameter>(element::f32, Shape{4, 5, 2, 3, 3});
    const auto output_shape = make_shared<op::Parameter>(element::i32, Shape{1});

    const auto strides = Strides{2, 1};
    const auto pads_begin = CoordinateDiff{3, 4};
    const auto pads_end = CoordinateDiff{4, 6};
    const auto dilations = Strides{3, 1};
    const auto auto_pad = op::PadType::EXPLICIT;
    const auto output_padding = CoordinateDiff{3, 4};

    const auto gcbd = make_shared<opset1::GroupConvolutionBackpropData>(data,
                                                                        filter,
                                                                        output_shape,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        dilations,
                                                                        auto_pad,
                                                                        output_padding);
    NodeBuilder builder(gcbd);
    const auto g_gcbd = as_type_ptr<opset1::GroupConvolutionBackpropData>(builder.create());

    EXPECT_EQ(g_gcbd->get_strides(), gcbd->get_strides());
    EXPECT_EQ(g_gcbd->get_pads_begin(), gcbd->get_pads_begin());
    EXPECT_EQ(g_gcbd->get_pads_end(), gcbd->get_pads_end());
    EXPECT_EQ(g_gcbd->get_dilations(), gcbd->get_dilations());
    EXPECT_EQ(g_gcbd->get_auto_pad(), gcbd->get_auto_pad());
    EXPECT_EQ(g_gcbd->get_output_padding(), gcbd->get_output_padding());
}

TEST(attributes, lrn_op)
{
    NodeBuilder::get_ops().register_factory<opset1::LRN>();
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto axes = make_shared<op::Parameter>(element::i32, Shape{2});

    const double alpha = 1.1;
    const double beta = 2.2;
    const double bias = 3.3;
    const size_t size = 4;

    const auto lrn = make_shared<opset1::LRN>(arg, axes, alpha, beta, bias, size);
    NodeBuilder builder(lrn);
    auto g_lrn = as_type_ptr<opset1::LRN>(builder.create());

    EXPECT_EQ(g_lrn->get_alpha(), lrn->get_alpha());
    EXPECT_EQ(g_lrn->get_beta(), lrn->get_beta());
    EXPECT_EQ(g_lrn->get_bias(), lrn->get_bias());
    EXPECT_EQ(g_lrn->get_nsize(), lrn->get_nsize());
}

TEST(attributes, lstm_cell_op)
{
    NodeBuilder::get_ops().register_factory<opset4::LSTMCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{12, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{12, 3});
    const auto initial_hidden_state = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    const auto initial_cell_state = make_shared<op::Parameter>(element::f32, Shape{2, 3});

    const auto hidden_size = 3;
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    const float clip = 0.5f;
    const auto lstm_cell = make_shared<opset4::LSTMCell>(X,
                                                         initial_hidden_state,
                                                         initial_cell_state,
                                                         W,
                                                         R,
                                                         hidden_size,
                                                         activations,
                                                         activations_alpha,
                                                         activations_beta,
                                                         clip);
    NodeBuilder builder(lstm_cell);
    auto g_lstm_cell = as_type_ptr<opset4::LSTMCell>(builder.create());

    EXPECT_EQ(g_lstm_cell->get_hidden_size(), lstm_cell->get_hidden_size());
    EXPECT_EQ(g_lstm_cell->get_activations(), lstm_cell->get_activations());
    EXPECT_EQ(g_lstm_cell->get_activations_alpha(), lstm_cell->get_activations_alpha());
    EXPECT_EQ(g_lstm_cell->get_activations_beta(), lstm_cell->get_activations_beta());
    EXPECT_EQ(g_lstm_cell->get_clip(), lstm_cell->get_clip());
}

TEST(attributes, lstm_sequence_op)
{
    NodeBuilder::get_ops().register_factory<opset5::LSTMSequence>();

    const size_t batch_size = 4;
    const size_t num_directions = 2;
    const size_t seq_length = 8;
    const size_t input_size = 16;
    const size_t hidden_size = 64;

    const auto X =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, seq_length, input_size});
    const auto initial_hidden_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto initial_cell_state =
        make_shared<op::Parameter>(element::f32, Shape{batch_size, num_directions, hidden_size});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{batch_size});
    const auto W = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32,
                                              Shape{num_directions, 4 * hidden_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{num_directions, 4 * hidden_size});

    const auto lstm_direction = op::RecurrentSequenceDirection::BIDIRECTIONAL;
    const std::vector<float> activations_alpha = {1, 2, 3};
    const std::vector<float> activations_beta = {4, 5, 6};
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    const float clip_threshold = 0.5f;

    const auto lstm_sequence = make_shared<opset5::LSTMSequence>(X,
                                                                 initial_hidden_state,
                                                                 initial_cell_state,
                                                                 sequence_lengths,
                                                                 W,
                                                                 R,
                                                                 B,
                                                                 hidden_size,
                                                                 lstm_direction,
                                                                 activations_alpha,
                                                                 activations_beta,
                                                                 activations,
                                                                 clip_threshold);
    NodeBuilder builder(lstm_sequence);
    auto g_lstm_sequence = as_type_ptr<opset5::LSTMSequence>(builder.create());

    EXPECT_EQ(g_lstm_sequence->get_hidden_size(), lstm_sequence->get_hidden_size());
    EXPECT_EQ(g_lstm_sequence->get_activations(), lstm_sequence->get_activations());
    EXPECT_EQ(g_lstm_sequence->get_activations_alpha(), lstm_sequence->get_activations_alpha());
    EXPECT_EQ(g_lstm_sequence->get_activations_beta(), lstm_sequence->get_activations_beta());
    EXPECT_EQ(g_lstm_sequence->get_clip(), lstm_sequence->get_clip());
    EXPECT_EQ(g_lstm_sequence->get_direction(), lstm_sequence->get_direction());
}

TEST(attributes, shuffle_channels_op)
{
    NodeBuilder::get_ops().register_factory<opset1::ShuffleChannels>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = 0;
    auto groups = 2;
    auto shuffle_channels = make_shared<opset1::ShuffleChannels>(data, axis, groups);
    NodeBuilder builder(shuffle_channels);
    auto g_shuffle_channels = as_type_ptr<opset1::ShuffleChannels>(builder.create());

    EXPECT_EQ(g_shuffle_channels->get_axis(), shuffle_channels->get_axis());
    EXPECT_EQ(g_shuffle_channels->get_group(), shuffle_channels->get_group());
}

TEST(attributes, softmax_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Softmax>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = 0;
    auto softmax = make_shared<opset1::Softmax>(data, axis);
    NodeBuilder builder(softmax);
    auto g_softmax = as_type_ptr<opset1::Softmax>(builder.create());

    EXPECT_EQ(g_softmax->get_axis(), softmax->get_axis());
}

TEST(attributes, space_to_depth_op)
{
    NodeBuilder::get_ops().register_factory<opset1::SpaceToDepth>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 50, 50});
    auto block_size = 2;
    auto mode = opset1::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<opset1::SpaceToDepth>(data, mode, block_size);
    NodeBuilder builder(space_to_depth);
    auto g_space_to_depth = as_type_ptr<opset1::SpaceToDepth>(builder.create());

    EXPECT_EQ(g_space_to_depth->get_block_size(), space_to_depth->get_block_size());
    EXPECT_EQ(g_space_to_depth->get_mode(), space_to_depth->get_mode());
}

TEST(attributes, split_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Split>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = make_shared<op::Parameter>(element::i32, Shape{});
    auto num_splits = 2;
    auto split = make_shared<opset1::Split>(data, axis, num_splits);
    NodeBuilder builder(split);
    auto g_split = as_type_ptr<opset1::Split>(builder.create());

    EXPECT_EQ(g_split->get_num_splits(), split->get_num_splits());
}

TEST(attributes, squared_difference_op)
{
    NodeBuilder::get_ops().register_factory<opset1::SquaredDifference>();
    auto x1 = make_shared<op::Parameter>(element::i32, Shape{200});
    auto x2 = make_shared<op::Parameter>(element::i32, Shape{200});
    auto auto_broadcast = op::AutoBroadcastType::NUMPY;
    auto squared_difference = make_shared<opset1::SquaredDifference>(x1, x2, auto_broadcast);
    NodeBuilder builder(squared_difference);
    auto g_squared_difference = as_type_ptr<opset1::SquaredDifference>(builder.create());

    EXPECT_EQ(g_squared_difference->get_autob(), squared_difference->get_autob());
}

TEST(attributes, strided_slice_op)
{
    NodeBuilder::get_ops().register_factory<opset1::StridedSlice>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto begin = make_shared<op::Parameter>(element::i32, Shape{2});
    auto end = make_shared<op::Parameter>(element::i32, Shape{2});
    auto stride = make_shared<op::Parameter>(element::i32, Shape{2});

    auto begin_mask = std::vector<int64_t>{0, 0};
    auto end_mask = std::vector<int64_t>{0, 0};
    auto new_axis_mask = std::vector<int64_t>{0, 0};
    auto shrink_axis_mask = std::vector<int64_t>{0, 0};
    auto ellipsis_mask = std::vector<int64_t>{0, 0};

    auto strided_slice = make_shared<opset1::StridedSlice>(data,
                                                           begin,
                                                           end,
                                                           stride,
                                                           begin_mask,
                                                           end_mask,
                                                           new_axis_mask,
                                                           shrink_axis_mask,
                                                           ellipsis_mask);
    NodeBuilder builder(strided_slice);
    auto g_strided_slice = as_type_ptr<opset1::StridedSlice>(builder.create());

    EXPECT_EQ(g_strided_slice->get_begin_mask(), strided_slice->get_begin_mask());
    EXPECT_EQ(g_strided_slice->get_end_mask(), strided_slice->get_end_mask());
    EXPECT_EQ(g_strided_slice->get_new_axis_mask(), strided_slice->get_new_axis_mask());
    EXPECT_EQ(g_strided_slice->get_shrink_axis_mask(), strided_slice->get_shrink_axis_mask());
    EXPECT_EQ(g_strided_slice->get_ellipsis_mask(), strided_slice->get_ellipsis_mask());
}

TEST(attributes, topk_op)
{
    NodeBuilder::get_ops().register_factory<opset1::TopK>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto k = make_shared<op::Parameter>(element::i32, Shape{});

    auto axis = 0;
    auto mode = opset1::TopK::Mode::MAX;
    auto sort_type = opset1::TopK::SortType::SORT_VALUES;

    auto topk = make_shared<opset1::TopK>(data, k, axis, mode, sort_type);
    NodeBuilder builder(topk);
    auto g_topk = as_type_ptr<opset1::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
}

TEST(attributes, logical_xor_op)
{
    NodeBuilder::get_ops().register_factory<opset1::LogicalXor>();
    auto x1 = make_shared<op::Parameter>(element::boolean, Shape{200});
    auto x2 = make_shared<op::Parameter>(element::boolean, Shape{200});

    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    auto logical_xor = make_shared<opset1::LogicalXor>(x1, x2, auto_broadcast);
    NodeBuilder builder(logical_xor);
    auto g_logical_xor = as_type_ptr<opset1::LogicalXor>(builder.create());

    EXPECT_EQ(g_logical_xor->get_autob(), logical_xor->get_autob());
}

TEST(attributes, extractimagepatches_op)
{
    NodeBuilder::get_ops().register_factory<opset3::ExtractImagePatches>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{64, 3, 10, 10});

    auto sizes = Shape{3, 3};
    auto strides = Strides{5, 5};
    auto rates = Shape{1, 1};
    auto padtype_padding = ngraph::op::PadType::VALID;

    auto extractimagepatches =
        make_shared<opset3::ExtractImagePatches>(data, sizes, strides, rates, padtype_padding);
    NodeBuilder builder(extractimagepatches);
    auto g_extractimagepatches = as_type_ptr<opset3::ExtractImagePatches>(builder.create());

    EXPECT_EQ(g_extractimagepatches->get_sizes(), sizes);
    EXPECT_EQ(g_extractimagepatches->get_strides(), strides);
    EXPECT_EQ(g_extractimagepatches->get_rates(), rates);
    EXPECT_EQ(g_extractimagepatches->get_auto_pad(), padtype_padding);
}

TEST(attributes, mvn_op)
{
    NodeBuilder::get_ops().register_factory<opset3::MVN>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});

    const auto axes = AxisSet{0, 1};

    const auto op = make_shared<opset3::MVN>(data, true, false, 0.1);
    op->set_reduction_axes(axes);
    NodeBuilder builder(op);
    const auto g_op = as_type_ptr<opset3::MVN>(builder.create());

    EXPECT_EQ(g_op->get_reduction_axes(), op->get_reduction_axes());
    EXPECT_EQ(g_op->get_across_channels(), op->get_across_channels());
    EXPECT_EQ(g_op->get_normalize_variance(), op->get_normalize_variance());
    EXPECT_EQ(g_op->get_eps(), op->get_eps());
}

TEST(attributes, reorg_yolo_op_stride)
{
    NodeBuilder::get_ops().register_factory<opset3::ReorgYolo>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{1, 64, 26, 26});

    const auto op = make_shared<op::v0::ReorgYolo>(data, 2);
    NodeBuilder builder(op);
    const auto g_op = as_type_ptr<op::v0::ReorgYolo>(builder.create());

    EXPECT_EQ(g_op->get_strides(), op->get_strides());
}

TEST(attributes, reorg_yolo_op_strides)
{
    NodeBuilder::get_ops().register_factory<opset3::ReorgYolo>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{1, 64, 26, 26});

    const auto op = make_shared<op::v0::ReorgYolo>(data, Strides{2});
    NodeBuilder builder(op);
    const auto g_op = as_type_ptr<op::v0::ReorgYolo>(builder.create());

    EXPECT_EQ(g_op->get_strides(), op->get_strides());
}

TEST(attributes, roi_pooling_op)
{
    NodeBuilder::get_ops().register_factory<opset3::ROIPooling>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    const auto coords = make_shared<op::Parameter>(element::f32, Shape{2, 5});

    const auto op = make_shared<opset3::ROIPooling>(data, coords, Shape{5, 5}, 0.123, "bilinear");
    NodeBuilder builder(op);
    const auto g_op = as_type_ptr<opset3::ROIPooling>(builder.create());

    EXPECT_EQ(g_op->get_output_size(), op->get_output_size());
    EXPECT_EQ(g_op->get_spatial_scale(), op->get_spatial_scale());
    EXPECT_EQ(g_op->get_method(), op->get_method());
}

TEST(attributes, constant_op)
{
    vector<float> data{5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f};
    auto k = make_shared<op::v0::Constant>(element::f32, Shape{2, 3}, data);
    NodeBuilder builder(k);
    auto g_k = as_type_ptr<op::v0::Constant>(builder.create());
    g_k->validate_and_infer_types();
    ASSERT_TRUE(g_k);
    EXPECT_EQ(k->get_element_type(), g_k->get_element_type());
    EXPECT_EQ(k->get_shape(), g_k->get_shape());
    vector<float> g_data = g_k->get_vector<float>();
    EXPECT_EQ(data, g_data);
}

TEST(attributes, bucketize_v3_op_default_attributes)
{
    NodeBuilder::get_ops().register_factory<opset3::Bucketize>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5});
    auto bucketize = make_shared<opset3::Bucketize>(data, buckets);
    NodeBuilder builder(bucketize);

    auto g_bucketize = as_type_ptr<opset3::Bucketize>(builder.create());

    EXPECT_EQ(g_bucketize->get_output_type(), bucketize->get_output_type());
    EXPECT_EQ(g_bucketize->get_with_right_bound(), bucketize->get_with_right_bound());
}

TEST(attributes, bucketize_v3_op_custom_attributes)
{
    NodeBuilder::get_ops().register_factory<opset3::Bucketize>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto buckets = make_shared<op::Parameter>(element::f32, Shape{5});
    element::Type output_type = element::i32;
    bool with_right_bound = false;

    auto bucketize = make_shared<opset3::Bucketize>(data, buckets, output_type, with_right_bound);
    NodeBuilder builder(bucketize);

    auto g_bucketize = as_type_ptr<opset3::Bucketize>(builder.create());

    EXPECT_EQ(g_bucketize->get_output_type(), bucketize->get_output_type());
    EXPECT_EQ(g_bucketize->get_with_right_bound(), bucketize->get_with_right_bound());
}

TEST(attributes, cum_sum_op_default_attributes)
{
    NodeBuilder::get_ops().register_factory<opset3::CumSum>();

    Shape shape{1, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axis = make_shared<op::Parameter>(element::i32, Shape{1});
    auto cs = make_shared<op::CumSum>(A, axis);

    NodeBuilder builder(cs);
    auto g_cs = as_type_ptr<opset3::CumSum>(builder.create());

    EXPECT_EQ(g_cs->is_exclusive(), cs->is_exclusive());
    EXPECT_EQ(g_cs->is_reverse(), cs->is_reverse());
}

TEST(attributes, cum_sum_op_custom_attributes)
{
    NodeBuilder::get_ops().register_factory<opset3::CumSum>();

    Shape shape{1, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axis = make_shared<op::Parameter>(element::i32, Shape{1});
    bool exclusive = true;
    bool reverse = true;
    auto cs = make_shared<op::CumSum>(A, axis, exclusive, reverse);

    NodeBuilder builder(cs);
    auto g_cs = as_type_ptr<opset3::CumSum>(builder.create());

    EXPECT_EQ(g_cs->is_exclusive(), cs->is_exclusive());
    EXPECT_EQ(g_cs->is_reverse(), cs->is_reverse());
}

TEST(attributes, interpolate_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Interpolate>();
    auto img = make_shared<op::Parameter>(element::f32, Shape{1, 3, 32, 32});
    auto out_shape = make_shared<op::Parameter>(element::i32, Shape{2});

    op::v0::InterpolateAttrs interp_atrs;
    interp_atrs.axes = AxisSet{1, 2};
    interp_atrs.mode = "cubic";
    interp_atrs.align_corners = true;
    interp_atrs.antialias = true;
    interp_atrs.pads_begin = vector<size_t>{0, 0};
    interp_atrs.pads_end = vector<size_t>{0, 0};

    auto interpolate = make_shared<opset1::Interpolate>(img, out_shape, interp_atrs);
    NodeBuilder builder(interpolate);
    auto g_interpolate = as_type_ptr<opset1::Interpolate>(builder.create());

    const auto i_attrs = interpolate->get_attrs();
    const auto g_i_attrs = g_interpolate->get_attrs();

    EXPECT_EQ(g_i_attrs.axes, i_attrs.axes);
    EXPECT_EQ(g_i_attrs.mode, i_attrs.mode);
    EXPECT_EQ(g_i_attrs.align_corners, i_attrs.align_corners);
    EXPECT_EQ(g_i_attrs.antialias, i_attrs.antialias);
    EXPECT_EQ(g_i_attrs.pads_begin, i_attrs.pads_begin);
    EXPECT_EQ(g_i_attrs.pads_end, i_attrs.pads_end);
}

TEST(attributes, detection_output_op)
{
    NodeBuilder::get_ops().register_factory<opset1::DetectionOutput>();
    const auto box_logits = make_shared<op::Parameter>(element::f32, Shape{1, 2 * 1 * 4});
    const auto class_preds = make_shared<op::Parameter>(element::f32, Shape{1, 2 * 32});
    const auto proposals = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2 * 4});
    const auto aux_class_preds = make_shared<op::Parameter>(element::f32, Shape{1, 2 * 2});
    const auto aux_box_pred = make_shared<op::Parameter>(element::f32, Shape{1, 2 * 1 * 4});

    op::DetectionOutputAttrs attrs;
    attrs.num_classes = 32;
    attrs.background_label_id = 0;
    attrs.top_k = 1;
    attrs.variance_encoded_in_target = false;
    attrs.keep_top_k = {1};
    attrs.code_type = string{"caffe.PriorBoxParameter.CORNER"};
    attrs.share_location = true;
    attrs.nms_threshold = 0.64f;
    attrs.confidence_threshold = 1e-4f;
    attrs.clip_after_nms = true;
    attrs.clip_before_nms = false;
    attrs.decrease_label_id = false;
    attrs.normalized = true;
    attrs.input_height = 32;
    attrs.input_width = 32;
    attrs.objectness_score = 0.73f;

    auto detection_output = make_shared<opset1::DetectionOutput>(
        box_logits, class_preds, proposals, aux_class_preds, aux_box_pred, attrs);
    NodeBuilder builder(detection_output);
    auto g_detection_output = as_type_ptr<opset1::DetectionOutput>(builder.create());

    const auto do_attrs = detection_output->get_attrs();
    const auto g_do_attrs = g_detection_output->get_attrs();

    EXPECT_EQ(g_do_attrs.num_classes, do_attrs.num_classes);
    EXPECT_EQ(g_do_attrs.background_label_id, do_attrs.background_label_id);
    EXPECT_EQ(g_do_attrs.top_k, do_attrs.top_k);
    EXPECT_EQ(g_do_attrs.variance_encoded_in_target, do_attrs.variance_encoded_in_target);
    EXPECT_EQ(g_do_attrs.keep_top_k, do_attrs.keep_top_k);
    EXPECT_EQ(g_do_attrs.code_type, do_attrs.code_type);
    EXPECT_EQ(g_do_attrs.share_location, do_attrs.share_location);
    EXPECT_EQ(g_do_attrs.nms_threshold, do_attrs.nms_threshold);
    EXPECT_EQ(g_do_attrs.confidence_threshold, do_attrs.confidence_threshold);
    EXPECT_EQ(g_do_attrs.clip_after_nms, do_attrs.clip_after_nms);
    EXPECT_EQ(g_do_attrs.clip_before_nms, do_attrs.clip_before_nms);
    EXPECT_EQ(g_do_attrs.decrease_label_id, do_attrs.decrease_label_id);
    EXPECT_EQ(g_do_attrs.normalized, do_attrs.normalized);
    EXPECT_EQ(g_do_attrs.input_height, do_attrs.input_height);
    EXPECT_EQ(g_do_attrs.input_width, do_attrs.input_width);
    EXPECT_EQ(g_do_attrs.objectness_score, do_attrs.objectness_score);
}

TEST(attributes, prior_box_op)
{
    NodeBuilder::get_ops().register_factory<opset1::PriorBox>();
    const auto layer_shape = make_shared<op::Parameter>(element::i64, Shape{128, 128});
    const auto image_shape = make_shared<op::Parameter>(element::i64, Shape{32, 32});

    op::PriorBoxAttrs attrs;
    attrs.min_size = vector<float>{16.f, 32.f};
    attrs.max_size = vector<float>{256.f, 512.f};
    attrs.aspect_ratio = vector<float>{0.66f, 1.56f};
    attrs.density = vector<float>{0.55f};
    attrs.fixed_ratio = vector<float>{0.88f};
    attrs.fixed_size = vector<float>{1.25f};
    attrs.clip = true;
    attrs.flip = false;
    attrs.step = 1.0f;
    attrs.offset = 0.0f;
    attrs.variance = vector<float>{2.22f, 3.14f};
    attrs.scale_all_sizes = true;

    auto prior_box = make_shared<opset1::PriorBox>(layer_shape, image_shape, attrs);
    NodeBuilder builder(prior_box);
    auto g_prior_box = as_type_ptr<opset1::PriorBox>(builder.create());

    const auto prior_box_attrs = prior_box->get_attrs();
    const auto g_prior_box_attrs = g_prior_box->get_attrs();

    EXPECT_EQ(g_prior_box_attrs.min_size, prior_box_attrs.min_size);
    EXPECT_EQ(g_prior_box_attrs.max_size, prior_box_attrs.max_size);
    EXPECT_EQ(g_prior_box_attrs.aspect_ratio, prior_box_attrs.aspect_ratio);
    EXPECT_EQ(g_prior_box_attrs.density, prior_box_attrs.density);
    EXPECT_EQ(g_prior_box_attrs.fixed_ratio, prior_box_attrs.fixed_ratio);
    EXPECT_EQ(g_prior_box_attrs.fixed_size, prior_box_attrs.fixed_size);
    EXPECT_EQ(g_prior_box_attrs.clip, prior_box_attrs.clip);
    EXPECT_EQ(g_prior_box_attrs.flip, prior_box_attrs.flip);
    EXPECT_EQ(g_prior_box_attrs.step, prior_box_attrs.step);
    EXPECT_EQ(g_prior_box_attrs.offset, prior_box_attrs.offset);
    EXPECT_EQ(g_prior_box_attrs.variance, prior_box_attrs.variance);
    EXPECT_EQ(g_prior_box_attrs.scale_all_sizes, prior_box_attrs.scale_all_sizes);
}

TEST(attributes, prior_box_clustered_op)
{
    NodeBuilder::get_ops().register_factory<opset1::PriorBoxClustered>();
    const auto layer_shape = make_shared<op::Parameter>(element::i64, Shape{128, 128});
    const auto image_shape = make_shared<op::Parameter>(element::i64, Shape{32, 32});

    op::PriorBoxClusteredAttrs attrs;
    attrs.widths = vector<float>{128.f, 512.f, 4096.f};
    attrs.heights = vector<float>{128.f, 512.f, 4096.f};
    attrs.clip = true;
    attrs.step_widths = 0.33f;
    attrs.step_heights = 1.55f;
    attrs.offset = 0.77f;
    attrs.variances = vector<float>{0.33f, 1.44f};

    auto prior_box_clust = make_shared<opset1::PriorBoxClustered>(layer_shape, image_shape, attrs);
    NodeBuilder builder(prior_box_clust);
    auto g_prior_box_clust = as_type_ptr<opset1::PriorBoxClustered>(builder.create());

    const auto prior_box_clust_attrs = prior_box_clust->get_attrs();
    const auto g_prior_box_clust_attrs = g_prior_box_clust->get_attrs();

    EXPECT_EQ(g_prior_box_clust_attrs.widths, prior_box_clust_attrs.widths);
    EXPECT_EQ(g_prior_box_clust_attrs.heights, prior_box_clust_attrs.heights);
    EXPECT_EQ(g_prior_box_clust_attrs.clip, prior_box_clust_attrs.clip);
    EXPECT_EQ(g_prior_box_clust_attrs.step_widths, prior_box_clust_attrs.step_widths);
    EXPECT_EQ(g_prior_box_clust_attrs.step_heights, prior_box_clust_attrs.step_heights);
    EXPECT_EQ(g_prior_box_clust_attrs.offset, prior_box_clust_attrs.offset);
    EXPECT_EQ(g_prior_box_clust_attrs.variances, prior_box_clust_attrs.variances);
}

TEST(attributes, proposal_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Proposal>();
    const auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1024, 2, 128, 128});
    const auto class_logits = make_shared<op::Parameter>(element::f32, Shape{1024, 4, 128, 128});
    const auto image_shape = make_shared<op::Parameter>(element::f32, Shape{4});

    op::ProposalAttrs attrs;
    attrs.base_size = 224;
    attrs.pre_nms_topn = 100;
    attrs.post_nms_topn = 110;
    attrs.nms_thresh = 0.12f;
    attrs.feat_stride = 2;
    attrs.min_size = 10;
    attrs.ratio = vector<float>{1.44f, 0.66f};
    attrs.scale = vector<float>{2.25f, 1.83f};
    attrs.clip_before_nms = true;
    attrs.clip_after_nms = true;
    attrs.normalize = false;
    attrs.box_size_scale = 2.f;
    attrs.box_coordinate_scale = 4.55f;
    attrs.framework = string{"nGraph"};

    auto proposal = make_shared<opset1::Proposal>(class_probs, class_logits, image_shape, attrs);
    NodeBuilder builder(proposal);
    auto g_proposal = as_type_ptr<opset1::Proposal>(builder.create());

    const auto proposal_attrs = proposal->get_attrs();
    const auto g_proposal_attrs = g_proposal->get_attrs();

    EXPECT_EQ(g_proposal_attrs.base_size, proposal_attrs.base_size);
    EXPECT_EQ(g_proposal_attrs.pre_nms_topn, proposal_attrs.pre_nms_topn);
    EXPECT_EQ(g_proposal_attrs.post_nms_topn, proposal_attrs.post_nms_topn);
    EXPECT_EQ(g_proposal_attrs.nms_thresh, proposal_attrs.nms_thresh);
    EXPECT_EQ(g_proposal_attrs.feat_stride, proposal_attrs.feat_stride);
    EXPECT_EQ(g_proposal_attrs.min_size, proposal_attrs.min_size);
    EXPECT_EQ(g_proposal_attrs.ratio, proposal_attrs.ratio);
    EXPECT_EQ(g_proposal_attrs.scale, proposal_attrs.scale);
    EXPECT_EQ(g_proposal_attrs.clip_before_nms, proposal_attrs.clip_before_nms);
    EXPECT_EQ(g_proposal_attrs.clip_after_nms, proposal_attrs.clip_after_nms);
    EXPECT_EQ(g_proposal_attrs.normalize, proposal_attrs.normalize);
    EXPECT_EQ(g_proposal_attrs.box_size_scale, proposal_attrs.box_size_scale);
    EXPECT_EQ(g_proposal_attrs.box_coordinate_scale, proposal_attrs.box_coordinate_scale);
    EXPECT_EQ(g_proposal_attrs.framework, proposal_attrs.framework);
}
