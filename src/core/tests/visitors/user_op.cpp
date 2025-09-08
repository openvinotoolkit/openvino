// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/op.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;

enum class TuringModel { XL400, XL1200 };

namespace ov {
template <>
EnumNames<TuringModel>& EnumNames<TuringModel>::get() {
    static auto enum_names =
        EnumNames<TuringModel>("TuringModel", {{"XL400", TuringModel::XL400}, {"XL1200", TuringModel::XL1200}});
    return enum_names;
}

template <>
class AttributeAdapter<TuringModel> : public EnumAttributeAdapterBase<TuringModel> {
public:
    AttributeAdapter(TuringModel& value) : EnumAttributeAdapterBase<TuringModel>(value) {}

    OPENVINO_RTTI("AttributeAdapter<TuringModel>");
};

struct Position {
    float x;
    float y;
    float z;
    bool operator==(const Position& p) const {
        return x == p.x && y == p.y && z == p.z;
    }
    Position& operator=(const Position& p) {
        x = p.x;
        y = p.y;
        z = p.z;
        return *this;
    }
};

template <>
class AttributeAdapter<Position> : public VisitorAdapter {
public:
    AttributeAdapter(Position& value) : m_ref(value) {}
    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("x", m_ref.x);
        visitor.on_attribute("y", m_ref.y);
        visitor.on_attribute("z", m_ref.z);
        return true;
    }
    OPENVINO_RTTI("AttributeAdapter<Position>");

protected:
    Position& m_ref;
};
}  // namespace ov

// Given a Turing machine program and data, return scalar 1 if the program would
// complete, 1 if it would not.
class Oracle : public ov::op::Op {
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
           const ov::Position& position,
           const shared_ptr<Node>& node,
           const NodeVector& node_vector,
           const ParameterVector& parameter_vector,
           const ResultVector& result_vector)
        : Op({program, data}),
          m_turing_model(turing_model),
          m_element_type(element_type),
          m_element_type_t(element_type_t),
          m_val_string(val_string),
          m_val_bool(val_bool),
          m_val_float(val_float),
          m_val_double(val_double),
          m_val_uint8_t(val_uint8_t),
          m_val_uint16_t(val_uint16_t),
          m_val_uint32_t(val_uint32_t),
          m_val_uint64_t(val_uint64_t),
          m_val_int8_t(val_int8_t),
          m_val_int16_t(val_int16_t),
          m_val_int32_t(val_int32_t),
          m_val_int64_t(val_int64_t),
          m_val_size_t(val_size_t),
          m_vec_string(vec_string),
          m_vec_float(vec_float),
          m_vec_double(vec_double),
          m_vec_uint8_t(vec_uint8_t),
          m_vec_uint16_t(vec_uint16_t),
          m_vec_uint32_t(vec_uint32_t),
          m_vec_uint64_t(vec_uint64_t),
          m_vec_int8_t(vec_int8_t),
          m_vec_int16_t(vec_int16_t),
          m_vec_int32_t(vec_int32_t),
          m_vec_int64_t(vec_int64_t),
          m_vec_size_t(vec_size_t),
          m_position(position),
          m_node(node),
          m_node_vector(node_vector),
          m_parameter_vector(parameter_vector),
          m_result_vector(result_vector) {}

    OPENVINO_OP("Oracle", "OracleOpset");

    Oracle() = default;

    TuringModel get_turing_model() const {
        return m_turing_model;
    }
    const element::Type get_element_type() const {
        return m_element_type;
    }
    element::Type_t get_element_type_t() const {
        return m_element_type_t;
    }
    const string& get_val_string() const {
        return m_val_string;
    }
    bool get_val_bool() const {
        return m_val_bool;
    }
    bool get_val_float() const {
        return m_val_float;
    }
    bool get_val_double() const {
        return m_val_double;
    }
    uint64_t get_val_uint8_t() const {
        return m_val_uint8_t;
    }
    uint64_t get_val_uint16_t() const {
        return m_val_uint16_t;
    }
    uint64_t get_val_uint32_t() const {
        return m_val_uint32_t;
    }
    uint64_t get_val_uint64_t() const {
        return m_val_uint64_t;
    }
    int64_t get_val_int8_t() const {
        return m_val_int8_t;
    }
    int64_t get_val_int16_t() const {
        return m_val_int16_t;
    }
    int64_t get_val_int32_t() const {
        return m_val_int32_t;
    }
    int64_t get_val_int64_t() const {
        return m_val_int64_t;
    }
    size_t get_val_size_t() const {
        return m_val_size_t;
    }
    const vector<uint8_t>& get_vec_uint8_t() const {
        return m_vec_uint8_t;
    }
    const vector<uint16_t>& get_vec_uint16_t() const {
        return m_vec_uint16_t;
    }
    const vector<uint32_t>& get_vec_uint32_t() const {
        return m_vec_uint32_t;
    }
    const vector<uint64_t>& get_vec_uint64_t() const {
        return m_vec_uint64_t;
    }
    const vector<int8_t>& get_vec_int8_t() const {
        return m_vec_int8_t;
    }
    const vector<int16_t>& get_vec_int16_t() const {
        return m_vec_int16_t;
    }
    const vector<int32_t>& get_vec_int32_t() const {
        return m_vec_int32_t;
    }
    const vector<int64_t>& get_vec_int64_t() const {
        return m_vec_int64_t;
    }
    const vector<string>& get_vec_string() const {
        return m_vec_string;
    }
    const vector<float>& get_vec_float() const {
        return m_vec_float;
    }
    const vector<double>& get_vec_double() const {
        return m_vec_double;
    }
    const vector<size_t>& get_vec_size_t() const {
        return m_vec_size_t;
    }
    const ov::Position& get_position() const {
        return m_position;
    }
    const shared_ptr<Node>& get_node() const {
        return m_node;
    }
    const NodeVector& get_node_vector() const {
        return m_node_vector;
    }
    const ParameterVector& get_parameter_vector() const {
        return m_parameter_vector;
    }
    const ResultVector& get_result_vector() const {
        return m_result_vector;
    }
    shared_ptr<Node> clone_with_new_inputs(const OutputVector& args) const override {
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

    void validate_and_infer_types() override {
        set_output_type(0, element::i64, {});
    }
    bool visit_attributes(AttributeVisitor& visitor) override {
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
    ov::Position m_position;
    shared_ptr<Node> m_node;
    NodeVector m_node_vector;
    ParameterVector m_parameter_vector;
    ResultVector m_result_vector;
};

TEST(attributes, user_op) {
    ov::test::NodeBuilder::opset().insert<Oracle>();
    auto program = make_shared<ov::op::v0::Parameter>(element::i32, Shape{200});
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{200});
    auto result = make_shared<ov::op::v0::Result>(data);
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
                                      ov::Position{1.3f, 5.1f, 2.3f},
                                      data,
                                      NodeVector{program, result, data},
                                      ParameterVector{data, data, program},
                                      ResultVector{result});
    ov::test::NodeBuilder builder;
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
    auto g_oracle = ov::as_type_ptr<Oracle>(builder.create());

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
