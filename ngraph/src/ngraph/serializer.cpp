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

#include <fstream>
#include <functional>
#include <queue>
#include <stack>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/provenance.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"

using namespace ngraph;
using namespace std;
using json = nlohmann::json;
using const_data_callback_t = shared_ptr<Node>(const string&, const element::Type&, const Shape&);

static json write_element_type(const ngraph::element::Type& n);
static element::Type read_element_type(json j);
static json write_partial_shape(const PartialShape& s);
static PartialShape read_partial_shape(json j);

static bool s_serialize_output_shapes_enabled = getenv_bool("NGRAPH_SERIALIZER_OUTPUT_SHAPES");

void ngraph::set_serialize_output_shapes(bool enable)
{
    s_serialize_output_shapes_enabled = enable;
}

bool ngraph::get_serialize_output_shapes()
{
    return s_serialize_output_shapes_enabled;
}

namespace
{
    // This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
    // Abs,
    // Acos,
    // ...
    enum class OP_TYPEID
    {
#define VSUF0(NAME) NAME
#define VSUF1(NAME) NAME##_v1
#define VSUF3(NAME) NAME##_v3
#define NGRAPH_OP(NAME, NAMESPACE, VERSION) VSUF##VERSION(NAME),
#include "ngraph/op/op_version_tbl.hpp"
#undef NGRAPH_OP
        UnknownOp
    };
}

static OP_TYPEID get_typeid(const NodeTypeInfo& type_info)
{
    // This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
    // {Abs::type_info, OP_TYPEID::Abs},
    // {Acos::type_info, OP_TYPEID::Acos},
    // ...
    static const map<NodeTypeInfo, OP_TYPEID> type_info_map{
#define NGRAPH_OP(NAME, NAMESPACE, VERSION)                                                        \
    {NAMESPACE::NAME::type_info, OP_TYPEID::VSUF##VERSION(NAME)},
#include "ngraph/op/op_version_tbl.hpp"
#undef NGRAPH_OP
    };
    OP_TYPEID rc = OP_TYPEID::UnknownOp;

    auto it = type_info_map.find(type_info);
    if (it != type_info_map.end())
    {
        rc = it->second;
    }
    return rc;
}

bool has_key(json j, const std::string& key)
{
    return j.count(key) != 0;
}

template <typename T>
T get_or_default(json j, const std::string& key, const T& default_value)
{
    return has_key(j, key) ? j.at(key).get<T>() : default_value;
}

class JSONAttributeSerializer : public AttributeVisitor
{
public:
    JSONAttributeSerializer(json& j)
        : m_json(j)
    {
    }
    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override
    {
        NGRAPH_CHECK(false, "Adapter ", adapter.get_type_info().name, " is not handled");
    }
    void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int64_t>>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint64_t>>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name,
                    ValueAccessor<std::vector<std::string>>& adapter) override
    {
        m_json[name] = adapter.get();
    }

protected:
    json& m_json;
};

class JSONSerializer
{
public:
    void set_indent(size_t indent) { m_indent = indent; }
    void set_serialize_output_shapes(bool serialize_output_shapes)
    {
        m_serialize_output_shapes = serialize_output_shapes;
    }

    void set_binary_constant_data(bool binary_constant_data)
    {
        m_binary_constant_data = binary_constant_data;
    }

    json serialize_function(const Function& function);
    json serialize_output(const Output<Node>& output);
    json serialize_parameter_vector(const ParameterVector& parameters);
    json serialize_output_vector(const OutputVector& output_vector);
    json serialize_node(const Node& node);
    json serialize_axis_set(const AxisSet& axis_set);
    json serialize_tensor_iterator_input_description(
        const std::shared_ptr<op::TensorIterator::InputDescription>&);
    json serialize_tensor_iterator_output_description(
        const std::shared_ptr<op::TensorIterator::OutputDescription>&);

protected:
    size_t m_indent{0};
    bool m_serialize_output_shapes{false};
    bool m_binary_constant_data{false};
    json m_json_nodes;
};

class JSONAttributeDeserializer : public AttributeVisitor
{
public:
    JSONAttributeDeserializer(json& j)
        : m_json(j)
    {
    }
    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override
    {
        NGRAPH_CHECK(false, "Adapter ", adapter.get_type_info().name, " is not handled");
    }
    void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::string>());
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<bool>());
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<int64_t>());
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<double>());
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<std::vector<int64_t>>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::vector<int64_t>>());
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint64_t>>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::vector<uint64_t>>());
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::vector<float>>());
        }
    }
    void on_adapter(const std::string& name,
                    ValueAccessor<std::vector<std::string>>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::vector<std::string>>());
        }
    }

protected:
    json& m_json;
};

class JSONDeserializer
{
public:
    void set_const_data_callback(function<const_data_callback_t> const_data_callback)
    {
        m_const_data_callback = const_data_callback;
    }

    shared_ptr<Function> deserialize_function(json j);
    Output<Node> deserialize_output(json j);
    OutputVector deserialize_output_vector(json j);
    ParameterVector deserialize_parameter_vector(json j);
    shared_ptr<Node> deserialize_node_reference(json j);
    shared_ptr<Node> deserialize_node(json j);
    AxisSet deserialize_axis_set(json j);
    shared_ptr<op::TensorIterator::InputDescription>
        deserialize_tensor_iterator_input_description(json j);
    shared_ptr<op::TensorIterator::OutputDescription>
        deserialize_tensor_iterator_output_description(json j);

protected:
    unordered_map<string, shared_ptr<Node>> m_node_map;
    unordered_map<string, shared_ptr<Function>> m_function_map;
    function<const_data_callback_t> m_const_data_callback;
};

static string
    serialize(shared_ptr<ngraph::Function> func, size_t indent, bool binary_constant_data);

static json write_dimension(Dimension d)
{
    if (d.is_dynamic())
    {
        return nullptr;
    }
    else
    {
        return d.get_length();
    }
}

static Dimension read_dimension(json j)
{
    if (j.is_null())
    {
        return Dimension::dynamic();
    }
    else
    {
        return Dimension(static_cast<int64_t>(j));
    }
}

static json write_partial_shape(const PartialShape& s)
{
    if (s.rank().is_dynamic())
    {
        return nullptr;
    }
    else
    {
        std::vector<json> vals(s.rank().get_length());
        for (size_t i = 0; i < vals.size(); i++)
        {
            vals[i] = write_dimension(s[i]);
        }
        return move(vals);
    }
}

static PartialShape read_partial_shape(json j)
{
    if (j.is_null())
    {
        return PartialShape::dynamic();
    }
    else
    {
        std::vector<Dimension> dims(j.size());
        for (size_t i = 0; i < j.size(); i++)
        {
            dims[i] = read_dimension(j[i]);
        }
        return PartialShape(dims);
    }
}

static json write_auto_broadcast(const op::AutoBroadcastSpec& autob)
{
    json j;
    j["type"] = autob.m_type;
    j["axis"] = autob.m_axis;
    return j;
}

static op::AutoBroadcastSpec
    read_auto_broadcast(json js_node,
                        const std::string& attr,
                        const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec())
{
    if (has_key(js_node, attr))
    {
        json j = js_node[attr];
        return op::AutoBroadcastSpec(static_cast<op::AutoBroadcastType>(j.at("type")),
                                     j.at("axis").get<int64_t>());
    }
    else
    {
        return autob;
    }
}

static op::PadType read_pad_type(json node_js)
{
    return has_key(node_js, "pad_type") ? static_cast<op::PadType>(node_js.at("pad_type"))
                                        : op::PadType::EXPLICIT;
}

static op::PadMode read_pad_mode(json node_js)
{
    return has_key(node_js, "pad_mode") ? static_cast<op::PadMode>(node_js.at("pad_mode"))
                                        : op::PadMode::CONSTANT;
}

static json write_element_type(const ngraph::element::Type& n)
{
    json j;
    j = n.c_type_string();
    return j;
}

static element::Type read_element_type(json j)
{
    size_t bitwidth = 0;
    bool is_real = false;
    bool is_signed = false;
    bool is_quantized = false;
    string c_type_string = "";
    if (j.is_object())
    {
        bitwidth = j.at("bitwidth").get<size_t>();
        is_real = j.at("is_real").get<bool>();
        is_signed = j.at("is_signed").get<bool>();
        is_quantized = j.at("is_quantized").get<bool>();
        c_type_string = j.at("c_type_string").get<string>();
    }
    else
    {
        string c_type = j.get<string>();
        for (const element::Type* t : element::Type::get_known_types())
        {
            if (t->c_type_string() == c_type)
            {
                bitwidth = t->bitwidth();
                is_real = t->is_real();
                is_signed = t->is_signed();
                is_quantized = t->is_quantized();
                c_type_string = t->c_type_string();
                break;
            }
        }
    }
    return element::Type(bitwidth, is_real, is_signed, is_quantized, c_type_string);
}

void ngraph::serialize(const string& path, shared_ptr<ngraph::Function> func, size_t indent)
{
    ofstream out(path);
    serialize(out, func, indent);
}

void ngraph::serialize(ostream& out, shared_ptr<ngraph::Function> func, size_t indent)
{
    out << ::serialize(func, indent, false);
}

#if defined ENABLE_CPIO_FILE
static void serialize_to_cpio(ostream& out, shared_ptr<ngraph::Function> func, size_t indent)
{
    string j = ::serialize(func, indent, true);
    cpio::Writer writer(out);
    writer.write(func->get_name(), j.c_str(), static_cast<uint32_t>(j.size()));

    traverse_nodes(const_cast<Function*>(func.get()),
                   [&](shared_ptr<Node> node) {
                       if (auto c = node->as_type<op::Constant>())
                       {
                           uint32_t size =
                               static_cast<uint32_t>(shape_size(c->get_output_shape(0)) *
                                                     c->get_output_element_type(0).size());
                           writer.write(c->get_name(), c->get_data_ptr(), size);
                       }
                   },
                   true);
}
#endif

static string serialize(shared_ptr<Function> func, size_t indent, bool binary_constant_data)
{
    JSONSerializer serializer;
    serializer.set_binary_constant_data(binary_constant_data);
    serializer.set_indent(indent);
    serializer.set_serialize_output_shapes(s_serialize_output_shapes_enabled);

    json j;
    j.push_back(serializer.serialize_function(*func));

    string rc;
    if (indent == 0)
    {
        rc = j.dump();
    }
    else
    {
        rc = j.dump(static_cast<int>(indent));
    }
    return rc;
}

std::string ngraph::serialize(std::shared_ptr<ngraph::Function> func, size_t indent)
{
    return ::serialize(func, indent, false);
}

shared_ptr<ngraph::Function> ngraph::deserialize(istream& in)
{
    shared_ptr<Function> rc;
    if (cpio::is_cpio(in))
    {
        cpio::Reader reader(in);
        vector<cpio::FileInfo> file_info = reader.get_file_info();
        if (file_info.size() > 0)
        {
            // The first file is the model
            uint32_t size = static_cast<uint32_t>(file_info[0].get_size());
            char* data = new char[size];
            reader.read(file_info[0].get_name(), data, size);
            string jstr(data, size);
            delete[] data;
            json js = json::parse(jstr);
            JSONDeserializer deserializer;
            deserializer.set_const_data_callback(
                [&](const string& const_name, const element::Type& et, const Shape& shape) {
                    shared_ptr<Node> const_node;
                    for (const cpio::FileInfo& info : file_info)
                    {
                        if (info.get_name() == const_name)
                        {
                            void* const_data = ngraph_malloc(info.get_size());
                            reader.read(const_name, const_data, info.get_size());
                            const_node = make_shared<op::Constant>(et, shape, const_data);
                            ngraph_free(const_data);
                            break;
                        }
                    }
                    return const_node;
                });
            for (json func : js)
            {
                rc = deserializer.deserialize_function(func);
            }
        }
    }
    else
    {
        // json file?
        std::stringstream ss;
        ss << in.rdbuf();
        rc = deserialize(ss.str());
    }
    return rc;
}

shared_ptr<ngraph::Function> ngraph::deserialize(const string& s)
{
    shared_ptr<Function> rc;
    if (file_util::exists(s))
    {
        // s is a file and not a json string
        ifstream in(s, ios_base::binary | ios_base::in);
        rc = deserialize(in);
    }
    else
    {
        json js = json::parse(s);
        JSONDeserializer deserializer;
        for (json func : js)
        {
            rc = deserializer.deserialize_function(func);
        }
    }
    return rc;
}

json JSONSerializer::serialize_parameter_vector(const ParameterVector& parameters)
{
    json json_parameters = json::array();
    for (auto param : parameters)
    {
        json_parameters.push_back(param->get_name());
    }
    return json_parameters;
}

json JSONSerializer::serialize_function(const Function& f)
{
    json function;
    function["name"] = f.get_name();
    function["parameters"] = serialize_parameter_vector(f.get_parameters());

    // TODO Functions can return multiple results
    for (size_t i = 0; i < f.get_output_size(); ++i)
    {
        function["result"].push_back(f.get_output_op(i)->get_name());
    }

    json nodes;
    for (shared_ptr<Node> node : f.get_ordered_ops())
    {
        nodes.push_back(serialize_node(*node));
    }

    function["ops"] = nodes;
    return function;
}

template <typename T>
T get_value(json js, const string& key)
{
    T rc = {};
    auto it = js.find(key);
    if (it != js.end())
    {
        rc = it->get<T>();
    }
    return rc;
}

shared_ptr<Node> JSONDeserializer::deserialize_node_reference(json j)
{
    const string& name = j;
    return m_node_map.at(name);
}

Output<Node> JSONDeserializer::deserialize_output(json j)
{
    size_t index;
    json json_node_reference;
    if (j.is_string())
    {
        json_node_reference = j;
        index = 0;
    }
    else if (j.is_object())
    {
        json_node_reference = j["node"];
        index = j["index"];
    }
    else
    {
        throw ngraph_error("Expected string or object an output while deserializing");
    }
    return Output<Node>(deserialize_node_reference(json_node_reference), index);
}

OutputVector JSONDeserializer::deserialize_output_vector(json j)
{
    OutputVector result;
    if (j.is_array())
    {
        for (json jelt : j)
        {
            result.push_back(deserialize_output(jelt));
        }
    }
    return result;
}

json JSONSerializer::serialize_axis_set(const AxisSet& axis_set)
{
    return static_cast<set<size_t>>(axis_set);
}

AxisSet JSONDeserializer::deserialize_axis_set(json j)
{
    AxisSet result;
    if (j.is_array())
    {
        result = j.get<set<size_t>>();
    }
    return result;
}

json JSONSerializer::serialize_tensor_iterator_input_description(
    const std::shared_ptr<op::TensorIterator::InputDescription>& input_description)
{
    json result;
    if (auto slice = as_type_ptr<op::TensorIterator::SliceInputDescription>(input_description))
    {
        result["kind"] = "slice";
        result["input_index"] = slice->m_input_index;
        result["body_parameter_index"] = slice->m_body_parameter_index;
        result["start"] = slice->m_start;
        result["stride"] = slice->m_stride;
        result["part_size"] = slice->m_part_size;
        result["end"] = slice->m_end;
        result["axis"] = slice->m_axis;
    }
    else if (auto merged =
                 as_type_ptr<op::TensorIterator::MergedInputDescription>(input_description))
    {
        result["kind"] = "merged";
        result["input_index"] = merged->m_input_index;
        result["body_parameter_index"] = merged->m_body_parameter_index;
        result["body_value_index"] = merged->m_body_value_index;
    }
    else if (auto constant =
                 as_type_ptr<op::TensorIterator::InvariantInputDescription>(input_description))
    {
        result["kind"] = "constant";
        result["input_index"] = constant->m_input_index;
        result["body_parameter_index"] = constant->m_body_parameter_index;
    }
    else
    {
        NGRAPH_UNREACHABLE("Unknown input description type");
    }
    return result;
}

shared_ptr<op::TensorIterator::InputDescription>
    JSONDeserializer::deserialize_tensor_iterator_input_description(json j)
{
    string kind = j["kind"];
    shared_ptr<op::TensorIterator::InputDescription> result;
    if (kind == "slice")
    {
        uint64_t input_index = j["input_index"].get<uint64_t>();
        uint64_t body_parameter_index = j["body_parameter_index"].get<uint64_t>();
        int64_t start = j["start"].get<int64_t>();
        int64_t stride = j["stride"].get<int64_t>();
        uint64_t part_size = j["part_size"].get<int64_t>();
        int64_t end = j["end"].get<int64_t>();
        int64_t axis = j["axis"].get<int64_t>();
        result = make_shared<op::TensorIterator::SliceInputDescription>(
            input_index, body_parameter_index, start, stride, part_size, end, axis);
    }
    else if (kind == "merged")
    {
        uint64_t input_index = j["input_index"].get<uint64_t>();
        uint64_t body_parameter_index = j["body_parameter_index"].get<uint64_t>();
        uint64_t body_value_index = j["body_value_index"].get<uint64_t>();
        result = make_shared<op::TensorIterator::MergedInputDescription>(
            input_index, body_parameter_index, body_value_index);
    }
    else if (kind == "constant")
    {
        uint64_t input_index = j["input_index"].get<uint64_t>();
        uint64_t body_parameter_index = j["body_parameter_index"].get<uint64_t>();
        result = make_shared<op::TensorIterator::InvariantInputDescription>(input_index,
                                                                            body_parameter_index);
    }
    else
    {
        NGRAPH_UNREACHABLE("Unknown input description type: ", kind);
    }
    return result;
}

json JSONSerializer::serialize_tensor_iterator_output_description(
    const std::shared_ptr<op::TensorIterator::OutputDescription>& output_description)
{
    json result;
    if (auto concat = as_type_ptr<op::TensorIterator::ConcatOutputDescription>(output_description))
    {
        result["kind"] = "concat";
        result["body_value_index"] = concat->m_body_value_index;
        result["output_index"] = concat->m_output_index;
        result["start"] = concat->m_start;
        result["stride"] = concat->m_stride;
        result["part_size"] = concat->m_part_size;
        result["end"] = concat->m_end;
        result["axis"] = concat->m_axis;
    }
    else if (auto body_output =
                 as_type_ptr<op::TensorIterator::BodyOutputDescription>(output_description))
    {
        result["kind"] = "body_output";
        result["body_value_index"] = body_output->m_body_value_index;
        result["output_index"] = body_output->m_output_index;
        result["iteration"] = body_output->m_iteration;
    }
    else
    {
        NGRAPH_UNREACHABLE("Unknown input description type");
    }
    return result;
}

std::shared_ptr<op::TensorIterator::OutputDescription>
    JSONDeserializer::deserialize_tensor_iterator_output_description(json j)
{
    string kind = j["kind"];
    shared_ptr<op::TensorIterator::OutputDescription> result;
    if (kind == "concat")
    {
        uint64_t body_value_index = j["body_value_index"].get<uint64_t>();
        uint64_t output_index = j["output_index"].get<uint64_t>();
        int64_t start = j["start"].get<int64_t>();
        int64_t stride = j["stride"].get<int64_t>();
        uint64_t part_size = j["part_size"].get<int64_t>();
        int64_t end = j["end"].get<int64_t>();
        int64_t axis = j["axis"].get<int64_t>();
        result = make_shared<op::TensorIterator::ConcatOutputDescription>(
            body_value_index, output_index, start, stride, part_size, end, axis);
    }
    else if (kind == "body_output")
    {
        uint64_t body_value_index = j["body_value_index"].get<uint64_t>();
        uint64_t output_index = j["output_index"].get<uint64_t>();
        int64_t iteration = j["iteration"].get<int64_t>();
        result = make_shared<op::TensorIterator::BodyOutputDescription>(
            body_value_index, output_index, iteration);
    }
    else
    {
        NGRAPH_UNREACHABLE("Unknown input description type: ", kind);
    }
    return result;
}

ParameterVector JSONDeserializer::deserialize_parameter_vector(json json_parameters)
{
    std::vector<std::shared_ptr<op::Parameter>> params;
    for (auto& param_ref : json_parameters)
    {
        params.push_back(as_type_ptr<op::Parameter>(deserialize_node_reference(param_ref)));
    }
    return params;
}

shared_ptr<Function> JSONDeserializer::deserialize_function(json func_js)
{
    string func_name = func_js.at("name").get<string>();
    vector<json> func_result = func_js.at("result");
    for (json node_js : func_js.at("ops"))
    {
        deserialize_node(node_js);
    }

    // This handles both graphs w/ `op::Result` and legacy graphs w/o it
    // If we are dealing w/ a legacy graph, add op::Result for each output node
    ResultVector result;
    size_t results = 0;
    for (auto& result_ref : func_result)
    {
        auto fr = deserialize_node_reference(result_ref);
        if (auto res = as_type_ptr<op::Result>(fr))
        {
            result.push_back(res);
            // make sure we have `op::Result` on top of all outputs
            results++;
        }
        else
        {
            result.push_back(std::make_shared<op::Result>(fr));
        }
    }

    if (results != 0 && results != func_result.size())
    {
        throw ngraph_error(
            "Graph serialization is inconsistent. Some op::Results appear to be missing");
    }

    ParameterVector params = deserialize_parameter_vector(func_js.at("parameters"));

    shared_ptr<Function> rc{make_shared<Function>(result, params, func_name)};
    m_function_map[func_name] = rc;
    return rc;
}

// This helps with conversions to old-style shared-ptr<Node> and new-style Output&
// arguments to node constructors. Uses of OutputHelper should be replaced with Output
// when all op constructors use the new style arguments.
struct OutputHelper
{
    OutputHelper(const Output<Node>& output)
        : m_output(output)
    {
    }

    operator shared_ptr<Node>() const { return get_output_element(m_output); }
    operator const Output<Node>&() const { return m_output; }
    Output<Node> m_output;
};

// This helps with conversions to old-style shared-ptr<Node> and new-style Output&
// arguments to node constructors. Uses of OutputVectorHelper should be replaced with OutputVector
// when all op constructors use the new style arguments.
struct OutputVectorHelper
{
    OutputVectorHelper(const OutputVector& output_vector)
        : m_vector(output_vector)
    {
    }
    OutputVectorHelper() = default;
    OutputHelper operator[](size_t i) const { return OutputHelper(m_vector[i]); }
    void push_back(const Output<Node>& output) { m_vector.push_back(output); }
    size_t size() const { return m_vector.size(); }
    operator vector<shared_ptr<Node>>() const
    {
        vector<shared_ptr<Node>> result;
        for (auto& o : m_vector)
        {
            result.push_back(OutputHelper(o));
        }
        return result;
    }
    operator const OutputVector&() const { return m_vector; }
    OutputVector m_vector;
};

shared_ptr<Node> JSONDeserializer::deserialize_node(json node_js)
{
    auto& factory_registry = FactoryRegistry<Node>::get();
    shared_ptr<Node> node;
    try
    {
        string node_op = node_js.at("op").get<string>();
        size_t op_version = get_value<size_t>(node_js, "op_version");
        Node::type_info_t type_info{node_op.c_str(), op_version};
        string node_name = node_js.at("name").get<string>();
        string friendly_name = get_value<string>(node_js, "friendly_name");
        vector<json> control_deps_inputs = get_value<vector<json>>(node_js, "control_deps");
        vector<string> node_outputs = get_value<vector<string>>(node_js, "outputs");
        OutputVectorHelper args(deserialize_output_vector(node_js["inputs"]));
        if (has_key(node_js, "attribute_visitor"))
        {
            if (factory_registry.has_factory(type_info))
            {
                node = shared_ptr<Node>(factory_registry.create(type_info));
                JSONAttributeDeserializer visitor(node_js);
                node->set_arguments(static_cast<OutputVector>(args));
                node->visit_attributes(visitor);
                for (auto& control_dep : control_deps_inputs)
                {
                    node->add_control_dependency(deserialize_node_reference(control_dep));
                }

                if (!friendly_name.empty())
                {
                    node->set_friendly_name(friendly_name);
                }
                else
                {
                    node->set_friendly_name(node_name);
                }
                if (ngraph::get_provenance_enabled())
                {
                    std::vector<json> prov_js = node_js.at("provenance_tags");
                    for (auto prov_tag : prov_js)
                    {
                        node->add_provenance_tag(prov_tag);
                    }
                }
                node->constructor_validate_and_infer_types();
                m_node_map[node_name] = node;
                return node;
            }
        }

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
// #pragma GCC diagnostic error "-Wswitch-enum"
// #pragma GCC diagnostic error "-Wimplicit-fallthrough"
#endif

        switch (get_typeid(type_info))
        {
        case OP_TYPEID::Abs:
        {
            node = make_shared<op::Abs>(args[0]);
            break;
        }
        case OP_TYPEID::Acos:
        {
            node = make_shared<op::Acos>(args[0]);
            break;
        }
        case OP_TYPEID::Add:
        {
            node = make_shared<op::v0::Add>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::All:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::All>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::AllReduce:
        {
            node = make_shared<op::AllReduce>(args[0]);
            break;
        }
        case OP_TYPEID::And:
        {
            node = make_shared<op::And>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Any:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::Any>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::ArgMin:
        {
            auto axis = node_js.at("axis").get<size_t>();
            auto target_type = read_element_type(node_js.at("index_element_type"));
            node = make_shared<op::ArgMin>(args[0], axis, target_type);
            break;
        }
        case OP_TYPEID::ArgMax:
        {
            auto axis = node_js.at("axis").get<size_t>();
            auto target_type = read_element_type(node_js.at("index_element_type"));
            node = make_shared<op::ArgMax>(args[0], axis, target_type);
            break;
        }
        case OP_TYPEID::Asin:
        {
            node = make_shared<op::Asin>(args[0]);
            break;
        }
        case OP_TYPEID::Atan:
        {
            node = make_shared<op::Atan>(args[0]);
            break;
        }

        case OP_TYPEID::BatchNormInference:
        {
            auto epsilon = node_js.at("eps").get<double>();
            // Odd order for back-compatibility
            node = make_shared<op::BatchNormInference>(
                args[2], args[0], args[1], args[3], args[4], epsilon);
            break;
        }
        case OP_TYPEID::Broadcast:
        {
            auto shape = node_js.at("shape").get<vector<size_t>>();
            auto axes = deserialize_axis_set(node_js.at("axes"));
            node = make_shared<op::v0::Broadcast>(args[0], shape, axes);
            break;
        }
        case OP_TYPEID::BroadcastDistributed:
        {
            node = make_shared<op::BroadcastDistributed>(args[0]);
            break;
        }
        case OP_TYPEID::BroadcastLike:
        {
            auto initial_axes = deserialize_axis_set(node_js.at("initial_axes"));
            node = make_shared<op::BroadcastLike>(args[0], args[1], initial_axes);
            break;
        }
        case OP_TYPEID::Ceiling:
        {
            node = make_shared<op::Ceiling>(args[0]);
            break;
        }
        case OP_TYPEID::Clamp:
        {
            const auto clamp_min = node_js.at("min").get<float>();
            const auto clamp_max = node_js.at("max").get<float>();
            node = make_shared<op::Clamp>(args[0], clamp_min, clamp_max);
            break;
        }
        case OP_TYPEID::Concat:
        {
            auto axis = node_js.at("axis").get<size_t>();
            node = make_shared<op::Concat>(static_cast<OutputVector>(args), axis);
            break;
        }
        case OP_TYPEID::Constant:
        {
            auto type_node_js =
                has_key(node_js, "element_type") ? node_js : node_js.at("value_type");
            auto element_type = read_element_type(type_node_js.at("element_type"));
            auto shape = type_node_js.at("shape");
            auto value = node_js.at("value").get<vector<string>>();
            node = make_shared<op::Constant>(element_type, shape, value);
            break;
        }
        case OP_TYPEID::Convert:
        {
            auto target_type = read_element_type(node_js.at("target_type"));
            node = make_shared<op::Convert>(args[0], target_type);
            break;
        }
        case OP_TYPEID::Convolution:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();

            // For backwards compatibility, we accept "image_dilation_strides" in place of
            // "data_dilation_strides", and we also allow it to be omitted altogether.
            json data_dilation_strides;
            if (has_key(node_js, "data_dilation_strides"))
            {
                data_dilation_strides = node_js["data_dilation_strides"];
            }
            else if (has_key(node_js, "image_dilation_strides"))
            {
                data_dilation_strides = node_js["image_dilation_strides"];
            }

            op::PadType pad_type = read_pad_type(node_js);

            if (data_dilation_strides.empty())
            {
                node = make_shared<op::v0::Convolution>(args[0],
                                                        args[1],
                                                        window_movement_strides,
                                                        window_dilation_strides,
                                                        padding_below,
                                                        padding_above);
            }
            else
            {
                node = make_shared<op::v0::Convolution>(
                    args[0],
                    args[1],
                    window_movement_strides,
                    window_dilation_strides,
                    padding_below,
                    padding_above,
                    data_dilation_strides.get<std::vector<size_t>>(),
                    pad_type);
            }
            break;
        }
        case OP_TYPEID::ConvolutionBackpropData:
        {
            auto data_batch_shape = node_js.at("data_batch_shape").get<vector<size_t>>();
            auto window_movement_strides_forward =
                node_js.at("window_movement_strides_forward").get<vector<size_t>>();
            auto window_dilation_strides_forward =
                node_js.at("window_dilation_strides_forward").get<vector<size_t>>();
            auto padding_below_forward =
                node_js.at("padding_below_forward").get<vector<std::ptrdiff_t>>();
            auto padding_above_forward =
                node_js.at("padding_above_forward").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides_forward =
                node_js.at("data_dilation_strides_forward").get<vector<size_t>>();
            node = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                                args[0],
                                                                args[1],
                                                                window_movement_strides_forward,
                                                                window_dilation_strides_forward,
                                                                padding_below_forward,
                                                                padding_above_forward,
                                                                data_dilation_strides_forward);
            break;
        }
        case OP_TYPEID::ConvolutionBias:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();

            node = make_shared<op::ConvolutionBias>(args[0],
                                                    args[1],
                                                    args[2],
                                                    window_movement_strides,
                                                    window_dilation_strides,
                                                    padding_below,
                                                    padding_above,
                                                    data_dilation_strides);
            break;
        }
        case OP_TYPEID::ConvolutionBiasAdd:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();

            node = make_shared<op::ConvolutionBiasAdd>(args[0],
                                                       args[1],
                                                       args[2],
                                                       args[3],
                                                       window_movement_strides,
                                                       window_dilation_strides,
                                                       padding_below,
                                                       padding_above,
                                                       data_dilation_strides);
            break;
        }
        case OP_TYPEID::Cos:
        {
            node = make_shared<op::Cos>(args[0]);
            break;
        }
        case OP_TYPEID::Cosh:
        {
            node = make_shared<op::Cosh>(args[0]);
            break;
        }
        case OP_TYPEID::CumSum:
        {
            auto exclusive = node_js.at("exclusive");
            auto reverse = node_js.at("reverse");
            node = make_shared<op::CumSum>(args[0], args[1], exclusive, reverse);
            break;
        }
        case OP_TYPEID::CrossEntropy:
        {
            auto soft_label = node_js.at("soft_label");
            auto ignore_index = node_js.at("ignore_index");
            node = make_shared<op::CrossEntropy>(args[0], args[1], soft_label, ignore_index);
            break;
        }
        case OP_TYPEID::CropAndResize:
        {
            auto resize_method =
                as_type<op::CropAndResize::ResizeMethod>(node_js.at("resize_method").get<string>());
            auto extrapolation_value = node_js.at("extrapolation_value").get<float>();
            node = make_shared<op::CropAndResize>(
                args[0], args[1], args[2], args[3], resize_method, extrapolation_value);
            break;
        }
        case OP_TYPEID::CTCGreedyDecoder: { break;
        }
        case OP_TYPEID::DeformableConvolution_v1:
        {
            const auto strides = node_js.at("strides").get<vector<size_t>>();
            const auto dilations = node_js.at("dilations").get<vector<size_t>>();
            const auto pads_begin = node_js.at("pads_begin").get<vector<std::ptrdiff_t>>();
            const auto pads_end = node_js.at("pads_end").get<vector<std::ptrdiff_t>>();
            const auto group = node_js.at("group").get<size_t>();
            const auto deformable_group = node_js.at("deformable_group").get<size_t>();

            const op::PadType auto_pad = read_pad_type(node_js);

            node = make_shared<op::v1::DeformableConvolution>(args[0],
                                                              args[1],
                                                              args[2],
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              auto_pad,
                                                              group,
                                                              deformable_group);
            break;
        }
        case OP_TYPEID::DepthToSpace:
        {
            auto mode = node_js.at("mode").get<op::DepthToSpace::DepthToSpaceMode>();
            auto block_size = node_js.at("block_size").get<size_t>();
            node = make_shared<op::DepthToSpace>(args[0], mode, block_size);
            break;
        }
        case OP_TYPEID::DetectionOutput: { break;
        }
        case OP_TYPEID::Dequantize:
        {
            auto type = read_element_type(node_js.at("type"));
            auto axes = deserialize_axis_set(node_js.at("axes"));
            node = make_shared<op::Dequantize>(args[0], args[1], args[2], type, axes);
            break;
        }
        case OP_TYPEID::Divide:
        {
            bool pythondiv = get_or_default(node_js, "pythondiv", true);
            node = make_shared<op::v0::Divide>(
                args[0], args[1], pythondiv, read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Dot:
        {
            // For backwards compatibility, reduction_axes_count is optional.
            if (has_key(node_js, "reduction_axes_count"))
            {
                size_t reduction_axes_count = node_js["reduction_axes_count"].get<size_t>();
                node = make_shared<op::Dot>(args[0], args[1], reduction_axes_count);
            }
            else
            {
                node = make_shared<op::Dot>(args[0], args[1]);
            }
            break;
        }
        case OP_TYPEID::DynBroadcast:
        {
            node = make_shared<op::DynBroadcast>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::DynPad:
        {
            node = make_shared<op::DynPad>(args[0], args[1], args[2], args[3]);
            break;
        }
        case OP_TYPEID::DynReplaceSlice:
        {
            auto lower_bounds_mask = node_js.at("lower_bounds_mask").get<set<size_t>>();
            auto upper_bounds_mask = node_js.at("upper_bounds_mask").get<set<size_t>>();
            auto new_axis = node_js.at("new_axis").get<set<size_t>>();
            auto shrink_axis = node_js.at("shrink_axis").get<set<size_t>>();
            auto ellipsis_mask = node_js.at("ellipsis_mask").get<set<size_t>>();
            node = make_shared<op::DynReplaceSlice>(args[0],
                                                    args[1],
                                                    args[2],
                                                    args[3],
                                                    args[4],
                                                    lower_bounds_mask,
                                                    upper_bounds_mask,
                                                    new_axis,
                                                    shrink_axis,
                                                    ellipsis_mask);
            break;
        }
        case OP_TYPEID::DynSlice:
        {
            auto lower_bounds_mask = node_js.at("lower_bounds_mask").get<set<size_t>>();
            auto upper_bounds_mask = node_js.at("upper_bounds_mask").get<set<size_t>>();
            auto new_axis = node_js.at("new_axis").get<set<size_t>>();
            auto shrink_axis = node_js.at("shrink_axis").get<set<size_t>>();
            auto ellipsis_mask = node_js.at("ellipsis_mask").get<set<size_t>>();
            node = make_shared<op::DynSlice>(args[0],
                                             args[1],
                                             args[2],
                                             args[3],
                                             lower_bounds_mask,
                                             upper_bounds_mask,
                                             new_axis,
                                             shrink_axis,
                                             ellipsis_mask);
            break;
        }
        case OP_TYPEID::Elu:
        {
            auto alpha = node_js.at("alpha").get<double>();
            node = make_shared<op::Elu>(args[0], alpha);
            break;
        }
        case OP_TYPEID::EmbeddingLookup:
        {
            node = make_shared<op::EmbeddingLookup>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Equal:
        {
            node = make_shared<op::v0::Equal>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Erf:
        {
            node = make_shared<op::Erf>(args[0]);
            break;
        }
        case OP_TYPEID::Exp:
        {
            node = make_shared<op::Exp>(args[0]);
            break;
        }
        case OP_TYPEID::FakeQuantize:
        {
            size_t levels = node_js.at("levels").get<size_t>();
            node =
                make_shared<op::FakeQuantize>(args[0], args[1], args[2], args[3], args[4], levels);
            break;
        }
        case OP_TYPEID::Floor:
        {
            node = make_shared<op::Floor>(args[0]);
            break;
        }
        case OP_TYPEID::Gather:
        {
            auto axis = node_js.at("axis").get<size_t>();
            node = make_shared<op::v0::Gather>(args[0], args[1], axis);
            break;
        }
        case OP_TYPEID::GatherND:
        {
            node = make_shared<op::GatherND>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Gelu:
        {
            node = make_shared<op::Gelu>(args[0]);
            break;
        }
        case OP_TYPEID::Gemm:
        {
            auto alpha = node_js.at("alpha").get<double>();
            auto beta = node_js.at("beta").get<double>();
            auto transA = node_js.at("transA").get<bool>();
            auto transB = node_js.at("transB").get<bool>();
            node = make_shared<op::Gemm>(args[0], args[1], args[2], alpha, beta, transA, transB);
            break;
        }
        case OP_TYPEID::GetOutputElement:
        {
            node = make_shared<op::GetOutputElement>(
                static_cast<Output<Node>>(args[0]).get_node_shared_ptr(),
                node_js.at("n").get<size_t>());
            break;
        }
        case OP_TYPEID::Greater:
        {
            node = make_shared<op::v0::Greater>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::GreaterEq:
        {
            node = make_shared<op::v0::GreaterEq>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::GRN:
        {
            auto bias = node_js.at("bias").get<float>();
            node = make_shared<op::GRN>(args[0], bias);
            break;
        }
        case OP_TYPEID::GroupConvolution:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();
            op::PadType pad_type = read_pad_type(node_js);
            if (has_key(node_js, "groups"))
            {
                auto groups = node_js.at("groups").get<size_t>();
                node = make_shared<op::GroupConvolution>(args[0],
                                                         args[1],
                                                         window_movement_strides,
                                                         window_dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilation_strides,
                                                         groups,
                                                         pad_type);
            }
            else
            {
                node = make_shared<op::GroupConvolution>(args[0],
                                                         args[1],
                                                         window_movement_strides,
                                                         window_dilation_strides,
                                                         padding_below,
                                                         padding_above,
                                                         data_dilation_strides,
                                                         pad_type);
            }
            break;
        }
        case OP_TYPEID::GroupConvolutionBackpropData:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto groups = node_js.at("groups").get<size_t>();

            node = make_shared<op::GroupConvolutionBackpropData>(args[0],
                                                                 args[1],
                                                                 args[2],
                                                                 window_movement_strides,
                                                                 window_dilation_strides,
                                                                 padding_below,
                                                                 padding_above,
                                                                 groups);
            break;
        }
        case OP_TYPEID::HardSigmoid:
        {
            node = make_shared<op::HardSigmoid>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::LayerNorm:
        {
            auto keep_stats = node_js.at("keep_stats").get<bool>();
            auto use_affine = node_js.at("use_affine").get<bool>();
            auto epsilon = node_js.at("epsilon").get<double>();
            auto begin_norm_axis = node_js.at("begin_norm_axis").get<int64_t>();
            if (use_affine)
            {
                node = make_shared<op::LayerNorm>(
                    args[0], args[1], args[2], keep_stats, begin_norm_axis, epsilon);
            }
            else
            {
                node = make_shared<op::LayerNorm>(args[0], keep_stats, begin_norm_axis, epsilon);
            }
            break;
        }
        case OP_TYPEID::Less:
        {
            node = make_shared<op::v0::Less>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::LessEq:
        {
            node = make_shared<op::v0::LessEq>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Log:
        {
            node = make_shared<op::Log>(args[0]);
            break;
        }
        case OP_TYPEID::LRN:
        {
            auto alpha = node_js.at("alpha").get<double>();
            auto beta = node_js.at("beta").get<double>();
            auto bias = node_js.at("bias").get<double>();
            auto nsize = node_js.at("nsize").get<size_t>();
            node = make_shared<op::LRN>(args[0], args[1], alpha, beta, bias, nsize);
            break;
        }
        case OP_TYPEID::LSTMCell:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto weights_format = get_or_default<op::LSTMWeightsFormat>(
                node_js, "weights_format", op::LSTMWeightsFormat::IFCO);
            auto clip = node_js.at("clip").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activations_alpha = node_js.at("activations_alpha").get<vector<float>>();
            auto activations_beta = node_js.at("activations_beta").get<vector<float>>();
            auto input_forget = node_js.at("input_forget").get<bool>();
            switch (args.size())
            {
            case 7:
                node = make_shared<op::LSTMCell>(args[0],
                                                 args[1],
                                                 args[2],
                                                 args[3],
                                                 args[4],
                                                 args[5],
                                                 args[6],
                                                 hidden_size,
                                                 weights_format,
                                                 activations,
                                                 activations_alpha,
                                                 activations_beta,
                                                 clip,
                                                 input_forget);
                break;
            case 6:
                node = make_shared<op::LSTMCell>(args[0],
                                                 args[1],
                                                 args[2],
                                                 args[3],
                                                 args[4],
                                                 args[5],
                                                 hidden_size,
                                                 weights_format,
                                                 activations,
                                                 activations_alpha,
                                                 activations_beta,
                                                 clip,
                                                 input_forget);
                break;
            case 5:
                node = make_shared<op::LSTMCell>(args[0],
                                                 args[1],
                                                 args[2],
                                                 args[3],
                                                 args[4],
                                                 hidden_size,
                                                 weights_format,
                                                 activations,
                                                 activations_alpha,
                                                 activations_beta,
                                                 clip,
                                                 input_forget);
                break;
            default: throw runtime_error("LSTMCell constructor not supported in serializer");
            }
            break;
        }
        case OP_TYPEID::LSTMSequence:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto clip = node_js.at("clip_threshold").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activations_alpha = node_js.at("activations_alpha").get<vector<float>>();
            auto activations_beta = node_js.at("activations_beta").get<vector<float>>();
            auto input_forget = node_js.at("input_forget").get<bool>();
            auto direction = node_js.at("direction").get<op::LSTMSequence::direction>();
            auto weights_format = get_or_default<op::LSTMWeightsFormat>(
                node_js, "weights_format", op::LSTMWeightsFormat::IFCO);
            if (args.size() == 8)
            {
                node = make_shared<op::LSTMSequence>(args[0],
                                                     args[1],
                                                     args[2],
                                                     args[3],
                                                     args[4],
                                                     args[5],
                                                     args[6],
                                                     args[7],
                                                     hidden_size,
                                                     direction,
                                                     weights_format,
                                                     activations_alpha,
                                                     activations_beta,
                                                     activations,
                                                     clip,
                                                     input_forget);
            }
            else
            {
                node = make_shared<op::LSTMSequence>(args[0],
                                                     args[1],
                                                     args[2],
                                                     args[3],
                                                     args[4],
                                                     args[5],
                                                     args[6],
                                                     hidden_size,
                                                     direction,
                                                     weights_format,
                                                     activations_alpha,
                                                     activations_beta,
                                                     activations,
                                                     clip,
                                                     input_forget);
            }
            break;
        }
        case OP_TYPEID::MatMul:
        {
            bool transpose_a = node_js.at("transpose_a").get<bool>();
            bool transpose_b = node_js.at("transpose_b").get<bool>();
            node = make_shared<op::MatMul>(args[0], args[1], transpose_a, transpose_b);
            break;
        }
        case OP_TYPEID::Max:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::Max>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::MaxPool:
        {
            auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            // For backwards compatibility, both (but not just one) of the padding_ fields may
            // be omitted.
            auto padding_below_maybe = get_or_default(node_js, "padding_below", json{});
            auto padding_above_maybe = get_or_default(node_js, "padding_above", json{});
            op::PadType pad_type = read_pad_type(node_js);
            if (padding_below_maybe.empty() && !padding_above_maybe.empty())
            {
                throw runtime_error(
                    "MaxPool: padding_below is absent but padding_above is present");
            }
            else if (!padding_below_maybe.empty() && padding_above_maybe.empty())
            {
                throw runtime_error(
                    "MaxPool: padding_below is present but padding_above is absent");
            }
            else if (!padding_below_maybe.empty() && !padding_above_maybe.empty())
            {
                auto padding_below = padding_below_maybe.get<vector<size_t>>();
                auto padding_above = padding_above_maybe.get<vector<size_t>>();
                node = make_shared<op::v0::MaxPool>(args[0],
                                                    window_shape,
                                                    window_movement_strides,
                                                    padding_below,
                                                    padding_above,
                                                    pad_type);
            }
            else
            {
                node = make_shared<op::v0::MaxPool>(args[0], window_shape, window_movement_strides);
            }

            break;
        }
        case OP_TYPEID::Maximum:
        {
            node = make_shared<op::v0::Maximum>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Min:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::Min>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::Minimum:
        {
            node = make_shared<op::v0::Minimum>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Multiply:
        {
            node = make_shared<op::v0::Multiply>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::MVN:
        {
            auto normalize_variance = node_js.at("normalize_variance").get<bool>();
            AxisSet reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            auto eps = node_js.at("eps").get<double>();
            if (reduction_axes.size() > 0)
            {
                node = make_shared<op::MVN>(args[0], reduction_axes, normalize_variance, eps);
            }
            else
            {
                node = make_shared<op::MVN>(args[0], true, normalize_variance, eps);
            }
            break;
        }
        case OP_TYPEID::Negative:
        {
            node = make_shared<op::Negative>(args[0]);
            break;
        }
        case OP_TYPEID::NonZero_v3:
        {
            auto target_type = read_element_type(node_js.at("index_element_type"));
            node = make_shared<op::v3::NonZero>(args[0], target_type);
            break;
        }
        case OP_TYPEID::NormalizeL2:
        {
            float eps = node_js.at("eps").get<float>();
            auto eps_mode = node_js.at("eps_mode").get<op::EpsMode>();
            node = make_shared<op::NormalizeL2>(args[0], args[1], eps, eps_mode);
            break;
        }
        case OP_TYPEID::NotEqual:
        {
            node = make_shared<op::v0::NotEqual>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Not:
        {
            node = make_shared<op::Not>(args[0]);
            break;
        }
        case OP_TYPEID::OneHot:
        {
            if (op_version == 0)
            {
                auto shape = node_js.at("shape").get<vector<size_t>>();
                auto one_hot_axis = node_js.at("one_hot_axis").get<size_t>();
                node =
                    make_shared<op::v0::OneHot>(args[0], read_partial_shape(shape), one_hot_axis);
            }
            if (op_version == 1)
            {
                auto axis = node_js.at("axis").get<int64_t>();
                node = make_shared<op::v1::OneHot>(args[0], args[1], args[2], args[3], axis);
            }
            break;
        }
        case OP_TYPEID::Or:
        {
            node = make_shared<op::v0::Or>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Pad:
        {
            auto padding_below = node_js.at("padding_below").get<vector<ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<ptrdiff_t>>();

            // This is a legacy field whose functionality is no longer supported. The new
            // behavior is equivalent to interior padding of 0, so we will accept it under
            // those conditions.
            auto padding_interior = get_value<vector<size_t>>(node_js, "padding_interior");
            NGRAPH_CHECK(std::all_of(padding_interior.begin(),
                                     padding_interior.end(),
                                     [](size_t s) { return s == 0; }),
                         "Legacy padding_interior field must be zero everywhere.");

            auto pad_mode = read_pad_mode(node_js);

            node =
                make_shared<op::v0::Pad>(args[0], args[1], padding_below, padding_above, pad_mode);

            break;
        }
        case OP_TYPEID::Parameter:
        {
            auto type_node_js =
                has_key(node_js, "element_type") ? node_js : node_js.at("value_type");
            auto element_type = read_element_type(type_node_js.at("element_type"));
            auto shape = type_node_js.at("shape");
            auto cacheable = get_or_default<bool>(node_js, "cacheable", false);
            node = make_shared<op::Parameter>(element_type, read_partial_shape(shape), cacheable);
            break;
        }
        case OP_TYPEID::PartialSlice:
        {
            auto axes = node_js.at("axes").get<vector<size_t>>();
            auto lower_bounds = node_js.at("lower_bounds").get<vector<int64_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<int64_t>>();
            auto decrease_axes = node_js.at("decrease_axes").get<vector<size_t>>();
            node = make_shared<op::PartialSlice>(
                args[0], axes, lower_bounds, upper_bounds, decrease_axes);
            break;
        }
        case OP_TYPEID::Passthrough:
        {
            std::vector<json> outputs_js = node_js.at("output_shapes");
            std::vector<std::tuple<element::Type, PartialShape>> outputs;
            for (auto output_js : outputs_js)
            {
                outputs.emplace_back(read_element_type(output_js.at("element_type")),
                                     read_partial_shape(output_js.at("shape")));
            }
            node = make_shared<op::Passthrough>(node_js.at("logical_type"),
                                                node_js.at("language"),
                                                node_js.at("function"),
                                                static_cast<OutputVector>(args),
                                                std::move(outputs));
            break;
        }
        case OP_TYPEID::Power:
        {
            node = make_shared<op::v0::Power>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::PRelu:
        {
            node = make_shared<op::PRelu>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Product:
        {
            set<size_t> reduction_axes =
                get_or_default<set<size_t>>(node_js, "reduction_axes", set<size_t>());
            if (reduction_axes.empty())
            {
                node = make_shared<op::v0::Product>(args[0], args[1]);
            }
            else
            {
                node = make_shared<op::v0::Product>(args[0], reduction_axes);
            }
            break;
        }
        case OP_TYPEID::PSROIPooling: { break;
        }
        case OP_TYPEID::PriorBox: { break;
        }
        case OP_TYPEID::PriorBoxClustered: { break;
        }
        case OP_TYPEID::Proposal: { break;
        }

        case OP_TYPEID::Quantize:
        {
            auto type = read_element_type(node_js.at("type"));
            auto axes = deserialize_axis_set(node_js.at("axes"));
            auto round_mode = node_js.at("round_mode").get<op::Quantize::RoundMode>();
            node = make_shared<op::Quantize>(args[0], args[1], args[2], type, axes, round_mode);
            break;
        }
        case OP_TYPEID::QuantizedConvolutionBias: { break;
        }
        case OP_TYPEID::QuantizedConvolutionBiasAdd: { break;
        }
        case OP_TYPEID::QuantizedConvolutionBiasSignedAdd: { break;
        }
        case OP_TYPEID::QuantizedConvolutionRelu: { break;
        }
        case OP_TYPEID::QuantizedConvolution:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js["data_dilation_strides"];
            auto output_type = read_element_type(node_js.at("output_type"));
            auto input_axes = node_js.at("input_axes").get<set<size_t>>();
            auto filter_axes = node_js.at("filter_axes").get<set<size_t>>();
            auto output_axes = node_js.at("output_axes").get<set<size_t>>();
            node = make_shared<op::QuantizedConvolution>(
                args[0],
                args[1],
                window_movement_strides,
                window_dilation_strides,
                padding_below,
                padding_above,
                data_dilation_strides.get<std::vector<size_t>>(),
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
                args[7],
                output_type,
                input_axes,
                filter_axes,
                output_axes);

            break;
        }
        case OP_TYPEID::QuantizedDotBias: { break;
        }
        case OP_TYPEID::QuantizedDot:
        {
            size_t reduction_axes_count = node_js["reduction_axes_count"].get<size_t>();
            auto output_type = read_element_type(node_js.at("output_type"));
            auto input0_axes = node_js.at("input0_axes").get<set<size_t>>();
            auto input1_axes = node_js.at("input1_axes").get<set<size_t>>();
            auto output_axes = node_js.at("output_axes").get<set<size_t>>();

            node = make_shared<op::QuantizedDot>(args[0],
                                                 args[1],
                                                 reduction_axes_count,
                                                 args[2],
                                                 args[3],
                                                 args[4],
                                                 args[5],
                                                 args[6],
                                                 args[7],
                                                 output_type,
                                                 input0_axes,
                                                 input1_axes,
                                                 output_axes);

            break;
        }
        case OP_TYPEID::Recv:
        {
            auto src_id = node_js.at("source_id").get<size_t>();
            node = make_shared<op::Recv>(args[0], src_id);
            break;
        }
        case OP_TYPEID::Range:
        {
            node = make_shared<op::Range>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Relu:
        {
            node = make_shared<op::Relu>(args[0]);
            break;
        }
        case OP_TYPEID::ReplaceSlice:
        {
            auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
            auto strides = node_js.at("strides").get<vector<size_t>>();
            node = make_shared<op::ReplaceSlice>(
                args[0], args[1], lower_bounds, upper_bounds, strides);
            break;
        }
        case OP_TYPEID::Reshape:
        {
            auto input_order = node_js.at("input_order").get<vector<size_t>>();
            auto output_shape = node_js.at("output_shape").get<vector<size_t>>();
            node = make_shared<op::Reshape>(args[0], input_order, output_shape);
            break;
        }
        case OP_TYPEID::Result:
        {
            auto needs_default_layout =
                get_or_default<bool>(node_js, "needs_default_layout", false);
            node = make_shared<op::Result>(args[0], needs_default_layout);
            break;
        }
        case OP_TYPEID::Reverse:
        {
            const auto reversed_axes = deserialize_axis_set(node_js.at("reversed_axes"));
            node = make_shared<op::Reverse>(args[0], reversed_axes);
            break;
        }
        case OP_TYPEID::ReverseSequence:
        {
            auto batch_axis = node_js.at("batch_axis").get<int64_t>();
            auto sequence_axis = node_js.at("sequence_axis").get<int64_t>();
            node = make_shared<op::ReverseSequence>(args[0], args[1], batch_axis, sequence_axis);
            break;
        }
        case OP_TYPEID::RNNCell:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto clip = node_js.at("clip").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activation_alpha = node_js.at("activations_alpha").get<vector<float>>();
            auto activation_beta = node_js.at("activations_beta").get<vector<float>>();
            switch (args.size())
            {
            case 4:
                node = make_shared<op::RNNCell>(args[0],
                                                args[1],
                                                args[2],
                                                args[3],
                                                hidden_size,
                                                activations,
                                                activation_alpha,
                                                activation_beta,
                                                clip);
                break;
            case 5:
                node = make_shared<op::RNNCell>(args[0],
                                                args[1],
                                                args[2],
                                                args[3],
                                                args[4],
                                                hidden_size,
                                                activations,
                                                activation_alpha,
                                                activation_beta,
                                                clip);
                break;
            default: throw runtime_error("GRUCell constructor not supported in serializer");
            }
            break;
        }
        case OP_TYPEID::ROIPooling: { break;
        }
        case OP_TYPEID::RegionYolo: { break;
        }
        case OP_TYPEID::ReorgYolo:
        {
            break;
            const auto strides = node_js.at("strides").get<vector<size_t>>();
            node = make_shared<op::ReorgYolo>(args[0], strides);
            break;
        }
        case OP_TYPEID::Round:
        {
            node = make_shared<op::Round>(args[0]);
            break;
        }
        case OP_TYPEID::ScalarConstantLike:
        {
            double value = node_js.at("value").get<double>();
            node = make_shared<op::ScalarConstantLike>(args[0], value);
            break;
        }
        case OP_TYPEID::ScaleShift:
        {
            node = make_shared<op::ScaleShift>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::ScatterAdd:
        {
            node = make_shared<op::ScatterAdd>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::ScatterND:
        {
            node = make_shared<op::ScatterND>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::ScatterNDAdd:
        {
            node = make_shared<op::ScatterNDAdd>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Select:
        {
            node = make_shared<op::Select>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Stack:
        {
            auto axis = node_js.at("axis").get<size_t>();
            node = make_shared<op::Stack>(static_cast<OutputVector>(args), axis);
            break;
        }
        case OP_TYPEID::Selu:
        {
            node = make_shared<op::Selu>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Send:
        {
            auto dest_id = node_js.at("dest_id").get<size_t>();
            node = make_shared<op::Send>(args[0], dest_id);
            break;
        }
        case OP_TYPEID::ShapeOf:
        {
            node = make_shared<op::ShapeOf>(args[0]);
            break;
        }
        case OP_TYPEID::ShuffleChannels:
        {
            const auto axis = node_js.at("axis").get<size_t>();
            const auto groups = node_js.at("groups").get<size_t>();
            node = make_shared<op::ShuffleChannels>(args[0], axis, groups);
            break;
        }
        case OP_TYPEID::Sigmoid:
        {
            node = make_shared<op::Sigmoid>(args[0]);
            break;
        }
        case OP_TYPEID::Sign:
        {
            node = make_shared<op::Sign>(args[0]);
            break;
        }
        case OP_TYPEID::Sin:
        {
            node = make_shared<op::Sin>(args[0]);
            break;
        }
        case OP_TYPEID::Sinh:
        {
            node = make_shared<op::Sinh>(args[0]);
            break;
        }
        case OP_TYPEID::Slice:
        {
            auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
            auto strides = node_js.at("strides").get<vector<size_t>>();
            node = make_shared<op::Slice>(args[0], lower_bounds, upper_bounds, strides);
            break;
        }
        case OP_TYPEID::Softmax:
        {
            if (has_key(node_js, "softmax_axes"))
            {
                auto softmax_axes = deserialize_axis_set(node_js.at("softmax_axes"));
                node = make_shared<op::Softmax>(args[0], softmax_axes);
            }
            else
            {
                node = make_shared<op::Softmax>(args[0], args[1]);
            }

            break;
        }
        case OP_TYPEID::SoftmaxCrossEntropy:
        {
            auto soft_label = node_js.at("soft_label");
            auto ignore_index = node_js.at("ignore_index");
            node = make_shared<op::SoftmaxCrossEntropy>(args[0], args[1], soft_label, ignore_index);
            break;
        }
        case OP_TYPEID::SpaceToDepth:
        {
            auto block_size = node_js.at("block_size").get<size_t>();
            auto mode = node_js.at("mode").get<op::SpaceToDepth::SpaceToDepthMode>();
            node = make_shared<op::SpaceToDepth>(args[0], mode, block_size);
            break;
        }
        case OP_TYPEID::Split:
        {
            const auto splits = node_js.at("splits").get<vector<size_t>>();
            node = make_shared<op::Split>(args[0], args[1], splits);
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            node = make_shared<op::Sqrt>(args[0]);
            break;
        }
        case OP_TYPEID::SquaredDifference:
        {
            node = make_shared<op::SquaredDifference>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Squeeze:
        {
            node = make_shared<op::Squeeze>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Subtract:
        {
            node = make_shared<op::Subtract>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Sum:
        {
            set<size_t> reduction_axes =
                get_or_default<set<size_t>>(node_js, "reduction_axes", set<size_t>());
            if (reduction_axes.empty())
            {
                node = make_shared<op::v0::Sum>(args[0], args[1]);
            }
            else
            {
                node = make_shared<op::v0::Sum>(args[0], reduction_axes);
            }
            break;
        }
        case OP_TYPEID::Tan:
        {
            node = make_shared<op::Tan>(args[0]);
            break;
        }
        case OP_TYPEID::Tanh:
        {
            node = make_shared<op::Tanh>(args[0]);
            break;
        }
        case OP_TYPEID::TensorIterator:
        {
            auto ti = make_shared<op::TensorIterator>(args);
            json jbody = node_js["body"];
            // Serializer assumes inputs are available before users sp we
            // need to make sure the body nodes are all deserialized before
            // referencing them.
            json jbody_nodes = jbody["nodes"];
            NodeVector body_nodes;
            for (json jnode : jbody_nodes)
            {
                body_nodes.push_back(deserialize_node(jnode));
            }
            json jparams = jbody["parameters"];
            ParameterVector parameters;
            for (json jparam : jparams)
            {
                parameters.push_back(as_type_ptr<op::Parameter>(deserialize_node(jparam)));
            }
            json jresults = jbody["results"];
            ResultVector results;
            for (json jresult : jresults)
            {
                results.push_back(as_type_ptr<op::Result>(deserialize_node(jresult)));
            }
            ti->set_body(make_shared<op::TensorIterator::BodyLambda>(results, parameters));
            json jins = node_js["input_descriptions"];
            for (json jin : jins)
            {
                ti->get_input_descriptions().push_back(
                    deserialize_tensor_iterator_input_description(jin));
            }
            json jouts = node_js["output_descriptions"];
            for (json jout : jouts)
            {
                ti->get_output_descriptions().push_back(
                    deserialize_tensor_iterator_output_description(jout));
            }
            ti->set_output_size(ti->get_output_descriptions().size());

            node = ti;
            break;
        }

        case OP_TYPEID::Tile:
        {
            node = make_shared<op::Tile>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::TopK:
        {
            auto compute_max = node_js.at("compute_max").get<bool>();
            auto target_type = read_element_type(node_js.at("index_element_type"));
            op::TopKSortType sort =
                get_or_default<op::TopKSortType>(node_js, "sort", op::TopKSortType::SORT_VALUES);
            if (has_key(node_js, "top_k_axis"))
            {
                auto top_k_axis = node_js.at("top_k_axis").get<size_t>();
                if (has_key(node_js, "k"))
                {
                    auto k = node_js.at("k").get<size_t>();
                    node = make_shared<op::TopK>(
                        args[0], top_k_axis, target_type, k, compute_max, sort);
                }
                else
                {
                    node = make_shared<op::TopK>(
                        args[0], args[1], top_k_axis, target_type, compute_max, sort);
                }
            }
            else
            {
                node = make_shared<op::TopK>(
                    args[0], args[1], args[2], target_type, compute_max, sort);
            }
            break;
        }
        case OP_TYPEID::StopGradient:
        {
            node = make_shared<op::StopGradient>(args[0]);
            break;
        }
        case OP_TYPEID::Unsqueeze:
        {
            node = make_shared<op::Unsqueeze>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Xor:
        {
            node = make_shared<op::v0::Xor>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        default:
        {
            stringstream ss;
            ss << "unsupported op " << type_info.name << ":" << type_info.version;
            throw runtime_error(ss.str());
        }
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

        for (auto& control_dep : control_deps_inputs)
        {
            node->add_control_dependency(deserialize_node_reference(control_dep));
        }

        if (!friendly_name.empty())
        {
            node->set_friendly_name(friendly_name);
        }
        else
        {
            node->set_friendly_name(node_name);
        }
        if (ngraph::get_provenance_enabled())
        {
            if (has_key(node_js, "provenance_tags"))
            {
                const std::vector<json> prov_js = node_js.at("provenance_tags");
                for (auto prov_tag : prov_js)
                {
                    node->add_provenance_tag(prov_tag);
                }
            }
        }
        m_node_map[node_name] = node;
    }
    catch (exception& err)
    {
        NGRAPH_INFO << err.what();
        string node_name;
        auto it = node_js.find("name");
        if (it != node_js.end())
        {
            node_name = it->get<string>();
        }
        else
        {
            node_name = "UNKNOWN";
        }
        throw runtime_error("Error parsing json at node '" + node_name + "'");
    }
    return node;
}

json JSONSerializer::serialize_output(const Output<Node>& output)
{
    json result;
    auto index = output.get_index();
    json json_node_reference = output.get_node()->get_name();
    if (index == 0)
    {
        result = json_node_reference;
    }
    else
    {
        result["node"] = json_node_reference;
        result["index"] = index;
    }
    return result;
}

json JSONSerializer::serialize_output_vector(const OutputVector& output_vector)
{
    json result;
    for (const Output<Node>& output : output_vector)
    {
        result.push_back(serialize_output(output));
    }
    return result;
}

json JSONSerializer::serialize_node(const Node& n)
{
    const NodeTypeInfo& type_info = n.get_type_info();
    json jtype_info = json::object();
    jtype_info["name"] = type_info.name;
    jtype_info["version"] = type_info.version;
    json node;
    node["type_info"] = jtype_info;
    node["name"] = n.get_name();
    auto op_version = n.get_version();
    node["op_version"] = op_version;

    if (n.get_name() != n.get_friendly_name())
    {
        node["friendly_name"] = n.get_friendly_name();
    }
    node["op"] = type_info.name;
    // TODO Multiple outputs
    json inputs = json::array();
    json control_deps = json::array();
    json outputs = json::array();

    for (auto& input : n.inputs())
    {
        inputs.push_back(serialize_output(input.get_source_output()));
    }
    for (auto cdep : n.get_control_dependencies())
    {
        control_deps.push_back(cdep->get_name());
    }
    for (auto& output : n.outputs())
    {
        outputs.push_back(output.get_tensor().get_name());
    }

    if (!inputs.empty())
    {
        node["inputs"] = inputs;
    }
    if (!control_deps.empty())
    {
        node["control_deps"] = control_deps;
    }
    if (!outputs.empty())
    {
        node["outputs"] = outputs;
    }

    if (s_serialize_output_shapes_enabled)
    {
        json output_shapes = json::array();
        for (size_t i = 0; i < n.get_output_size(); ++i)
        {
            output_shapes.push_back(n.get_output_shape(i));
        }
        node["output_shapes"] = output_shapes;
    }
    if (ngraph::get_provenance_enabled())
    {
        json provenance_tags = json::array();
        for (auto prov_tag : n.get_provenance_tags())
        {
            provenance_tags.push_back(prov_tag);
        }
        node["provenance_tags"] = provenance_tags;
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
// #pragma GCC diagnostic error "-Wswitch-enum"
// #pragma GCC diagnostic error "-Wimplicit-fallthrough"
#endif
    switch (get_typeid(type_info))
    {
    case OP_TYPEID::Abs: { break;
    }
    case OP_TYPEID::Acos: { break;
    }
    case OP_TYPEID::Add:
    {
        const op::util::BinaryElementwiseArithmetic* tmp = nullptr;
        tmp = static_cast<const op::v0::Add*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::ArgMin:
    {
        auto tmp = static_cast<const op::ArgMin*>(&n);
        node["axis"] = tmp->get_reduction_axis();
        node["index_element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::ArgMax:
    {
        auto tmp = static_cast<const op::ArgMax*>(&n);
        node["axis"] = tmp->get_reduction_axis();
        node["index_element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::All:
    {
        auto tmp = static_cast<const op::All*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::AllReduce: { break;
    }
    case OP_TYPEID::And:
    {
        auto tmp = static_cast<const op::And*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Any:
    {
        auto tmp = static_cast<const op::Any*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::Asin: { break;
    }
    case OP_TYPEID::Atan: { break;
    }
    case OP_TYPEID::BatchNormInference:
    {
        auto tmp = static_cast<const op::BatchNormInference*>(&n);
        node["eps"] = tmp->get_eps_value();
        break;
    }
    case OP_TYPEID::Broadcast:
    {
        auto tmp = dynamic_cast<const op::v0::Broadcast*>(&n);
        node["axes"] = serialize_axis_set(tmp->get_broadcast_axes());
        node["shape"] = tmp->get_broadcast_shape();
        break;
    }
    case OP_TYPEID::BroadcastDistributed: { break;
    }
    case OP_TYPEID::BroadcastLike:
    {
        auto tmp = static_cast<const op::BroadcastLike*>(&n);
        node["initial_axes"] = serialize_axis_set(tmp->get_initial_broadcast_axes());
        break;
    }
    case OP_TYPEID::Ceiling: { break;
    }
    case OP_TYPEID::Clamp:
    {
        auto tmp = static_cast<const op::Clamp*>(&n);
        node["min"] = tmp->get_min();
        node["max"] = tmp->get_max();
        break;
    }
    case OP_TYPEID::Concat:
    {
        auto tmp = static_cast<const op::Concat*>(&n);
        node["axis"] = tmp->get_concatenation_axis();
        break;
    }
    case OP_TYPEID::Constant:
    {
        auto tmp = static_cast<const op::Constant*>(&n);
        if (tmp->get_all_data_elements_bitwise_identical() && shape_size(tmp->get_shape()) > 0)
        {
            vector<string> vs;
            vs.push_back(tmp->convert_value_to_string(0));
            node["value"] = vs;
        }
        else
        {
            node["value"] = tmp->get_value_strings();
        }
        node["shape"] = tmp->get_shape();
        node["element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::Convert:
    {
        auto tmp = static_cast<const op::Convert*>(&n);
        node["target_type"] = write_element_type(tmp->get_convert_element_type());
        break;
    }
    case OP_TYPEID::Convolution:
    {
        auto tmp = static_cast<const op::v0::Convolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        node["pad_type"] = tmp->get_pad_type();
        break;
    }
    case OP_TYPEID::ConvolutionBackpropData:
    {
        auto tmp = static_cast<const op::v0::ConvolutionBackpropData*>(&n);
        node["data_batch_shape"] = tmp->get_data_batch_shape();
        node["window_movement_strides_forward"] = tmp->get_window_movement_strides_forward();
        node["window_dilation_strides_forward"] = tmp->get_window_dilation_strides_forward();
        node["padding_below_forward"] = tmp->get_padding_below_forward();
        node["padding_above_forward"] = tmp->get_padding_above_forward();
        node["data_dilation_strides_forward"] = tmp->get_data_dilation_strides_forward();
        break;
    }
    case OP_TYPEID::ConvolutionBias:
    {
        auto tmp = static_cast<const op::ConvolutionBias*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        break;
    }
    case OP_TYPEID::ConvolutionBiasAdd:
    {
        auto tmp = static_cast<const op::ConvolutionBiasAdd*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        break;
    }
    case OP_TYPEID::Cos: { break;
    }
    case OP_TYPEID::Cosh: { break;
    }
    case OP_TYPEID::CumSum:
    {
        auto tmp = static_cast<const op::CumSum*>(&n);
        node["exclusive"] = tmp->is_exclusive();
        node["reverse"] = tmp->is_reverse();
        break;
    }
    case OP_TYPEID::CrossEntropy:
    {
        auto tmp = static_cast<const op::CrossEntropy*>(&n);
        node["soft_label"] = tmp->get_soft_label();
        node["ignore_index"] = tmp->get_ignore_index();
        break;
    }
    case OP_TYPEID::CropAndResize:
    {
        auto tmp = static_cast<const op::CropAndResize*>(&n);
        node["resize_method"] = as_string(tmp->get_resize_method());
        node["extrapolation_value"] = tmp->get_extrapolation_value();
        break;
    }
    case OP_TYPEID::CTCGreedyDecoder: { break;
    }
    case OP_TYPEID::DetectionOutput: { break;
    }
    case OP_TYPEID::PSROIPooling: { break;
    }
    case OP_TYPEID::PriorBox: { break;
    }
    case OP_TYPEID::PriorBoxClustered: { break;
    }
    case OP_TYPEID::Proposal: { break;
    }
    case OP_TYPEID::ROIPooling: { break;
    }
    case OP_TYPEID::RegionYolo: { break;
    }
    case OP_TYPEID::ReorgYolo:
    {
        auto tmp = static_cast<const op::ReorgYolo*>(&n);
        node["strides"] = tmp->get_strides();
        break;
    }
    case OP_TYPEID::Round: { break;
    }
    case OP_TYPEID::Dequantize:
    {
        auto tmp = static_cast<const op::Dequantize*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["axes"] = serialize_axis_set(tmp->get_axes());
        break;
    }
    case OP_TYPEID::DepthToSpace:
    {
        auto tmp = static_cast<const op::DepthToSpace*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["mode"] = tmp->get_mode();
        node["block_size"] = tmp->get_block_size();
        break;
    }
    case OP_TYPEID::Divide:
    {
        const op::util::BinaryElementwiseArithmetic* bea_node = nullptr;
        auto tmp = static_cast<const op::v0::Divide*>(&n);
        bea_node = tmp;
        node["pythondiv"] = tmp->is_pythondiv();
        if (bea_node != nullptr && bea_node->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(bea_node->get_autob());
        }
        break;
    }
    case OP_TYPEID::Dot:
    {
        auto tmp = static_cast<const op::Dot*>(&n);
        node["reduction_axes_count"] = tmp->get_reduction_axes_count();
        break;
    }
    case OP_TYPEID::DynBroadcast: { break;
    }
    case OP_TYPEID::DynPad: { break;
    }
    case OP_TYPEID::DynReplaceSlice:
    {
        auto tmp = static_cast<const op::DynReplaceSlice*>(&n);
        node["lower_bounds_mask"] = tmp->get_lower_bounds_mask();
        node["upper_bounds_mask"] = tmp->get_upper_bounds_mask();
        node["new_axis"] = tmp->get_new_axis();
        node["shrink_axis"] = tmp->get_shrink_axis();
        node["ellipsis_mask"] = tmp->get_ellipsis_mask();
        break;
    }
    case OP_TYPEID::DynSlice:
    {
        auto tmp = static_cast<const op::DynSlice*>(&n);
        node["lower_bounds_mask"] = tmp->get_lower_bounds_mask();
        node["upper_bounds_mask"] = tmp->get_upper_bounds_mask();
        node["new_axis"] = tmp->get_new_axis();
        node["shrink_axis"] = tmp->get_shrink_axis();
        node["ellipsis_mask"] = tmp->get_ellipsis_mask();
        break;
    }
    case OP_TYPEID::Elu:
    {
        auto tmp = static_cast<const op::Elu*>(&n);
        node["alpha"] = tmp->get_alpha();
        break;
    }
    case OP_TYPEID::EmbeddingLookup: { break;
    }
    case OP_TYPEID::Equal:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v0::Equal*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Erf: { break;
    }
    case OP_TYPEID::Exp: { break;
    }
    case OP_TYPEID::FakeQuantize:
    {
        auto tmp = static_cast<const op::FakeQuantize*>(&n);
        node["levels"] = tmp->get_levels();
        break;
    }
    case OP_TYPEID::Floor: { break;
    }
    case OP_TYPEID::Gather:
    {
        auto tmp = static_cast<const op::v0::Gather*>(&n);
        node["axis"] = tmp->get_axis();
        break;
    }
    case OP_TYPEID::GatherND: { break;
    }
    case OP_TYPEID::GetOutputElement:
    {
        auto tmp = static_cast<const op::GetOutputElement*>(&n);
        node["n"] = tmp->get_n();
        break;
    }
    case OP_TYPEID::Gelu: { break;
    }
    case OP_TYPEID::Gemm:
    {
        auto tmp = static_cast<const op::Gemm*>(&n);
        node["alpha"] = tmp->get_alpha();
        node["beta"] = tmp->get_beta();
        node["transA"] = tmp->get_transA();
        node["transB"] = tmp->get_transB();
        break;
    }
    case OP_TYPEID::Greater:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v0::Greater*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::GreaterEq:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v0::GreaterEq*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::GRN:
    {
        auto tmp = static_cast<const op::GRN*>(&n);
        node["bias"] = tmp->get_bias();
        break;
    }
    case OP_TYPEID::GroupConvolution:
    {
        auto tmp = static_cast<const op::GroupConvolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        if (!tmp->has_groups_in_filters())
        {
            node["groups"] = tmp->get_groups();
        }
        node["pad_type"] = tmp->get_pad_type();
        break;
    }
    case OP_TYPEID::GroupConvolutionBackpropData:
    {
        auto tmp = static_cast<const op::GroupConvolutionBackpropData*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["groups"] = tmp->get_groups();
        break;
    }
    case OP_TYPEID::HardSigmoid: { break;
    }
    case OP_TYPEID::LayerNorm:
    {
        auto tmp = static_cast<const op::LayerNorm*>(&n);
        node["keep_stats"] = tmp->get_keep_stats();
        node["use_affine"] = tmp->get_use_affine();
        node["epsilon"] = tmp->get_epsilon();
        node["begin_norm_axis"] = tmp->get_begin_norm_axis();
        break;
    }
    case OP_TYPEID::Less:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v0::Less*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::LessEq:
    {
        auto tmp = static_cast<const op::v0::LessEq*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Log: { break;
    }
    case OP_TYPEID::LRN:
    {
        auto tmp = static_cast<const op::LRN*>(&n);
        node["alpha"] = tmp->get_alpha();
        node["beta"] = tmp->get_beta();
        node["bias"] = tmp->get_bias();
        node["nsize"] = tmp->get_nsize();
        break;
    }
    case OP_TYPEID::LSTMCell:
    {
        auto tmp = static_cast<const op::LSTMCell*>(&n);
        node["hidden_size"] = tmp->get_hidden_size();
        node["weights_format"] = tmp->get_weights_format();
        node["clip"] = tmp->get_clip();
        node["activations"] = tmp->get_activations();
        node["activations_alpha"] = tmp->get_activations_alpha();
        node["activations_beta"] = tmp->get_activations_beta();
        node["input_forget"] = tmp->get_input_forget();
        break;
    }
    case OP_TYPEID::LSTMSequence:
    {
        auto tmp = dynamic_cast<const op::LSTMSequence*>(&n);
        node["direction"] = tmp->get_direction();
        node["hidden_size"] = tmp->get_hidden_size();
        node["weights_format"] = tmp->get_weights_format();
        node["clip_threshold"] = tmp->get_clip_threshold();
        node["activations"] = tmp->get_activations();
        node["activations_alpha"] = tmp->get_activations_alpha();
        node["activations_beta"] = tmp->get_activations_beta();
        node["input_forget"] = tmp->get_input_forget();
        break;
    }
    case OP_TYPEID::MatMul:
    {
        auto tmp = static_cast<const op::MatMul*>(&n);
        node["transpose_a"] = tmp->get_transpose_a();
        node["transpose_b"] = tmp->get_transpose_b();
        break;
    }
    case OP_TYPEID::Max:
    {
        auto tmp = static_cast<const op::Max*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::MaxPool:
    {
        auto tmp = static_cast<const op::v0::MaxPool*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["pad_type"] = tmp->get_pad_type();
        break;
    }
    case OP_TYPEID::Maximum:
    {
        const op::util::BinaryElementwiseArithmetic* tmp = nullptr;
        tmp = static_cast<const op::v0::Maximum*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Min:
    {
        auto tmp = static_cast<const op::Min*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::Minimum:
    {
        const op::util::BinaryElementwiseArithmetic* tmp = nullptr;
        tmp = static_cast<const op::v0::Minimum*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Multiply:
    {
        const op::util::BinaryElementwiseArithmetic* tmp = nullptr;
        tmp = static_cast<const op::v0::Multiply*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::MVN:
    {
        auto tmp = static_cast<const op::MVN*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        node["normalize_variance"] = tmp->get_normalize_variance();
        node["eps"] = tmp->get_eps();
        break;
    }
    case OP_TYPEID::Negative: { break;
    }
    case OP_TYPEID::NormalizeL2:
    {
        auto tmp = static_cast<const op::NormalizeL2*>(&n);
        node["eps"] = tmp->get_eps();
        node["eps_mode"] = tmp->get_eps_mode();
        break;
    }
    case OP_TYPEID::NotEqual:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v0::NotEqual*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Not: { break;
    }
    case OP_TYPEID::OneHot:
    {
        if (op_version == 0)
        {
            auto tmp = static_cast<const op::v0::OneHot*>(&n);
            node["shape"] = write_partial_shape(tmp->get_output_partial_shape(0));
            node["one_hot_axis"] = tmp->get_one_hot_axis();
        }
        if (op_version == 1)
        {
            auto tmp = static_cast<const op::v1::OneHot*>(&n);
            node["axis"] = tmp->get_axis();
        }
        break;
    }
    case OP_TYPEID::Or:
    {
        auto tmp = static_cast<const op::v0::Or*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Pad:
    {
        auto tmp = static_cast<const op::v0::Pad*>(&n);
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["pad_mode"] = tmp->get_pad_mode();
        break;
    }
    case OP_TYPEID::Parameter:
    {
        auto tmp = static_cast<const op::Parameter*>(&n);
        node["shape"] = write_partial_shape(tmp->get_output_partial_shape(0));
        node["cacheable"] = tmp->get_cacheable();
        node["element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::PartialSlice:
    {
        auto tmp = dynamic_cast<const op::PartialSlice*>(&n);
        node["axes"] = tmp->get_axes();
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["decrease_axes"] = tmp->get_decrease_axes();
        break;
    }
    case OP_TYPEID::Passthrough:
    {
        auto tmp = static_cast<const op::Passthrough*>(&n);
        node["logical_type"] = tmp->logical_type();
        node["language"] = tmp->language();
        node["function"] = tmp->function();
        std::vector<json> outputs_js;
        for (const auto& output_shape : tmp->output_shapes())
        {
            json output_js;
            output_js["element_type"] = write_element_type(std::get<0>(output_shape));
            output_js["shape"] = write_partial_shape(std::get<1>(output_shape));
            outputs_js.emplace_back(std::move(output_js));
        }
        node["output_shapes"] = std::move(outputs_js);
        break;
    }
    case OP_TYPEID::PRelu: { break;
    }
    case OP_TYPEID::Product:
    {
        auto tmp = static_cast<const op::Product*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::Power:
    {
        const op::util::BinaryElementwiseArithmetic* tmp = nullptr;
        tmp = static_cast<const op::v0::Power*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Quantize:
    {
        auto tmp = static_cast<const op::Quantize*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["axes"] = serialize_axis_set(tmp->get_axes());
        node["round_mode"] = tmp->get_round_mode();
        break;
    }
    case OP_TYPEID::QuantizedConvolutionBias: { break;
    }
    case OP_TYPEID::QuantizedConvolutionBiasAdd: { break;
    }
    case OP_TYPEID::QuantizedConvolutionBiasSignedAdd: { break;
    }
    case OP_TYPEID::QuantizedConvolutionRelu: { break;
    }
    case OP_TYPEID::QuantizedConvolution:
    {
        auto tmp = static_cast<const op::QuantizedConvolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        node["output_type"] = write_element_type(tmp->get_element_type());
        node["input_axes"] = tmp->get_input_axes();
        node["filter_axes"] = tmp->get_filter_axes();
        node["output_axes"] = tmp->get_output_axes();
        break;
    }
    case OP_TYPEID::QuantizedDotBias: { break;
    }
    case OP_TYPEID::QuantizedDot:
    {
        auto tmp = static_cast<const op::QuantizedDot*>(&n);
        node["reduction_axes_count"] = tmp->get_reduction_axes_count();
        node["output_type"] = write_element_type(tmp->get_element_type());
        node["input0_axes"] = tmp->get_input0_axes();
        node["input1_axes"] = tmp->get_input1_axes();
        node["output_axes"] = tmp->get_output_axes();
        break;
    }
    case OP_TYPEID::Range: { break;
    }
    case OP_TYPEID::Recv:
    {
        auto tmp = static_cast<const op::Recv*>(&n);
        node["source_id"] = tmp->get_src_id();
        break;
    }
    case OP_TYPEID::Relu: { break;
    }
    case OP_TYPEID::ReplaceSlice:
    {
        auto tmp = static_cast<const op::ReplaceSlice*>(&n);
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["strides"] = tmp->get_strides();
        break;
    }
    case OP_TYPEID::Reshape:
    {
        auto tmp = static_cast<const op::Reshape*>(&n);
        node["input_order"] = tmp->get_input_order();
        node["output_shape"] = tmp->get_output_shape(0);
        break;
    }
    case OP_TYPEID::Result:
    {
        auto tmp = static_cast<const op::Result*>(&n);
        node["needs_default_layout"] = tmp->needs_default_layout();
        break;
    }
    case OP_TYPEID::Reverse:
    {
        const auto tmp = static_cast<const op::Reverse*>(&n);
        node["reversed_axes"] = serialize_axis_set(tmp->get_reversed_axes());
        break;
    }
    case OP_TYPEID::ReverseSequence:
    {
        auto tmp = static_cast<const op::ReverseSequence*>(&n);
        node["batch_axis"] = tmp->get_origin_batch_axis();
        node["sequence_axis"] = tmp->get_origin_sequence_axis();
        break;
    }
    case OP_TYPEID::RNNCell:
    {
        auto tmp = static_cast<const op::RNNCell*>(&n);
        node["hidden_size"] = tmp->get_hidden_size();
        node["clip"] = tmp->get_clip();
        node["activations"] = tmp->get_activations();
        node["activations_alpha"] = tmp->get_activations_alpha();
        node["activations_beta"] = tmp->get_activations_beta();
        break;
    }
    case OP_TYPEID::ScalarConstantLike:
    {
        auto tmp = static_cast<const op::ScalarConstantLike*>(&n);
        auto constant = tmp->as_constant();
        char* p_end;
        node["value"] = strtod(constant->get_value_strings()[0].c_str(), &p_end);
        break;
    }
    case OP_TYPEID::ScaleShift: { break;
    }
    case OP_TYPEID::ScatterAdd: { break;
    }
    case OP_TYPEID::ScatterND: { break;
    }
    case OP_TYPEID::ScatterNDAdd: { break;
    }
    case OP_TYPEID::Select: { break;
    }
    case OP_TYPEID::Selu: { break;
    }
    case OP_TYPEID::Send:
    {
        auto tmp = static_cast<const op::Send*>(&n);
        node["dest_id"] = tmp->get_dest_id();
        break;
    }
    case OP_TYPEID::ShapeOf: { break;
    }
    case OP_TYPEID::ShuffleChannels:
    {
        const auto tmp = static_cast<const op::ShuffleChannels*>(&n);
        node["axis"] = tmp->get_axis();
        node["groups"] = tmp->get_group();
        break;
    }
    case OP_TYPEID::Sigmoid: { break;
    }
    case OP_TYPEID::Sign: { break;
    }
    case OP_TYPEID::Sin: { break;
    }
    case OP_TYPEID::Sinh: { break;
    }
    case OP_TYPEID::Slice:
    {
        auto tmp = static_cast<const op::Slice*>(&n);
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["strides"] = tmp->get_strides();
        break;
    }
    case OP_TYPEID::SpaceToDepth:
    {
        auto tmp = static_cast<const op::SpaceToDepth*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["mode"] = tmp->get_mode();
        node["block_size"] = tmp->get_block_size();
        break;
    }
    case OP_TYPEID::Split:
    {
        const auto tmp = static_cast<const op::Split*>(&n);
        node["splits"] = tmp->get_splits();
        break;
    }
    case OP_TYPEID::Sqrt: { break;
    }
    case OP_TYPEID::SquaredDifference:
    {
        auto tmp = static_cast<const op::SquaredDifference*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Squeeze: { break;
    }
    case OP_TYPEID::StopGradient: { break;
    }
    case OP_TYPEID::Subtract:
    {
        auto tmp = static_cast<const op::Subtract*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Sum:
    {
        auto tmp = static_cast<const op::Sum*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::Stack:
    {
        auto tmp = static_cast<const op::Stack*>(&n);
        node["axis"] = tmp->get_axis();
        break;
    }
    case OP_TYPEID::Softmax: { break;
    }
    case OP_TYPEID::SoftmaxCrossEntropy:
    {
        auto tmp = static_cast<const op::SoftmaxCrossEntropy*>(&n);
        node["soft_label"] = tmp->get_soft_label();
        node["ignore_index"] = tmp->get_ignore_index();
        break;
    }
    case OP_TYPEID::Tan: { break;
    }
    case OP_TYPEID::Tanh: { break;
    }
    case OP_TYPEID::TensorIterator:
    {
        auto tmp = static_cast<const op::TensorIterator*>(&n);
        json body = json::object();
        {
            auto& body_results = tmp->get_body()->get_results();
            // Serializer assumes node inputs are already serialized, so
            // we need to capture the body-referenced nodes here.
            json body_nodes = json::array();
            for (auto n : topological_sort(body_results))
            {
                body_nodes.push_back(serialize_node(*n));
            }
            body["nodes"] = body_nodes;
            json params = json::array();
            for (auto param : tmp->get_body()->get_parameters())
            {
                params.push_back(serialize_node(*param));
            }
            body["parameters"] = params;
            json results = json::array();
            for (auto result : body_results)
            {
                results.push_back(serialize_node(*result));
            }
            body["results"] = results;
        }
        node["body"] = body;
        json ins = json::array();
        for (auto in : tmp->get_input_descriptions())
        {
            ins.push_back(serialize_tensor_iterator_input_description(in));
        }
        node["input_descriptions"] = ins;
        json outs = json::array();
        for (auto out : tmp->get_output_descriptions())
        {
            outs.push_back(serialize_tensor_iterator_output_description(out));
        }
        node["output_descriptions"] = outs;
        break;
    }
    case OP_TYPEID::Tile: { break;
    }
    case OP_TYPEID::TopK:
    {
        const auto tmp = static_cast<const op::TopK*>(&n);
        node["index_element_type"] = write_element_type(tmp->get_index_element_type());
        node["compute_max"] = tmp->get_compute_max();
        node["sort"] = tmp->get_sort();
        switch (tmp->inputs().size())
        {
        case 1:
            node["k"] = tmp->get_k();
            node["top_k_axis"] = tmp->get_top_k_axis();
            break;
        case 2: node["top_k_axis"] = tmp->get_top_k_axis(); break;
        case 3: break;
        default: throw runtime_error("TopK constructor not supported in serializer");
        }
        break;
    }
    case OP_TYPEID::Unsqueeze: { break;
    }
    case OP_TYPEID::Xor:
    {
        auto tmp = static_cast<const op::v0::Xor*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::UnknownOp: { break;
    }
    default:
    {
        auto& factory_registry = FactoryRegistry<Node>::get();
        if (factory_registry.has_factory(type_info))
        {
            node["attribute_visitor"] = true;
            JSONAttributeSerializer visitor(node);
            if (!const_cast<Node&>(n).visit_attributes(visitor))
            {
                NGRAPH_ERR << "Cannot serialize: " << node;
            }
            return node;
        }
    }
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic pop
#endif
    return node;
}
