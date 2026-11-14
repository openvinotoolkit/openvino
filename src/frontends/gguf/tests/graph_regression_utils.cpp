// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "graph_regression_utils.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>

#include "builder/api/metadata_store.hpp"
#include "builder/arch_registry.hpp"
#include "builder/gguf_builder.hpp"
#include "builder/gguf_builder_decoder.hpp"
#include "builder/gguf_graph.hpp"
#include "helper_ops/gguf_constant.hpp"
#include "model_adapter_attribute.hpp"
#include "op_table.hpp"
#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "pass/materialize_weights.hpp"
#include "quant/weights.hpp"

namespace ov_gguf_test {
namespace {
using Json = nlohmann::json;
using Materializer = ov::frontend::gguf::pass::MaterializeGGUFWeights;

class DescriptorTensor : public ov::ITensor {
public:
    DescriptorTensor(ov::element::Type type, ov::Shape shape, std::string name)
        : m_type(type),
          m_shape(std::move(shape)),
          m_name(std::move(name)),
          m_strides(m_shape.size()) {
        OPENVINO_ASSERT(type != ov::element::dynamic && type != ov::element::string,
                        "Invalid weight descriptor type: ",
                        m_name);
        size_t stride = type.size();
        for (size_t i = m_shape.size(); i-- > 0;) {
            m_strides[i] = stride;
            OPENVINO_ASSERT(m_shape[i] && stride <= std::numeric_limits<size_t>::max() / m_shape[i],
                            "Invalid descriptor size: ",
                            m_name);
            stride *= m_shape[i];
        }
    }

    void set_shape(ov::Shape shape) override {
        OPENVINO_ASSERT(shape == m_shape, "Cannot reshape descriptor storage: ", m_name);
    }
    const ov::element::Type& get_element_type() const override {
        return m_type;
    }
    const ov::Shape& get_shape() const override {
        return m_shape;
    }
    const ov::Strides& get_strides() const override {
        OPENVINO_ASSERT(m_type.bitwidth() >= 8, "Sub-byte descriptor storage has no byte strides: ", m_name);
        return m_strides;
    }
    void* data() override {
        OPENVINO_THROW("Descriptor-only weight payload accessed: ", m_name);
    }
    const void* data() const override {
        OPENVINO_THROW("Descriptor-only weight payload accessed: ", m_name);
    }
    void* data_rw() override {
        return data();
    }
    void* data(const ov::element::Type&) override {
        return data();
    }
    const void* data(const ov::element::Type&) const override {
        return data();
    }
    void* data_rw(const ov::element::Type&) override {
        return data();
    }

private:
    ov::element::Type m_type;
    ov::Shape m_shape;
    std::string m_name;
    ov::Strides m_strides;
};

template <typename T>
void fill_metadata(ov::Tensor& tensor, const Json& values) {
    auto* data = tensor.data<T>();
    for (size_t i = 0; i < values.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (values.at(i).is_object()) {
                const auto tag = values.at(i).at("nonfinite").get<std::string>();
                OPENVINO_ASSERT(values.at(i).size() == 1 && (tag == "nan" || tag == "+inf" || tag == "-inf"),
                                "Invalid nonfinite metadata tag");
                data[i] = tag == "nan"    ? std::numeric_limits<T>::quiet_NaN()
                          : tag == "+inf" ? std::numeric_limits<T>::infinity()
                                          : -std::numeric_limits<T>::infinity();
                continue;
            }
        }
        data[i] = values.at(i).get<T>();
    }
}

ov::frontend::gguf::GGUFMetaData metadata_value(const Json& entry) {
    const auto type = entry.at("type").get<int>();
    const auto element = type == 9 ? entry.at("element_type").get<int>() : type;
    const auto values = type == 9 ? entry.at("value") : Json::array({entry.at("value")});
    if (element == 8) {
        if (type == 9)
            return values.get<std::vector<std::string>>();
        return entry.at("value").get<std::string>();
    }
    static const std::map<int, ov::element::Type> types{{0, ov::element::u8},
                                                        {1, ov::element::i8},
                                                        {2, ov::element::u16},
                                                        {3, ov::element::i16},
                                                        {4, ov::element::u32},
                                                        {5, ov::element::i32},
                                                        {6, ov::element::f32},
                                                        {7, ov::element::boolean},
                                                        {10, ov::element::u64},
                                                        {11, ov::element::i64},
                                                        {12, ov::element::f64}};
    OPENVINO_ASSERT(types.count(element), "Invalid metadata element type: ", element);
    ov::Tensor tensor(types.at(element), type == 9 ? ov::Shape{values.size()} : ov::Shape{});
    switch (element) {
    case 0:
        fill_metadata<uint8_t>(tensor, values);
        break;
    case 1:
        fill_metadata<int8_t>(tensor, values);
        break;
    case 2:
        fill_metadata<uint16_t>(tensor, values);
        break;
    case 3:
        fill_metadata<int16_t>(tensor, values);
        break;
    case 4:
        fill_metadata<uint32_t>(tensor, values);
        break;
    case 5:
        fill_metadata<int32_t>(tensor, values);
        break;
    case 6:
        fill_metadata<float>(tensor, values);
        break;
    case 7:
        for (size_t i = 0; i < values.size(); ++i)
            tensor.data<uint8_t>()[i] = values.at(i).get<bool>();
        break;
    case 10:
        fill_metadata<uint64_t>(tensor, values);
        break;
    case 11:
        fill_metadata<int64_t>(tensor, values);
        break;
    case 12:
        fill_metadata<double>(tensor, values);
        break;
    default:
        OPENVINO_THROW("Invalid metadata type");
    }
    return tensor;
}

ov::Shape recipe_shape(const Json& value, const std::string& name) {
    OPENVINO_ASSERT(value.is_array() && !value.empty() && value.size() <= 4, "Invalid descriptor rank: ", name);
    for (const auto& dimension : value)
        OPENVINO_ASSERT(
            dimension.is_number_integer() && dimension > 0 && dimension <= std::numeric_limits<int64_t>::max(),
            "Invalid descriptor dimension: ",
            name);
    return ov::Shape(value.get<std::vector<size_t>>());
}

std::shared_ptr<ov::Model> convert_recipe(const Json& recipe) {
    using namespace ov::frontend::gguf;
    OPENVINO_ASSERT(recipe.at("version") == 1, "Unsupported recipe version");
    std::unordered_map<std::string, GGUFMetaData> metadata;
    for (const auto& entry : recipe.at("metadata")) {
        const auto name = entry.at("name").get<std::string>();
        OPENVINO_ASSERT(metadata.emplace(name, metadata_value(entry)).second, "Duplicate metadata: ", name);
    }
    GgufWeightTable weights;
    for (const auto& entry : recipe.at("tensors")) {
        GgufWeight source;
        source.source_name = entry.at("name").get<std::string>();
        source.shape = recipe_shape(entry.at("shape"), source.source_name);
        source.qtype = static_cast<GgufTensorType>(entry.at("ggml_type").get<int>());
        const auto type = ov::element::Type(entry.at("storage_type").get<std::string>());
        source.data = ov::make_tensor(ov::SoPtr<ov::ITensor>{
            std::make_shared<DescriptorTensor>(type,
                                               recipe_shape(entry.at("storage_shape"), source.source_name),
                                               source.source_name)});
        OPENVINO_ASSERT(weights.emplace(source.source_name, source).second, "Duplicate tensor: ", source.source_name);
    }
    const auto translators = get_supported_ops();
    const auto builder =
        make_gguf_builder(GGUFLoad{std::move(metadata), std::move(weights), {}}, default_arch_registry());
    auto graph = builder(translators);
    FrontEnd frontend;
    frontend.add_extension(structural_weight_extension());
    std::shared_ptr<GgufDecoder> decoder = std::make_shared<GgufBuilderDecoder>(graph);
    return frontend.convert(frontend.load(decoder));
}

Json weight_layout(const ov::frontend::gguf::GgufWeight& source, const ov::Shape& shape) {
    const auto tensors = ov::frontend::gguf::allocate_gguf_weight(
        source,
        shape,
        [&](const ov::element::Type& type, const ov::Shape& tensor_shape) {
            return ov::make_tensor(
                ov::SoPtr<ov::ITensor>{std::make_shared<DescriptorTensor>(type, tensor_shape, source.source_name)});
        });
    auto result = Json::object();
    for (const auto& [name, tensor] : std::map<std::string, ov::Tensor>{{"weight", tensors.weight},
                                                                        {"scales", tensors.scales},
                                                                        {"zero_point", tensors.zero_point}}) {
        if (tensor)
            result[name] = {{"type", tensor.get_element_type().get_type_name()}, {"shape", tensor.get_shape()}};
    }
    return result;
}

Json source_descriptor(const ov::frontend::gguf::GgufWeight& source) {
    return {{"name", source.source_name},
            {"shape", source.shape},
            {"ggml_type", static_cast<int>(source.qtype)},
            {"native", source.native},
            {"storage_type", source.data.get_element_type().get_type_name()},
            {"storage_shape", source.data.get_shape()},
            {"unpacked_layout", weight_layout(source, source.shape)},
            {"group_size", source.shape.empty() ? 1 : ov::frontend::gguf::gguf_weight_group_size(source)}};
}

class PlannedWeight : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("GGUFPlannedWeight", "gguf", ov::op::util::FrameworkNode);

    PlannedWeight(const ov::PartialShape& shape, ov::element::Type type, Json plan)
        : FrameworkNode({}, 1),
          m_shape(shape),
          m_type(type),
          m_plan(std::move(plan)) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, m_type, m_shape);
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        OPENVINO_ASSERT(inputs.empty(), "Planned weight has no inputs");
        return std::make_shared<PlannedWeight>(m_shape, m_type, m_plan);
    }

    const Json& plan() const {
        return m_plan;
    }

private:
    ov::PartialShape m_shape;
    ov::element::Type m_type;
    Json m_plan;
};

template <typename T>
Json scalar_value(T value) {
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(value))
            return std::isnan(value) ? "nan" : (value > 0 ? "+inf" : "-inf");
    }
    return value;
}

class Attributes : public ov::AttributeVisitor {
public:
    Json values = Json::object();
    Json bodies = Json::array();

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        if (auto* shape = dynamic_cast<ov::AttributeAdapter<ov::PartialShape>*>(&adapter)) {
            std::ostringstream stream;
            stream << shape->get();
            values[name] = {{"kind", "PartialShape"}, {"value", stream.str()}};
            return;
        }
        if (auto* variable = dynamic_cast<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>*>(&adapter)) {
            OPENVINO_ASSERT(variable->get(), "Null graph variable");
            const auto info = variable->get()->get_info();
            std::ostringstream shape;
            shape << info.data_shape;
            values[name] = {{"id", info.variable_id}, {"type", info.data_type.get_type_name()}, {"shape", shape.str()}};
            return;
        }
        if (auto* ports = dynamic_cast<ov::AttributeAdapter<ov::op::v5::Loop::SpecialBodyPorts>*>(&adapter)) {
            values[name] = {{"iteration_input", ports->get().current_iteration_input_idx},
                            {"condition_output", ports->get().body_condition_output_idx}};
            return;
        }
        using SubGraph = ov::op::util::MultiSubGraphOp;
        if (auto* inputs = dynamic_cast<ov::AttributeAdapter<ov::op::util::InputDescriptionVector>*>(&adapter)) {
            auto descriptions = Json::array();
            for (const auto& input : inputs->get()) {
                OPENVINO_ASSERT(input, "Null body input description");
                Json item{{"type", input->get_type_info().name},
                          {"input", input->m_input_index},
                          {"parameter", input->m_body_parameter_index}};
                if (const auto slice = std::dynamic_pointer_cast<SubGraph::SliceInputDescription>(input)) {
                    item["slice"] = {slice->m_start, slice->m_stride, slice->m_part_size, slice->m_end, slice->m_axis};
                } else if (const auto merged = std::dynamic_pointer_cast<SubGraph::MergedInputDescription>(input)) {
                    item["body_value"] = merged->m_body_value_index;
                } else {
                    OPENVINO_ASSERT(std::dynamic_pointer_cast<SubGraph::InvariantInputDescription>(input),
                                    "Unclassified body input description");
                }
                descriptions.push_back(std::move(item));
            }
            values[name] = std::move(descriptions);
            return;
        }
        if (auto* outputs = dynamic_cast<ov::AttributeAdapter<ov::op::util::OutputDescriptionVector>*>(&adapter)) {
            auto descriptions = Json::array();
            for (const auto& output : outputs->get()) {
                OPENVINO_ASSERT(output, "Null body output description");
                Json item{{"type", output->get_type_info().name},
                          {"body_value", output->m_body_value_index},
                          {"output", output->m_output_index}};
                if (const auto concat = std::dynamic_pointer_cast<SubGraph::ConcatOutputDescription>(output)) {
                    item["slice"] = {concat->m_start,
                                     concat->m_stride,
                                     concat->m_part_size,
                                     concat->m_end,
                                     concat->m_axis};
                } else if (const auto body = std::dynamic_pointer_cast<SubGraph::BodyOutputDescription>(output)) {
                    item["iteration"] = body->m_iteration;
                } else {
                    OPENVINO_THROW("Unclassified body output description");
                }
                descriptions.push_back(std::move(item));
            }
            values[name] = std::move(descriptions);
            return;
        }
        if (auto* visitor = dynamic_cast<ov::VisitorAdapter*>(&adapter)) {
            Attributes nested;
            OPENVINO_ASSERT(visitor->visit_attributes(nested), "Cannot capture attribute: ", name);
            values[name] = {{"type", adapter.get_type_info().name}, {"attributes", nested.values}};
            for (auto& body : nested.bodies)
                bodies.push_back(std::move(body));
            return;
        }
        OPENVINO_THROW("Unclassified graph attribute '", name, "' of type ", adapter.get_type_info().name);
    }

#define SNAPSHOT_SCALAR(TYPE)                                                             \
    void on_adapter(const std::string& name, ov::ValueAccessor<TYPE>& adapter) override { \
        values[name] = { {"kind", #TYPE}, {"value", scalar_value(adapter.get())} };       \
    }
    SNAPSHOT_SCALAR(bool)
    SNAPSHOT_SCALAR(std::string)
    SNAPSHOT_SCALAR(int8_t)
    SNAPSHOT_SCALAR(int16_t)
    SNAPSHOT_SCALAR(int32_t)
    SNAPSHOT_SCALAR(int64_t)
    SNAPSHOT_SCALAR(uint8_t)
    SNAPSHOT_SCALAR(uint16_t)
    SNAPSHOT_SCALAR(uint32_t)
    SNAPSHOT_SCALAR(uint64_t)
    SNAPSHOT_SCALAR(float)
    SNAPSHOT_SCALAR(double)
#undef SNAPSHOT_SCALAR

#define SNAPSHOT_VECTOR(TYPE)                                                                          \
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<TYPE>>& adapter) override { \
        auto result = Json::array();                                                                   \
        for (const auto& value : adapter.get())                                                        \
            result.push_back(scalar_value(value));                                                     \
        values[name] = {{"kind", "vector<" #TYPE ">"}, {"value", std::move(result)}};                  \
    }
    SNAPSHOT_VECTOR(int8_t)
    SNAPSHOT_VECTOR(int16_t)
    SNAPSHOT_VECTOR(int32_t)
    SNAPSHOT_VECTOR(int64_t)
    SNAPSHOT_VECTOR(uint8_t)
    SNAPSHOT_VECTOR(uint16_t)
    SNAPSHOT_VECTOR(uint32_t)
    SNAPSHOT_VECTOR(uint64_t)
    SNAPSHOT_VECTOR(float)
    SNAPSHOT_VECTOR(double)
    SNAPSHOT_VECTOR(std::string)
#undef SNAPSHOT_VECTOR

    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        auto snapshot = graph_snapshot(adapter.get());
        values[name] = {{"kind", "Model"}, {"value", std::move(snapshot.at("canonical"))}};
        snapshot.erase("canonical");
        bodies.push_back(std::move(snapshot));
    }
};

Json attributes(const std::shared_ptr<ov::Node>& node, Json& bodies) {
    if (const auto weight = ov::as_type_ptr<PlannedWeight>(node))
        return weight->plan();
    if (const auto weight = ov::as_type_ptr<ov::frontend::gguf::GGUFConstant>(node))
        return source_descriptor(weight->source());
    if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        Json result{{"type", constant->get_element_type().get_type_name()}, {"shape", constant->get_shape()}};
        const auto size = ov::shape_size(constant->get_shape());
        if (size == 0) {
            result["values"] = Json::array();
        } else if (size <= 64 || constant->get_element_type() == ov::element::boolean ||
                   (constant->get_element_type().is_integral_number() &&
                    constant->get_element_type().bitwidth() >= 32)) {
            try {
                result["values"] = constant->get_value_strings();
            } catch (const ov::Exception& error) {
                OPENVINO_THROW("Cannot capture Constant of type ",
                               constant->get_element_type(),
                               " and shape ",
                               constant->get_shape(),
                               ": ",
                               error.what());
            }
        }
        return result;
    }
    Attributes visitor;
    OPENVINO_ASSERT(node->visit_attributes(visitor), "Cannot visit ", node->get_type_name());
    bodies = std::move(visitor.bodies);
    return visitor.values;
}

Json runtime_value(const ov::Any& value) {
    if (value.is<bool>())
        return value.as<bool>();
    if (value.is<std::string>())
        return value.as<std::string>();
    if (value.is<ov::Shape>())
        return value.as<ov::Shape>();
    if (value.is<int64_t>())
        return value.as<int64_t>();
    if (value.is<int>())
        return value.as<int>();
    if (value.is<uint64_t>())
        return value.as<uint64_t>();
    if (value.is<float>())
        return scalar_value(value.as<float>());
    if (value.is<double>())
        return scalar_value(value.as<double>());
    if (value.is<std::vector<std::string>>())
        return value.as<std::vector<std::string>>();
    if (value.is<std::vector<int64_t>>())
        return value.as<std::vector<int64_t>>();
    if (value.is<ov::AnyMap>()) {
        auto result = Json::object();
        for (const auto& [key, entry] : value.as<ov::AnyMap>())
            result[key] = runtime_value(entry);
        return result;
    }
    OPENVINO_THROW("Unclassified GGUF runtime value type");
}

Json runtime_keys(const ov::RTMap& info) {
    static const std::set<std::string> semantic{"gguf.token_embedding",
                                                "gguf.per_layer_token_embedding",
                                                "gguf_u2_zero_point_shape",
                                                "gguf_hadamard_block",
                                                "gguf_hadamard_input_sign",
                                                "gguf_swa_window_size",
                                                "gguf_recurrent_states",
                                                "gguf_mmproj",
                                                "gguf_architecture",
                                                "gguf_is_imrope",
                                                "gguf_model_contract"};
    Json result = Json::object();
    for (const auto& [name, value] : info) {
        if (name.compare(0, 4, "gguf") != 0)
            continue;
        if (name == std::string(ov::frontend::gguf::ModelAdapterAttribute::get_type_info_static()) ||
            name == "gguf_tokenizer_metadata")
            continue;
        OPENVINO_ASSERT(semantic.count(name), "Unclassified GGUF runtime key: ", name);
        result[name] = runtime_value(value);
    }
    return result;
}

std::string shape_string(const ov::PartialShape& shape) {
    std::ostringstream stream;
    stream << shape;
    return stream.str();
}

std::string projection_storage(const std::shared_ptr<ov::Node>& root) {
    std::set<const ov::Node*> visited;
    std::set<std::string> formats;
    const std::function<void(const std::shared_ptr<ov::Node>&)> visit = [&](const std::shared_ptr<ov::Node>& node) {
        if (!visited.insert(node.get()).second)
            return;
        if (const auto planned = ov::as_type_ptr<PlannedWeight>(node)) {
            for (const auto& row : planned->plan().at("rows")) {
                const auto& source = row.at("source");
                formats.insert("ggml:" + std::to_string(source.at("ggml_type").get<int>()) +
                               "/group:" + std::to_string(source.at("group_size").get<size_t>()) +
                               "/weight:" + source.at("unpacked_layout").at("weight").at("type").get<std::string>());
            }
            return;
        }
        if (const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
            formats.insert(std::string("constant:") + constant->get_element_type().get_type_name());
            return;
        }
        const std::string type = node->get_type_name();
        if (type == "Convert" || type == "Reshape" || type == "Transpose" || type == "Split" ||
            type == "VariadicSplit" || type == "Multiply" || type == "Subtract") {
            visit(node->get_input_node_shared_ptr(0));
        } else if (type == "Concat") {
            for (const auto& input : node->input_values())
                visit(input.get_node_shared_ptr());
        }
    };
    visit(root);
    std::string result;
    for (const auto& format : formats) {
        if (!result.empty())
            result += "+";
        result += format;
    }
    return result.empty() ? "computed" : result;
}

class Graph {
public:
    explicit Graph(const std::shared_ptr<ov::Model>& model) {
        for (const auto& parameter : model->get_parameters())
            m_named_ports.emplace(parameter.get(), 0);
        for (const auto& result : model->get_results()) {
            m_named_ports.emplace(result.get(), 0);
            const auto input = result->input_value(0);
            m_named_ports.emplace(input.get_node(), input.get_index());
        }
    }

    Json nodes = Json::array();
    size_t edges = 0;
    size_t body_nodes = 0;
    size_t body_states = 0;
    std::map<std::pair<std::string, std::string>, size_t> ops;
    std::map<std::tuple<std::string, std::string, std::string, std::string>, size_t> projections;

    size_t visit(const std::shared_ptr<ov::Node>& node) {
        if (!discover(node))
            return m_ids.at(node.get());
        struct Frame {
            std::shared_ptr<ov::Node> node;
            size_t input = 0;
            size_t control = 0;
        };
        std::vector<Frame> pending{{node}};
        while (!pending.empty()) {
            auto& frame = pending.back();
            if (frame.input < frame.node->get_input_size()) {
                auto input = frame.node->get_input_node_shared_ptr(frame.input++);
                if (discover(input))
                    pending.push_back({std::move(input)});
            } else if (frame.control < frame.node->get_control_dependencies().size()) {
                auto control = frame.node->get_control_dependencies().at(frame.control++);
                if (discover(control))
                    pending.push_back({std::move(control)});
            } else {
                emit(frame.node);
                pending.pop_back();
            }
        }
        return m_ids.at(node.get());
    }

private:
    bool discover(const std::shared_ptr<ov::Node>& node) {
        if (m_ids.count(node.get()))
            return false;
        const auto id = nodes.size();
        m_ids.emplace(node.get(), id);
        nodes.push_back(nullptr);
        const auto& type = node->get_type_info();
        const std::string version = type.version_id ? type.version_id : "";
        ++ops[{type.name, version}];
        return true;
    }

    void emit(const std::shared_ptr<ov::Node>& node) {
        const auto id = m_ids.at(node.get());
        const auto& type = node->get_type_info();
        const std::string version = type.version_id ? type.version_id : "";
        Json inputs = Json::array();
        for (const auto& input : node->input_values()) {
            inputs.push_back({m_ids.at(input.get_node()), input.get_index()});
            ++edges;
        }
        Json outputs = Json::array();
        for (size_t port = 0; port < node->get_output_size(); ++port) {
            const auto& names = node->output(port).get_names();
            Json output{{"shape", shape_string(node->get_output_partial_shape(port))},
                        {"type", node->get_output_element_type(port).get_type_name()}};
            if (m_named_ports.count({node.get(), port}))
                output["names"] = std::set<std::string>(names.begin(), names.end());
            outputs.push_back(std::move(output));
        }
        Json controls = Json::array();
        for (const auto& dependency : node->get_control_dependencies())
            controls.push_back(m_ids.at(dependency.get()));
        if (ov::is_type<ov::op::v0::MatMul>(node)) {
            const auto rank = node->get_input_partial_shape(0).rank();
            ++projections[{type.name,
                           rank.is_static() ? std::to_string(rank.get_length()) : "?",
                           node->get_input_element_type(1).get_type_name(),
                           projection_storage(node->get_input_node_shared_ptr(1))}];
        }
        Json bodies = Json::array();
        auto node_attributes = attributes(node, bodies);
        for (const auto& body : bodies) {
            body_nodes += body.at("totals").at("nodes").get<size_t>();
            edges += body.at("totals").at("edges").get<size_t>();
            body_states += body.at("totals").at("states").get<size_t>();
            for (const auto& row : body.at("ops"))
                ops[{row.at(0).get<std::string>(), row.at(1).get<std::string>()}] += row.at(2).get<size_t>();
            for (const auto& row : body.at("projections"))
                projections[{row.at(0).get<std::string>(),
                             row.at(1).get<std::string>(),
                             row.at(2).get<std::string>(),
                             row.at(3).get<std::string>()}] += row.at(4).get<size_t>();
        }
        nodes[id] = {{"type", type.name},
                     {"version", version},
                     {"inputs", inputs},
                     {"outputs", outputs},
                     {"attributes", node_attributes},
                     {"control", controls},
                     {"runtime", runtime_keys(node->get_rt_info())}};
    }

    std::map<const ov::Node*, size_t> m_ids;
    std::set<std::pair<const ov::Node*, size_t>> m_named_ports;
};
}  // namespace

std::shared_ptr<ov::frontend::gguf::WeightNormalizationExtension> structural_weight_extension() {
    return std::make_shared<ov::frontend::gguf::WeightNormalizationExtension>([](ov::pass::Manager& manager) {
        manager.register_pass<Materializer>([](const std::shared_ptr<ov::Node>& target,
                                               const std::vector<Materializer::SourceRows>& sources,
                                               const std::vector<int64_t>& columns) {
            Json rows = Json::array();
            for (const auto& source : sources)
                rows.push_back(
                    {{"source", source_descriptor(source.source)}, {"first", source.first}, {"count", source.count}});
            return std::make_shared<PlannedWeight>(
                target->get_output_partial_shape(0),
                target->get_output_element_type(0),
                Json{{"rows", rows},
                     {"columns", columns},
                     {"unpacked_layout", weight_layout(sources.front().source, target->get_output_shape(0))}});
        });
    });
}

nlohmann::json graph_snapshot(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "Cannot snapshot an empty model");
    Graph graph(model);
    Json results = Json::array(), parameters = Json::array(), sinks = Json::array(), variables = Json::array();
    for (const auto& result : model->get_results())
        results.push_back(graph.visit(result));
    for (const auto& parameter : model->get_parameters())
        parameters.push_back(graph.visit(parameter));
    for (const auto& sink : model->get_sinks())
        sinks.push_back(graph.visit(sink));
    auto states = model->get_variables();
    std::sort(states.begin(), states.end(), [](const auto& left, const auto& right) {
        return left->get_info().variable_id < right->get_info().variable_id;
    });
    for (const auto& state : states) {
        const auto info = state->get_info();
        variables.push_back({{"id", info.variable_id},
                             {"type", info.data_type.get_type_name()},
                             {"shape", shape_string(info.data_shape)}});
    }
    Json ops = Json::array(), projections = Json::array();
    for (const auto& [type, count] : graph.ops)
        ops.push_back({type.first, type.second, count});
    for (const auto& [key, count] : graph.projections)
        projections.push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key), count});
    return {{"canonical",
             {{"nodes", graph.nodes},
              {"results", results},
              {"parameters", parameters},
              {"sinks", sinks},
              {"variables", variables},
              {"runtime", runtime_keys(model->get_rt_info())}}},
            {"ops", ops},
            {"projections", projections},
            {"totals",
             {{"nodes", graph.nodes.size() + graph.body_nodes},
              {"top_level_nodes", graph.nodes.size()},
              {"inputs", model->get_parameters().size()},
              {"outputs", model->get_results().size()},
              {"edges", graph.edges},
              {"states", model->get_variables().size() + graph.body_states}}}};
}

int graph_snapshot_main(int argc, char** argv) {
    OPENVINO_ASSERT(argc == 5 || argc == 6,
                    "Usage: --gguf-snapshot <input.gguf> --output <snapshot.json> [--materialized]");
    OPENVINO_ASSERT(std::string(argv[3]) == "--output", "Expected --output");
    std::shared_ptr<ov::Model> model;
    if (argc == 5) {
        std::ifstream input(argv[2]);
        OPENVINO_ASSERT(input, "Cannot read recipe: ", argv[2]);
        Json recipe;
        input >> recipe;
        model = convert_recipe(recipe);
    } else {
        OPENVINO_ASSERT(std::string(argv[5]) == "--materialized", "Unknown snapshot option");
        ov::frontend::gguf::FrontEnd frontend;
        model = frontend.convert(frontend.load(std::string(argv[2])));
    }
    const auto snapshot = graph_snapshot(model);
    std::ofstream output(argv[4]);
    OPENVINO_ASSERT(output, "Cannot create snapshot: ", argv[4]);
    output << snapshot.dump() << '\n';
    OPENVINO_ASSERT(output, "Cannot write snapshot: ", argv[4]);
    return 0;
}

}  // namespace ov_gguf_test
