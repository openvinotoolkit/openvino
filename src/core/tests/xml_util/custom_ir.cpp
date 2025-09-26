// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <pugixml.hpp>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/add.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "openvino/xml_util/constant_writer.hpp"
#include "openvino/xml_util/xml_deserialize_util.hpp"
#include "openvino/xml_util/xml_serialize_util.hpp"

namespace ov::test {

using op::v0::Parameter, op::v0::Constant, op::v1::Add;
using WeightsMap = std::unordered_map<size_t, std::shared_ptr<ov::AlignedBuffer>>;

class CustomIRTest : public testing::Test {
protected:
    std::filesystem::path m_out_xml_path;
    std::filesystem::path m_out_bin_path;
    std::shared_ptr<Model> m_model;

    void SetUp() override {
        const auto file_prefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path = file_prefix + ".xml";
        m_out_bin_path = file_prefix + ".bin";
    }

    void TearDown() override {
        if (std::filesystem::exists(m_out_xml_path)) {
            std::filesystem::remove(m_out_xml_path);
        }

        if (std::filesystem::exists(m_out_bin_path)) {
            std::filesystem::remove(m_out_bin_path);
        }
    }

    static FunctionsComparator model_comparator() {
        return FunctionsComparator::with_default()
            .enable(FunctionsComparator::ATTRIBUTES)
            .enable(FunctionsComparator::CONST_VALUES);
    }

public:
    // Example of read model to use modified IR deserializer
    template <class Deserializer>
    std::shared_ptr<ov::Model> read_model(const std::string& smodel,
                                          const std::shared_ptr<ov::AlignedBuffer>& org_weights,
                                          const WeightsMap& weights_map) {
        pugi::xml_document xml_doc;

        // get root node for IR model (xml)
        const auto root = [&] {
            // get IR blob header
            auto* buffer_base = smodel.data();
            pass::StreamSerialize::DataHeader hdr = {};
            std::memcpy(reinterpret_cast<char*>(&hdr), buffer_base, sizeof(hdr));
            // Skip header validation, assume it is OK

            // skip custom data, and weights
            auto res = xml_doc.load_buffer(buffer_base + hdr.model_offset,
                                           hdr.model_size,
                                           pugi::parse_default,
                                           pugi::encoding_utf8);
            OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
            return xml_doc.document_element();
        }();

        // get available opsets (all from OV)
        const auto opsets = [] {
            std::unordered_map<std::string, ov::OpSet> opsets;
            for (const auto& [name, mk_opset] : ov::get_available_opsets()) {
                opsets[name] = mk_opset();
            }
            return opsets;
        }();
        const auto version = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(root, "version", 0));

        // create required extensions (none in this example)
        auto create_extensions_map = [&]() -> std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> {
            std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> exts;
            return exts;
        }();

        std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
        // deserialize model
        std::shared_ptr<ov::Model> model;
        if constexpr (std::is_same_v<Deserializer, ov::util::XmlDeserializer>) {
            Deserializer visitor(root, org_weights, opsets, create_extensions_map, variables, version);
            visitor.on_attribute("net", model);
        } else {
            Deserializer visitor(root, org_weights, weights_map, opsets, create_extensions_map, variables, version);
            visitor.on_attribute("net", model);
        }

        model->get_rt_info()["version"] = int64_t(version);
        return model;
    }
};

// Custom constant writer for weightless (skip store)
class WeightlessWriter : public ov::util::ConstantWriter {
public:
    explicit WeightlessWriter(ov::util::ConstantWriter& other) : ov::util::ConstantWriter(other) {}

    FilePosition write(const char*, size_t, size_t&, bool, ov::element::Type, bool) override {
        // use new_size not modified and return offset 0 to store these in modifed IR (xmL) only
        return 0;
    }
};

// Custom serializer to store weights in the map during serialization (which not exists in original model)
class WeightMapWriter : public ov::util::ConstantWriter {
public:
    explicit WeightMapWriter(ov::util::ConstantWriter& other, WeightsMap& weights_map)
        : ov::util::ConstantWriter(other),
          m_weights_map{std::ref(weights_map)} {}

    FilePosition write(const char* ptr, size_t size, size_t& new_size, bool, ov::element::Type, bool) override {
        auto weights = std::make_shared<ov::AlignedBuffer>(size);
        std::memcpy(weights->get_ptr(), ptr, size);

        auto w_id = reinterpret_cast<size_t>(weights.get());
        m_weights_map.get().emplace(w_id, std::move(weights));
        return static_cast<FilePosition>(w_id);
    }

private:
    std::reference_wrapper<WeightsMap> m_weights_map;
};

// custom serializer will skip original weights and store new weights in the map
class XmlSerializer : public ov::util::XmlSerializer {
public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  ov::util::ConstantWriter& constant_write_handler,
                  int64_t version,
                  WeightsMap& weights_map)
        : ov::util::XmlSerializer(data,
                                  node_type_name,
                                  constant_write_handler,
                                  version,
                                  false,
                                  false,
                                  ov::element::dynamic,
                                  false),
          m_use_weightless_writer{false},
          m_wl_const_writer(constant_write_handler),
          m_wm_const_writer(constant_write_handler, weights_map),
          m_weights_map(std::ref(weights_map)) {}

private:
    bool append_rt_attribute(pugi::xml_node& node, const ov::RuntimeAttribute& attribute) override {
        // IR customization serialize weightless attribute
        if (auto wl_attr = ov::as_type<const ov::WeightlessCacheAttribute>(&attribute)) {
            const auto& type_info = attribute.get_type_info();
            node.append_attribute("name").set_value(type_info.name);
            node.append_attribute("version").set_value(type_info.get_version().c_str());
            node.append_attribute("type").set_value(ov::util::get_ir_precision_name(wl_attr->original_dtype).c_str());
            node.append_attribute("offset").set_value(wl_attr->bin_offset);
            node.append_attribute("size").set_value(wl_attr->original_size);
            return true;
        } else {
            return ov::util::XmlSerializer::append_rt_attribute(node, attribute);
        }
    }

    bool append_node_attributes(ov::Node& node) override {
        // depends on node RT info use custom constant writer when serialize attributes
        m_use_weightless_writer = node.get_rt_info().count(ov::WeightlessCacheAttribute::get_type_info_static()) != 0;
        auto result = ov::util::XmlSerializer::append_node_attributes(node);
        m_use_weightless_writer = false;
        return result;
    }

    ov::util::ConstantWriter& get_constant_write_handler() override {
        if (m_use_weightless_writer) {
            // Will skip serialize weights
            return m_wl_const_writer;
        } else {
            // Will add weights to the map
            return m_wm_const_writer;
        }
    }

    std::unique_ptr<ov::util::XmlSerializer> make_visitor(pugi::xml_node& data,
                                                          const std::string& node_type_name,
                                                          ov::util::ConstantWriter& constant_write_handler,
                                                          int64_t version,
                                                          bool,
                                                          bool,
                                                          ov::element::Type,
                                                          bool) const override {
        return std::make_unique<XmlSerializer>(data,
                                               node_type_name,
                                               constant_write_handler,
                                               version,
                                               std::ref(m_weights_map));
    }

    bool m_use_weightless_writer;
    WeightlessWriter m_wl_const_writer;
    WeightMapWriter m_wm_const_writer;
    std::reference_wrapper<WeightsMap> m_weights_map;
};
class StreamSerialize : public ov::pass::StreamSerialize {
public:
    StreamSerialize(std::ostream& stream, ov::pass::Serialize::Version version, WeightsMap& map)
        : ov::pass::StreamSerialize(stream, {}, {}, version),
          m_weights_map(std::ref(map)) {}

private:
    std::unique_ptr<util::XmlSerializer> make_serializer(pugi::xml_node& data,
                                                         const std::string& node_type_name,
                                                         util::ConstantWriter& constant_write_handler,
                                                         int64_t version,
                                                         bool,
                                                         bool,
                                                         ov::element::Type,
                                                         bool) const override {
        return std::make_unique<XmlSerializer>(data, node_type_name, constant_write_handler, version, m_weights_map);
    }

public:
    std::reference_wrapper<WeightsMap> m_weights_map;
};

class XmlDeserializer : public ov::util::XmlDeserializer {
    // Local pugi helpers
    template <class T>
    static void str_to_container(const std::string& value, T& res) {
        std::stringstream ss(value);
        std::string field;
        while (getline(ss, field, ',')) {
            if (field.empty())
                OPENVINO_THROW("Cannot get vector of parameters! \"", value, "\" is incorrect");
            std::stringstream fs(field);
            typename T::value_type val;
            fs >> val;
            res.insert(res.end(), val);
        }
    }

    template <class T>
    static bool getParameters(const pugi::xml_node& node, const std::string& name, std::vector<T>& value) {
        str_to_container(ov::util::pugixml::get_str_attr(node, name.c_str()), value);
        return true;
    }

public:
    explicit XmlDeserializer(const pugi::xml_node& node,
                             const std::shared_ptr<ov::AlignedBuffer>& origin_weights,
                             const WeightsMap& weights_map,
                             const std::unordered_map<std::string, ov::OpSet>& opsets,
                             const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
                             std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
                             size_t version)
        : ov::util::XmlDeserializer(node, origin_weights, opsets, extensions, variables, version),
          m_origin_weights{origin_weights},
          m_weights_map{std::ref(weights_map)} {}

protected:
    static const std::shared_ptr<ov::AlignedBuffer>& get_origin_weights(const WeightsMap& weights_map) {
        return weights_map.at(0);
    }

    ov::Any parse_weightless_cache_attribute(const pugi::xml_node& node) const override {
        // custom parse constant properties to weightless cache attribute
        if (auto rt_info = node.child("rt_info")) {
            for (const auto& child : rt_info.children()) {
                for (const auto& attr : child.attributes()) {
                    if (strcmp(attr.name(), "name") == 0 &&
                        strcmp(attr.value(), ov::WeightlessCacheAttribute::get_type_info_static().name) == 0) {
                        const auto origin_size = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(child, "size"));
                        const auto offset = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(child, "offset"));
                        const ov::element::Type original_dt(child.attribute("type").value());
                        return {ov::WeightlessCacheAttribute{origin_size, offset, original_dt}};
                    }
                }
            }
        }
        return {};
    }

    void set_constant_num_buffer(ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>& adapter) override {
        // custom adapter to set AlignedBuffer as node (Constant) attribute
        const auto node = get_node();
        const auto& dn = node.child("data");
        const auto el_type = ov::element::Type(ov::util::pugixml::get_str_attr(dn, "element_type"));
        if (el_type == element::string) {
            ov::util::XmlDeserializer::set_constant_num_buffer(adapter);
        } else {
            ov::Shape shape;
            {
                std::vector<int64_t> shapev;
                if (!getParameters<int64_t>(dn, "shape", shapev)) {
                    return;
                }
                shape.assign(shapev.begin(), shapev.end());
            }
            // Some test case Here is an issue as after call the code below is not executed and Const buffer is
            auto wl_attr = parse_weightless_cache_attribute(node);
            // Dummy read attributes will throw if not exists
            auto offset = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "offset"));
            auto actual_size = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "size"));
            // auto original_dtype = el_type;
            if (wl_attr.is<ov::WeightlessCacheAttribute>()) {
                // Use weightless cache attribute to map original weights
                const auto& wl = wl_attr.as<ov::WeightlessCacheAttribute>();

                actual_size = wl.original_size;
                offset = wl.bin_offset;
                auto original_dtype = wl.original_dtype;
                char* data = m_origin_weights->get_ptr<char>() + offset;
                auto w_size = m_origin_weights->size();
                auto w_so = m_origin_weights;

                OPENVINO_ASSERT(w_size >= offset + actual_size, "Incorrect weights in bin file!");
                if (original_dtype != el_type) {
                    OPENVINO_THROW("Conversion of weights to another type is not supported in weightless mode!");
                } else {
                    OPENVINO_ASSERT(w_size >= offset + actual_size, "Incorrect weights in bin file!");

                    if (actual_size < ((ov::shape_size(shape) * el_type.bitwidth() + 7) >> 3)) {
                        const auto type = ov::util::pugixml::get_str_attr(get_node(), "type");
                        OPENVINO_THROW("Attribute and shape size are inconsistent for ",
                                       type,
                                       " op!",
                                       actual_size,
                                       ", ",
                                       ((ov::shape_size(shape) * el_type.bitwidth() + 7) >> 3),
                                       ", ",
                                       ov::util::get_memory_size(el_type, ov::shape_size(shape)));
                    }

                    auto buffer =
                        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(data, actual_size, w_so);
                    adapter.set(buffer);
                }
            } else {
                auto& buff = m_weights_map.get().at(offset);
                adapter.set(buff);
            }
        }
    }

private:
    std::unique_ptr<ov::util::XmlDeserializer> make_visitor(
        const pugi::xml_node& node,
        const std::shared_ptr<ov::AlignedBuffer>& origin_weights,
        const std::unordered_map<std::string, ov::OpSet>& opsets,
        const std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>& extensions,
        std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
        size_t version) const override {
        return std::make_unique<XmlDeserializer>(node,
                                                 origin_weights,
                                                 m_weights_map,
                                                 opsets,
                                                 extensions,
                                                 variables,
                                                 version);
    }

    std::shared_ptr<ov::AlignedBuffer> m_origin_weights;
    std::reference_wrapper<const WeightsMap> m_weights_map;
};

TEST_F(CustomIRTest, modified_serialization_deserialization) {
    // create sample OV model
    {
        auto in0 = std::make_shared<Parameter>(element::f32, Shape{1, 3, 22, 22});
        auto in1 = std::make_shared<Parameter>(element::f32, Shape{1, 3, 22, 22});
        auto c1 = std::make_shared<Constant>(element::f32, Shape{10, 3, 22, 22}, std::vector<float>{1.f});
        auto c2 = std::make_shared<Constant>(element::f32, Shape{1, 3, 22, 22}, std::vector<float>{2.f});
        c1->set_friendly_name("const1");
        c2->set_friendly_name("const2");
        auto add1 = std::make_shared<Add>(in0, c1);
        auto add2 = std::make_shared<Add>(add1, in1);
        add2->set_friendly_name("add2");
        auto add = std::make_shared<Add>(add2, c2);
        auto model = std::make_shared<Model>(OutputVector{add}, ParameterVector{in0, in1}, "Sample");
        ov::serialize(model, m_out_xml_path, m_out_bin_path);
    };

    // read IR model
    auto ov_model = ov::Core().read_model(m_out_xml_path);  // read model with weights

    {
        // simulate some transformations on model
        for (auto&& op : ov_model->get_ops()) {
            if (op->get_friendly_name() == "add2") {
                auto c3 = std::make_shared<Constant>(element::f32, Shape{10, 3, 22, 22}, std::vector<float>{33.f});
                auto c4 = std::make_shared<Constant>(element::f32, Shape{1, 1, 22, 22}, std::vector<float>{4.4f});
                c3->set_friendly_name("const3");
                c4->set_friendly_name("const4");
                auto new_add3 = std::make_shared<Add>(op->input_value(0), c3);
                auto new_add4 = std::make_shared<Add>(op->input_value(0), c4);
                new_add4->set_friendly_name("new_add4");
                ov_model->add_output(new_add4);
                ov_model->validate_nodes_and_infer_types();
            }
        }
    }

    WeightsMap weights_map;
    std::stringstream blob_stream;
    {
        // plugin may export model as customized IR (no weights).
        ov::test::StreamSerialize dev_exporter(blob_stream, ov::pass::Serialize::Version::IR_V11, weights_map);
        dev_exporter.run_on_model(ov_model);
    }

    // Driver will import model with custom deserializer for IR
    auto weights = ov::read_tensor_data(m_out_bin_path);  // read weights as mapped file
    auto w_buffer = std::make_shared<SharedBuffer<Tensor>>(weights.data<char>(), weights.get_byte_size(), weights);
    auto drv_model = read_model<ov::test::XmlDeserializer>(blob_stream.str(), w_buffer, weights_map);

    const auto& [is_valid, error_msg] = model_comparator().compare(ov_model, drv_model);
    EXPECT_TRUE(is_valid) << error_msg;
}
}  // namespace ov::test
