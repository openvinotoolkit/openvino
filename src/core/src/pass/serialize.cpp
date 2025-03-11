// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/serialize.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <openvino/cc/pass/itt.hpp>
#include <unordered_map>
#include <unordered_set>

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/meta_data.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/compute_hash.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"
#include "openvino/util/file_util.hpp"
#include "pugixml.hpp"
#include "transformations/hash.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

namespace ov {
class OstreamHashWrapperBin final : public std::streambuf {
    uint64_t m_res = 0lu;

public:
    uint64_t getResult() const {
        return m_res;
    }
    std::streamsize xsputn(const char* s, std::streamsize n) override;
};
}  // namespace ov

namespace {  // helpers
template <typename Container>
std::string join(const Container& c, const char* glue = ", ") {
    std::stringstream oss;
    const char* s = "";
    for (const auto& v : c) {
        oss << s << v;
        s = glue;
    }
    return oss.str();
}

struct Edge {
    int from_layer = 0;
    int from_port = 0;
    int to_layer = 0;
    int to_port = 0;
};

// Here operation type names are translated from OpenVINO Model convention to IR
// convention. Most of them are the same, but there are exceptions, e.g
// Constant (OpenVINO Model name) and Const (IR name). If there will be more
// discrepancies discovered, translations needs to be added here.
const std::unordered_map<std::string, std::string>& get_translate_type_name_translator() {
    static const std::unordered_map<std::string, std::string> translate_type_name_translator = {{"Constant", "Const"},
                                                                                                {"PRelu", "PReLU"},
                                                                                                {"Relu", "ReLU"},
                                                                                                {"Softmax", "SoftMax"}};
    return translate_type_name_translator;
}

std::string translate_type_name(const std::string& name) {
    auto found = get_translate_type_name_translator().find(name);
    if (found != end(get_translate_type_name_translator())) {
        return found->second;
    }
    return name;
}

class ConstantWriter {
public:
    using FilePosition = int64_t;
    using HashValue = size_t;
    using ConstWritePositions = std::multimap<HashValue, std::pair<FilePosition, void const*>>;

    ConstantWriter(std::ostream& bin_data, bool enable_compression = true)
        : m_binary_output(bin_data),
          m_enable_compression(enable_compression),
          m_blob_offset(bin_data.tellp()) {
        m_write_hash_value = (dynamic_cast<ov::OstreamHashWrapperBin*>(bin_data.rdbuf())) ? true : false;
    }

    FilePosition write(const char* ptr,
                       size_t size,
                       size_t& new_size,
                       bool compress_to_fp16 = false,
                       ov::element::Type src_type = ov::element::dynamic,
                       bool ptr_is_temporary = false) {  // when true, do not rely on ptr after this function call, data
                                                         // is temporary allocated
        const FilePosition write_pos = m_binary_output.tellp();
        const auto offset = write_pos - m_blob_offset;
        new_size = size;

        if (!m_enable_compression) {
            if (!compress_to_fp16) {
                m_binary_output.write(ptr, size);
            } else {
                OPENVINO_ASSERT(size % src_type.size() == 0);
                auto fp16_buffer = compress_data_to_fp16(ptr, size, src_type, new_size);
                m_binary_output.write(fp16_buffer.get(), new_size);
            }
            return offset;
        } else {
            std::unique_ptr<char[]> fp16_buffer = nullptr;
            if (compress_to_fp16) {
                OPENVINO_ASSERT(size % src_type.size() == 0);
                fp16_buffer = compress_data_to_fp16(ptr, size, src_type, new_size);
            }
            const char* ptr_to_write;
            if (fp16_buffer) {
                ptr_to_write = fp16_buffer.get();
            } else {
                ptr_to_write = ptr;
            }

            // This hash is weak (but efficient). For example current hash algorithms gives
            // the same hash for {2, 2} and {0, 128} arrays.
            // But even strong hashing algorithms sometimes give collisions.
            // Therefore we always have to compare values when finding a match in the hash multimap.
            const HashValue hash = ov::runtime::compute_hash(ptr_to_write, new_size);

            auto found = m_hash_to_file_positions.equal_range(hash);
            // iterate over all matches of the key in the multimap
            for (auto it = found.first; it != found.second; ++it) {
                if (memcmp(ptr, it->second.second, size) == 0) {
                    return it->second.first;
                }
            }
            if (!ptr_is_temporary) {
                // Since fp16_compressed data will be disposed at exit point and since we cannot reread it from the
                // ostream, we store pointer to the original uncompressed blob.
                m_hash_to_file_positions.insert({hash, {offset, static_cast<void const*>(ptr)}});
            }
            if (m_write_hash_value) {
                m_binary_output.write(reinterpret_cast<const char*>(&hash), sizeof(uint64_t));
            } else {
                m_binary_output.write(ptr_to_write, new_size);
            }
        }
        return offset;
    }

private:
    static std::unique_ptr<char[]> compress_data_to_fp16(const char* ptr,
                                                         size_t size,
                                                         ov::element::Type src_type,
                                                         size_t& compressed_size) {
        auto num_src_elements = size / src_type.size();
        compressed_size = num_src_elements * ov::element::f16.size();
        if (src_type == ov::element::f32) {
            auto new_ptr = std::unique_ptr<char[]>(new char[compressed_size]);
            auto dst_data = reinterpret_cast<ov::float16*>(new_ptr.get());
            auto src_data = reinterpret_cast<const float*>(ptr);
            ov::reference::convert_from_f32_to_f16_with_clamp(src_data, dst_data, num_src_elements);
            return new_ptr;
        } else if (src_type == ov::element::f64) {
            auto new_ptr = std::unique_ptr<char[]>(new char[compressed_size]);
            auto dst_data = reinterpret_cast<ov::float16*>(new_ptr.get());
            auto src_data = reinterpret_cast<const double*>(ptr);

            // Reference implementation for fp64 to fp16 conversoin
            for (size_t i = 0; i < num_src_elements; ++i) {
                // if abs value is smaller than the smallest positive fp16, but not zero
                if (std::abs(src_data[i]) < ov::float16::from_bits(0x0001) && src_data[i] != 0.0f) {
                    dst_data[i] = 0;
                } else if (src_data[i] > std::numeric_limits<ov::float16>::max()) {
                    dst_data[i] = std::numeric_limits<ov::float16>::max();
                } else if (src_data[i] < std::numeric_limits<ov::float16>::lowest()) {
                    dst_data[i] = std::numeric_limits<ov::float16>::lowest();
                } else {
                    dst_data[i] = static_cast<ov::float16>(src_data[i]);
                }
            }
            return new_ptr;
        } else {
            OPENVINO_THROW("[ INTERNAL ERROR ] Not supported source type for weights compression: ", src_type);
        }
    }

    ConstWritePositions m_hash_to_file_positions;
    std::ostream& m_binary_output;
    bool m_enable_compression;
    bool m_write_hash_value = false;
    FilePosition m_blob_offset;  // blob offset inside output stream
};

void ngfunction_2_ir(pugi::xml_node& node,
                     const ov::Model& model,
                     ConstantWriter& constant_write_handler,
                     int64_t version,
                     bool deterministic);

namespace rt_info {
static const std::vector<std::string> list_of_names{
    "PrimitivesPriority",
    "alt_width",
};

class XmlSerializer {
public:
    explicit XmlSerializer(pugi::xml_node& xml_node) : m_xml_node(xml_node) {}

    void serialize(const ov::Node::RTMap& rt_info) {
        for (const auto& rt_info_name : list_of_names) {
            const auto& found_rt_info = rt_info.find(rt_info_name);
            if (found_rt_info != rt_info.end()) {
                std::stringstream strm;
                found_rt_info->second.print(strm);
                m_xml_node.append_attribute(rt_info_name.c_str()).set_value(strm.str().c_str());
            }
        }
    }

private:
    pugi::xml_node& m_xml_node;
};

class RTInfoSerializer : public ov::AttributeVisitor {
    pugi::xml_node m_node;

public:
    RTInfoSerializer(const pugi::xml_node node) : m_node(node) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        check_attribute_name(name);
        if (auto a = ov::as_type<ov::AttributeAdapter<std::set<std::string>>>(&adapter)) {
            const auto& value = join(a->get());
            m_node.append_attribute(name.c_str()).set_value(value.c_str());
        } else {
            OPENVINO_THROW("Unsupported attribute type for serialization: ", name);
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        check_attribute_name(name);
        m_node.append_attribute(name.c_str()).set_value(adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        check_attribute_name(name);
        m_node.append_attribute(name.c_str()).set_value(adapter.get().c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        check_attribute_name(name);
        m_node.append_attribute(name.c_str()).set_value(static_cast<long long>(adapter.get()));
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        check_attribute_name(name);
        m_node.append_attribute(name.c_str()).set_value(adapter.get());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        check_attribute_name(name);
        const auto& value = join(adapter.get());
        m_node.append_attribute(name.c_str()).set_value(value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        check_attribute_name(name);
        const auto& value = join(adapter.get());
        m_node.append_attribute(name.c_str()).set_value(value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        check_attribute_name(name);
        const auto& value = join(adapter.get());
        m_node.append_attribute(name.c_str()).set_value(value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        check_attribute_name(name);
        const auto& value = join(adapter.get());
        m_node.append_attribute(name.c_str()).set_value(value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        check_attribute_name(name);
        const auto& value = join(adapter.get());
        m_node.append_attribute(name.c_str()).set_value(value.c_str());
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        OPENVINO_THROW("Model type is unsupported for rt info serialization");
    }

    void check_attribute_name(const std::string& name) const {
        if (name == "name" || name == "version") {
            OPENVINO_THROW("Attribute key with name: ", name, " is not allowed. Please use another name");
        }
    }
};
}  // namespace rt_info

class XmlSerializer : public ov::AttributeVisitor {
    pugi::xml_node& m_xml_node;
    const std::string& m_node_type_name;
    ConstantWriter& m_constant_write_handler;
    int64_t m_version;
    bool m_deterministic;
    bool m_compress_to_fp16;
    ov::element::Type m_output_element_type;
    bool m_data_is_temporary;

    template <typename T>
    std::string create_atribute_list(ov::ValueAccessor<std::vector<T>>& adapter) {
        return join(adapter.get());
    }

    std::vector<std::string> map_type_from_body(const pugi::xml_node& xml_node,
                                                const std::string& map_type,
                                                int64_t ir_version,
                                                const std::string& body_name = "body") {
        std::vector<std::string> output;
        for (pugi::xml_node node : xml_node.child(body_name.c_str()).child("layers")) {
            if (map_type == node.attribute("type").value()) {
                output.emplace_back(node.attribute("id").value());
            }
        }

        return output;
    }

    void input_descriptions_on_adapter(
        const std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>>& input_descriptions,
        const std::vector<std::string>& parameter_mapping,
        const std::vector<std::string>& result_mapping,
        pugi::xml_node& port_map,
        const std::string& portmap_name) {
        if (!m_xml_node.parent().child(portmap_name.c_str())) {
            port_map = m_xml_node.parent().insert_child_before(portmap_name.c_str(), m_xml_node.parent().first_child());
        }

        for (const auto& input_description : input_descriptions) {
            pugi::xml_node input = port_map.append_child("input");
            input.append_attribute("external_port_id")
                .set_value(static_cast<unsigned long long>(input_description->m_input_index));
            input.append_attribute("internal_layer_id")
                .set_value(parameter_mapping[input_description->m_body_parameter_index].c_str());

            if (auto slice_input =
                    ov::as_type_ptr<ov::op::util::SubGraphOp::SliceInputDescription>(input_description)) {
                input.prepend_attribute("axis").set_value(static_cast<long long>(slice_input->m_axis));
                input.append_attribute("start").set_value(static_cast<long long>(slice_input->m_start));
                input.append_attribute("end").set_value(static_cast<long long>(slice_input->m_end));
                input.append_attribute("stride").set_value(static_cast<long long>(slice_input->m_stride));
                input.append_attribute("part_size").set_value(static_cast<long long>(slice_input->m_part_size));
            } else if (auto merged_input =
                           ov::as_type_ptr<ov::op::util::SubGraphOp::MergedInputDescription>(input_description)) {
                pugi::xml_node back_edges = m_xml_node.parent().child("back_edges");
                if (!back_edges) {
                    back_edges = m_xml_node.parent().insert_child_after("back_edges", port_map);
                }
                pugi::xml_node edge = back_edges.append_child("edge");
                edge.append_attribute("from-layer").set_value(result_mapping[merged_input->m_body_value_index].c_str());
                edge.append_attribute("to-layer")
                    .set_value(parameter_mapping[merged_input->m_body_parameter_index].c_str());
            }
        }
    }

    void output_descriptions_on_adapter(
        const std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>>& output_descriptions,
        const uint32_t& input_count,
        const std::vector<std::string>& result_mapping,
        pugi::xml_node& port_map,
        const std::string& portmap_name) {
        OPENVINO_ASSERT(!result_mapping.empty(), "No results found in body Model.");

        if (!port_map) {
            port_map = m_xml_node.parent().insert_child_before(portmap_name.c_str(), m_xml_node.parent().first_child());
        }

        for (const auto& output_description : output_descriptions) {
            pugi::xml_node output = port_map.append_child("output");
            output.append_attribute("external_port_id")
                .set_value(static_cast<unsigned long long>(input_count + output_description->m_output_index));
            output.append_attribute("internal_layer_id")
                .set_value(result_mapping[output_description->m_body_value_index].c_str());

            if (auto concat_output =
                    ov::as_type_ptr<ov::op::util::SubGraphOp::ConcatOutputDescription>(output_description)) {
                output.prepend_attribute("axis").set_value(static_cast<long long>(concat_output->m_axis));
                output.append_attribute("start").set_value(static_cast<long long>(concat_output->m_start));
                output.append_attribute("end").set_value(static_cast<long long>(concat_output->m_end));
                output.append_attribute("stride").set_value(static_cast<long long>(concat_output->m_stride));
                output.append_attribute("part_size").set_value(static_cast<long long>(concat_output->m_part_size));
            }
        }
    }

    void special_body_ports_on_adapter(const ov::op::v5::Loop::SpecialBodyPorts& special_body_ports,
                                       const std::vector<std::string>& parameter_mapping,
                                       const std::vector<std::string>& result_mapping,
                                       pugi::xml_node& port_map) {
        OPENVINO_ASSERT(port_map, "port_map section not found, purpose attribute cannot be added.");

        if (special_body_ports.current_iteration_input_idx != -1) {
            pugi::xml_node iter_input = port_map.append_child("input");
            iter_input.append_attribute("external_port_id").set_value("-1");
            iter_input.append_attribute("internal_layer_id")
                .set_value(parameter_mapping[special_body_ports.current_iteration_input_idx].c_str());
            iter_input.append_attribute("purpose").set_value("current_iteration");
        }

        if (special_body_ports.body_condition_output_idx != -1) {
            pugi::xml_node exec_output = port_map.append_child("output");
            exec_output.append_attribute("external_port_id").set_value("-1");
            exec_output.append_attribute("internal_layer_id")
                .set_value(result_mapping[special_body_ports.body_condition_output_idx].c_str());
            exec_output.append_attribute("purpose").set_value("execution_condition");
        }
    }

public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  ConstantWriter& constant_write_handler,
                  int64_t version,
                  bool deterministic = false,
                  bool compress_to_fp16 = false,
                  ov::element::Type output_element_type = ov::element::dynamic,
                  bool data_is_temporary = false)
        : m_xml_node(data),
          m_node_type_name(node_type_name),
          m_constant_write_handler(constant_write_handler),
          m_version(version),
          m_deterministic(deterministic),
          m_compress_to_fp16(compress_to_fp16),
          m_output_element_type(output_element_type),
          m_data_is_temporary(data_is_temporary) {}

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override {
        using BodyTargetNames = std::tuple<std::string, std::string, std::vector<std::string>>;

        const std::vector<BodyTargetNames> body_names = {
            BodyTargetNames{"body", "port_map", {"input_descriptions", "output_descriptions", "special_body_ports"}},
            BodyTargetNames{"then_body", "then_port_map", {"then_inputs", "then_outputs"}},
            BodyTargetNames{"else_body", "else_port_map", {"else_inputs", "else_outputs"}}};
        BodyTargetNames bnames;
        bool is_body_target = false;
        for (const auto& _body_target : body_names) {
            if (m_xml_node.parent().child(std::get<0>(_body_target).c_str())) {
                auto vec_names = std::get<2>(_body_target);

                if (std::find(vec_names.begin(), vec_names.end(), name) != vec_names.end()) {
                    is_body_target = true;
                    bnames = _body_target;
                    break;
                }
            }
        }
        if (!is_body_target) {
            std::string id = "input_descriptions";
            std::string od = "output_descriptions";
            const auto& id_pos = name.find("input_descriptions");
            const auto& od_pos = name.find("output_descriptions");
            auto id_str = name;
            size_t body_id;
            if (id_pos != std::string::npos) {
                id_str.erase(id_pos, id.length());
                (void)std::stoi(id_str, &body_id);
                is_body_target = true;
            } else if (od_pos != std::string::npos) {
                id_str.erase(od_pos, od.length());
                (void)std::stoi(id_str, &body_id);
                is_body_target = true;
            }
            if (is_body_target) {
                auto body_name = "body" + id_str;
                if (m_xml_node.parent().child(body_name.c_str())) {
                    bnames = BodyTargetNames{body_name,
                                             "port_map" + id_str,
                                             {"input_descriptions" + id_str, "output_descriptions" + id_str}};
                } else {
                    is_body_target = false;
                }
            }
        }
        if (is_body_target) {
            const auto& body_name = std::get<0>(bnames);
            const auto& portmap_name = std::get<1>(bnames);
            std::vector<std::string> result_mapping =
                map_type_from_body(m_xml_node.parent(), "Result", m_version, body_name);
            std::vector<std::string> parameter_mapping =
                map_type_from_body(m_xml_node.parent(), "Parameter", m_version, body_name);

            pugi::xml_node port_map = m_xml_node.parent().child(portmap_name.c_str());
            // Bodies can be without parameters(dependig on constants), but can not be without results
            OPENVINO_ASSERT(!result_mapping.empty(), "No results found in body Model.");
            // TI, Loop do not have attributtes as regular ops, it is necessary to append "port_map" and
            // "back_edges" to layer above (m_xml_node.parent()) as in ngfunction_2_ir() layer (here "m_xml_node")
            // with empty attributes is removed.
            if (const auto& a = ov::as_type<ov::AttributeAdapter<
                    std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>>>>(&adapter)) {
                input_descriptions_on_adapter(a->get(), parameter_mapping, result_mapping, port_map, portmap_name);
            } else if (const auto& a = ov::as_type<ov::AttributeAdapter<
                           std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>>>>(&adapter)) {
                uint32_t op_input_count = 0;
                for (auto c = m_xml_node.parent().child("input").first_child(); !c.empty(); c = c.next_sibling()) {
                    op_input_count++;
                }
                output_descriptions_on_adapter(a->get(), op_input_count, result_mapping, port_map, portmap_name);
            } else if (const auto& a =
                           ov::as_type<ov::AttributeAdapter<ov::op::v5::Loop::SpecialBodyPorts>>(&adapter)) {
                special_body_ports_on_adapter(a->get(), parameter_mapping, result_mapping, port_map);
            }
        } else if (const auto& a =
                       ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>>(&adapter)) {
            m_xml_node.append_attribute(name.c_str()).set_value(a->get()->get_info().variable_id.c_str());
        } else if (ov::is_type<ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>>(&adapter) ||
                   ov::is_type<ov::AttributeAdapter<std::shared_ptr<ov::SharedStringAlignedBuffer>>>(&adapter)) {
            if (name == "value" && translate_type_name(m_node_type_name) == "Const") {
                auto a1 = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>>(&adapter);
                auto a2 = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::SharedStringAlignedBuffer>>>(&adapter);
                size_t new_size = 0;
                size_t inter_size = 0;
                // write a header of packed string tensor
                std::shared_ptr<uint8_t> header_ptr = nullptr;
                size_t header_size = 0;
                if (a1) {
                    a1->get_header(header_ptr, header_size);
                } else {
                    a2->get_header(header_ptr, header_size);
                }

                int64_t offset = m_constant_write_handler.write(
                    reinterpret_cast<const char*>(header_ptr.get()),
                    header_size,
                    inter_size,
                    m_compress_to_fp16,
                    m_output_element_type,
                    true);  // header_ptr is allocated in AttributeAdapter that has limited life time
                new_size += inter_size;

                // write raw strings part
                size_t num_elements = 0;
                if (a1) {
                    num_elements = a1->get()->get_num_elements();
                } else {
                    num_elements = a2->get()->get_num_elements();
                }
                for (size_t ind = 0; ind < num_elements; ++ind) {
                    const char* raw_string_ptr;
                    size_t raw_string_size;
                    if (a1) {
                        a1->get_raw_string_by_index(raw_string_ptr, raw_string_size, ind);
                    } else {
                        a2->get_raw_string_by_index(raw_string_ptr, raw_string_size, ind);
                    }

                    m_constant_write_handler.write(raw_string_ptr,
                                                   raw_string_size,
                                                   inter_size,
                                                   m_compress_to_fp16,
                                                   m_output_element_type,
                                                   m_data_is_temporary);

                    new_size += inter_size;
                }
                m_xml_node.append_attribute("offset").set_value(static_cast<unsigned long long>(offset));
                m_xml_node.append_attribute("size").set_value(static_cast<unsigned long long>(new_size));
            }
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>>(&adapter)) {
            if (name == "value" && translate_type_name(m_node_type_name) == "Const") {
                const int64_t size = a->get()->size();
                size_t new_size;
                int64_t offset = m_constant_write_handler.write(static_cast<const char*>(a->get()->get_ptr()),
                                                                size,
                                                                new_size,
                                                                m_compress_to_fp16,
                                                                m_output_element_type,
                                                                m_data_is_temporary);

                m_xml_node.append_attribute("offset").set_value(static_cast<unsigned long long>(offset));
                m_xml_node.append_attribute("size").set_value(static_cast<unsigned long long>(new_size));
            }
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>>(&adapter)) {
            const auto& attrs = a->get();

            // Update type and version attributes
            pugi::xml_node layer = m_xml_node.parent();

            auto type_attr = layer.attribute("type");
            auto version_attr = layer.attribute("version");

            type_attr.set_value(attrs.get_type_name().c_str());

            if (!attrs.get_opset_name().empty()) {
                version_attr.set_value(attrs.get_opset_name().c_str());
            } else {
                layer.remove_attribute("version");
            }

            // Update node attributes in data field
            for (const auto& attr : attrs) {
                m_xml_node.append_attribute(attr.first.c_str()).set_value(attr.second.c_str());
            }
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::element::TypeVector>>(&adapter)) {
            const auto& attrs = a->get();
            m_xml_node.append_attribute(name.c_str()).set_value(join(attrs).c_str());
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
            const auto& attrs = a->get();
            auto shape_str = attrs.to_string();
            if (shape_str[0] == '[' && shape_str[shape_str.size() - 1] == ']')
                shape_str = shape_str.substr(1, shape_str.size() - 2);
            m_xml_node.append_attribute(name.c_str()).set_value(shape_str.c_str());
        } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
            const auto& attrs = a->get();
            std::stringstream dim_str_stream;
            dim_str_stream << attrs;
            auto dim_str = dim_str_stream.str();
            if (dim_str[0] == '{' && dim_str[dim_str.size() - 1] == '}')
                dim_str = dim_str.substr(1, dim_str.size() - 2);
            m_xml_node.append_attribute(name.c_str()).set_value(dim_str.c_str());
        } else {
            OPENVINO_THROW("Unsupported attribute type for serialization: ", name);
        }
    }

    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override {
        std::string value;
        if (m_compress_to_fp16 && name == "element_type") {
            value = ov::as_string(static_cast<ov::element::Type_t>(ov::element::f16));
        } else {
            value = adapter.get();
        }
        m_xml_node.append_attribute(name.c_str()).set_value(value.c_str());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(static_cast<long long>(adapter.get()));
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(adapter.get());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override {
        m_xml_node.append_attribute(name.c_str()).set_value(create_atribute_list(adapter).c_str());
    }
    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override {
        if (name.find("body") != std::string::npos) {
            // name that contains subgraphs: body{n}, then_body, else_body
            // TI, Loop do not have attributtes as regular ops, it is necessary to append "body"
            // to layer above (m_xml_node.parent()) as in ngfunction_2_ir() layer (m_xml_node) with empty attributes
            // is removed.
            pugi::xml_node xml_body = m_xml_node.parent().append_child(name.c_str());
            ngfunction_2_ir(xml_body, *adapter.get(), m_constant_write_handler, m_version, m_deterministic);
            xml_body.remove_attribute("name");
            xml_body.remove_attribute("version");
        } else if (name == "net") {
            ngfunction_2_ir(m_xml_node, *adapter.get(), m_constant_write_handler, m_version, m_deterministic);
        } else {
            OPENVINO_THROW("Unsupported Model name.");
        }
    }
};

const std::unordered_map<ov::Node*, int> create_layer_ids(const ov::Model& model) {
    std::unordered_map<ov::Node*, int> layer_ids;
    int id = 0;
    for (const auto& node : model.get_ordered_ops()) {
        layer_ids[node.get()] = id++;
    }
    return layer_ids;
}

const std::vector<Edge> create_edge_mapping(const std::unordered_map<ov::Node*, int>& layer_ids,
                                            const ov::Model& model) {
    std::vector<Edge> edges;
    for (const auto& node : model.get_ordered_ops()) {
        if (ov::op::util::is_parameter(node)) {
            continue;
        }

        for (const auto& i : node->inputs()) {
            auto source_output = i.get_source_output();
            auto source_node = source_output.get_node();
            auto current_node = i.get_node();

            OPENVINO_ASSERT(layer_ids.find(source_node) != layer_ids.end(), "Internal error");
            OPENVINO_ASSERT(layer_ids.find(current_node) != layer_ids.end(), "Internal error");

            Edge e{};
            e.from_layer = layer_ids.find(source_node)->second;
            e.from_port = static_cast<int>(source_node->get_input_size() + source_output.get_index());
            e.to_layer = layer_ids.find(current_node)->second;
            e.to_port = static_cast<int>(i.get_index());
            edges.push_back(e);
        }
    }
    std::sort(begin(edges), end(edges), [](const Edge& a, const Edge& b) -> bool {
        return a.from_layer < b.from_layer;
    });
    return edges;
}

std::string get_opset_name(const ov::Node* n) {
    OPENVINO_ASSERT(n != nullptr);

    // TODO: remove it one day: try to find opset name from RT info
    // It's a dirty hack to TypeRelaxed and similar template internal operations
    auto opset_it = n->get_rt_info().find("opset");
    if (opset_it != n->get_rt_info().end()) {
        if (opset_it->second.is<std::string>()) {
            return opset_it->second.as<std::string>();
        }
    }

    return n->get_type_info().version_id == nullptr ? "experimental" : n->get_type_info().version_id;
}

std::string get_precision_name(const ov::element::Type& elem_type) {
    switch (elem_type) {
    case ::ov::element::Type_t::dynamic:
        return "UNSPECIFIED";
    case ::ov::element::Type_t::f16:
        return "FP16";
    case ::ov::element::Type_t::f32:
        return "FP32";
    case ::ov::element::Type_t::bf16:
        return "BF16";
    case ::ov::element::Type_t::f64:
        return "FP64";
    case ::ov::element::Type_t::i4:
        return "I4";
    case ::ov::element::Type_t::i8:
        return "I8";
    case ::ov::element::Type_t::i16:
        return "I16";
    case ::ov::element::Type_t::i32:
        return "I32";
    case ::ov::element::Type_t::i64:
        return "I64";
    case ::ov::element::Type_t::u4:
        return "U4";
    case ::ov::element::Type_t::u8:
        return "U8";
    case ::ov::element::Type_t::u16:
        return "U16";
    case ::ov::element::Type_t::u32:
        return "U32";
    case ::ov::element::Type_t::u64:
        return "U64";
    case ::ov::element::Type_t::u1:
        return "BIN";
    case ::ov::element::Type_t::boolean:
        return "BOOL";
    case ::ov::element::Type_t::nf4:
        return "NF4";
    case ::ov::element::Type_t::f8e4m3:
        return "F8E4M3";
    case ::ov::element::Type_t::f8e5m2:
        return "F8E5M2";
    case ::ov::element::Type_t::string:
        return "STRING";
    case ::ov::element::Type_t::f4e2m1:
        return "F4E2M1";
    case ::ov::element::Type_t::f8e8m0:
        return "F8E8M0";
    default:
        OPENVINO_THROW("Unsupported precision: ", elem_type);
    }
}

std::string escape_delim(const std::string& name, const char delim = ',') {
    std::string result_name = name;
    const std::string escaped_delim = std::string("\\") + delim;
    size_t index = result_name.find(delim, 0);
    while (index != std::string::npos) {
        result_name.replace(index, 1, escaped_delim);
        index = result_name.find(delim, index + 2);
    }
    return result_name;
}

void visit_exec_graph_node(pugi::xml_node& layer, const ov::Node* n) {
    auto data = layer.child("data");
    for (const auto& param : n->get_rt_info()) {
        if (param.second.is<std::string>()) {
            const std::string& name = param.first;
            const std::string& value = param.second.as<std::string>();

            if (name == "layerType") {
                layer.attribute("type").set_value(value.c_str());
                continue;
            }

            data.append_attribute(name.c_str()).set_value(value.c_str());
        }
    }
}

bool is_exec_graph(const ov::Model& model) {
    // go over all operations and check whether performance stat is set
    for (const auto& op : model.get_ops()) {
        const auto& rtInfo = op->get_rt_info();
        if (rtInfo.find("execTimeMcs") != rtInfo.end()) {
            return true;
        }
    }
    return false;
}

class PaddingsFixer {
private:
    ov::Node* m_node;

    ov::OutputVector m_parameters;
    std::shared_ptr<ov::Node> m_cloned_node;

    const std::set<ov::op::PadType> pad_agnostic_types = {
        ov::op::PadType::SAME_LOWER,
        ov::op::PadType::SAME_UPPER,
        ov::op::PadType::VALID,
        ov::op::PadType::AUTO,
    };

    template <class T, class P>
    void clone_op_and_fix_paddings(const T* op) {
        for (const auto& input : op->inputs()) {
            m_parameters.emplace_back(
                std::make_shared<ov::opset1::Parameter>(input.get_element_type(), input.get_partial_shape()));
        }
        m_cloned_node = op->clone_with_new_inputs(m_parameters);
        auto typed_cloned_node = ov::as_type_ptr<T>(m_cloned_node);
        OPENVINO_ASSERT(typed_cloned_node);
        typed_cloned_node->set_pads_begin(P(op->get_pads_begin().size(), 0));
        typed_cloned_node->set_pads_end(P(op->get_pads_end().size(), 0));
        m_node = m_cloned_node.get();
    }

public:
    ov::Node* get_node() {
        return m_node;
    }

    explicit PaddingsFixer(ov::Node* node) : m_node(node) {
        if (auto op = ov::as_type<ov::opset1::Convolution>(node)) {
            if (pad_agnostic_types.count(op->get_auto_pad())) {
                clone_op_and_fix_paddings<ov::opset1::Convolution, ov::CoordinateDiff>(op);
            }
        } else if (auto op = ov::as_type<ov::opset1::GroupConvolution>(node)) {
            if (pad_agnostic_types.count(op->get_auto_pad())) {
                clone_op_and_fix_paddings<ov::opset1::GroupConvolution, ov::CoordinateDiff>(op);
            }
        } else if (auto op = ov::as_type<ov::opset1::ConvolutionBackpropData>(node)) {
            if (pad_agnostic_types.count(op->get_auto_pad())) {
                clone_op_and_fix_paddings<ov::opset1::ConvolutionBackpropData, ov::CoordinateDiff>(op);
            }
        } else if (auto op = ov::as_type<ov::opset1::GroupConvolutionBackpropData>(node)) {
            if (pad_agnostic_types.count(op->get_auto_pad())) {
                clone_op_and_fix_paddings<ov::opset1::GroupConvolutionBackpropData, ov::CoordinateDiff>(op);
            }
        } else if (auto op = ov::as_type<ov::op::util::DeformableConvolutionBase>(node)) {
            if (pad_agnostic_types.count(op->get_auto_pad())) {
                clone_op_and_fix_paddings<ov::op::util::DeformableConvolutionBase, ov::CoordinateDiff>(op);
            }
        } else if (auto op = ov::as_type<ov::opset1::BinaryConvolution>(node)) {
            if (pad_agnostic_types.count(op->get_auto_pad())) {
                clone_op_and_fix_paddings<ov::opset1::BinaryConvolution, ov::CoordinateDiff>(op);
            }
        } else if (auto op = ov::as_type<ov::op::util::AvgPoolBase>(node)) {
            if (pad_agnostic_types.count(op->get_auto_pad())) {
                clone_op_and_fix_paddings<ov::op::util::AvgPoolBase, ov::Shape>(op);
            }
        } else if (auto op = ov::as_type<ov::op::util::MaxPoolBase>(node)) {
            if (pad_agnostic_types.count(op->get_auto_pad())) {
                clone_op_and_fix_paddings<ov::op::util::MaxPoolBase, ov::Shape>(op);
            }
        }
    }
};

// Substitute a Constant node instead of a node by calling node->constant_fold if 'postponed_constant' rt_info attribute
// is present in the node
class PostponedConstantReplacer {
private:
    ov::Node* m_node;
    std::shared_ptr<ov::Node> m_constant;

public:
    ov::Node* get_node() {
        return m_node;
    }

    bool data_is_temporary() const {
        return m_constant != nullptr;
    }

    PostponedConstantReplacer(ov::Node* node) : m_node(node), m_constant() {
        if (node->get_rt_info().count("postponed_constant")) {
            OPENVINO_ASSERT(node->get_output_size() == 1);
            ov::OutputVector outputs(1);
            OPENVINO_ASSERT(
                node->constant_fold(outputs, node->input_values()),
                "Node with set `postponed_constant` attribute cannot be fold to constant when saving model to IR file");
            m_constant = outputs[0].get_node_shared_ptr();
            m_node = m_constant.get();
            m_node->set_friendly_name(node->get_friendly_name());
        }
    }
};

bool is_correct_tag_name(const std::string& name) {
    if (name.length() == 0) {
        return false;
    }
    if (!std::all_of(name.begin(), name.end(), [](const int c) {
            return std::isalnum(c) || (c == '_') || (c == '-') || (c == '.');
        })) {
        return false;
    }
    if (std::isalpha(name[0]) == false && name[0] != '_') {
        return false;
    }
    if (name.length() >= 3 && (name[0] == 'X' || name[0] == 'x') && (name[1] == 'M' || name[1] == 'm') &&
        (name[2] == 'l' || name[2] == 'L')) {
        return false;
    }
    return true;
}

void serialize_rt_info(pugi::xml_node& root, const std::string& name, const ov::Any& data) {
    pugi::xml_node child;
    if (is_correct_tag_name(name)) {
        child = root.append_child(name.c_str());
    } else {
        // Name may brake XML-naming specification, so better to store it as an attribute of typical
        // node
        child = root.append_child("info");
        child.append_attribute("name").set_value(name.c_str());
    }
    if (data.is<std::shared_ptr<ov::Meta>>()) {
        auto meta = data.as<std::shared_ptr<ov::Meta>>();
        do {
            if (auto meta_with_pugixml_node = std::dynamic_pointer_cast<ov::MetaDataWithPugixml>(meta)) {
                if (auto pugi_node = meta_with_pugixml_node->get_pugi_node()) {
                    root.remove_child(child);
                    auto added_node = root.append_copy(pugi_node);
                    OPENVINO_ASSERT(added_node, "Cannot add pugixml node with name: ", name);
                    added_node.set_name(name.c_str());
                    break;
                }
            }
            // Meta in ov::Meta cannot be accessed by MetaDataWithPugixml::get_pugi_node. Read it as ov::AnyMap
            ov::AnyMap& map = *meta;
            for (const auto& it : map) {
                serialize_rt_info(child, it.first, it.second);
            }
        } while (false);
    } else if (data.is<ov::AnyMap>()) {
        const ov::AnyMap& any_map = data.as<ov::AnyMap>();
        for (const auto& it : any_map) {
            serialize_rt_info(child, it.first, it.second);
        }
    } else {
        std::string value = data.as<std::string>();
        child.append_attribute("value").set_value(value.c_str());
    }
}

void ngfunction_2_ir(pugi::xml_node& netXml,
                     const ov::Model& model,
                     ConstantWriter& constant_node_write_handler,
                     int64_t version,
                     bool deterministic) {
    // If determinism is not required, include auto-generated names into xml
    // model name is not critical for hash computing
    if (!deterministic) {
        netXml.append_attribute("name").set_value(model.get_friendly_name().c_str());
    }
    netXml.append_attribute("version").set_value(static_cast<long long>(version));
    pugi::xml_node layers = netXml.append_child("layers");

    const std::unordered_map<ov::Node*, int> layer_ids = create_layer_ids(model);

    const bool exec_graph = is_exec_graph(model);

    auto sorted_ops = model.get_ordered_ops();

    // get_ordered_ops() returns operations after a topological sort. The topological sort reverses order of Parameters
    // and Results. So we need to put them into sorted_ops separately to ensure correct order of inputs and outputs.
    {
        std::vector<std::shared_ptr<ov::Node>> result;
        result.reserve(sorted_ops.size());
        for (const auto& param : model.get_parameters()) {
            result.emplace_back(param);
        }
        auto model_sinks = model.get_sinks();
        for (auto&& node : sorted_ops) {
            if (!ov::op::util::is_parameter(node) && !ov::op::util::is_output(node) &&
                std::find(model_sinks.begin(), model_sinks.end(), node) == model_sinks.end())
                result.emplace_back(node);
        }
        for (const auto& sink : model.get_sinks()) {
            result.emplace_back(sink);
        }
        for (const auto& res : model.get_results()) {
            result.emplace_back(res);
        }
        sorted_ops = std::move(result);
    }

    for (const auto& n : sorted_ops) {
        ov::Node* node = n.get();
        int node_id{};
        {
            auto it = layer_ids.find(node);
            OPENVINO_ASSERT(it != layer_ids.end(), "Internal error");
            node_id = it->second;
        }
        PostponedConstantReplacer modified_node(node);
        node = modified_node.get_node();

        const std::string& node_type_name{node->get_type_name()};

        // <layers>
        pugi::xml_node layer = layers.append_child("layer");
        layer.append_attribute("id").set_value(node_id);
        // If determinism is not required, include auto-generated names into xml
        // layer name is not critical for hash computing
        if (!deterministic) {
            layer.append_attribute("name").set_value(node->get_friendly_name().c_str());
        }
        layer.append_attribute("type").set_value(translate_type_name(node_type_name).c_str());
        if (!exec_graph) {
            layer.append_attribute("version").set_value(get_opset_name(node).c_str());
        }

        // <layers/data> general attributes
        pugi::xml_node data = layer.append_child("data");

        auto append_runtime_info = [](pugi::xml_node& node, ov::RTMap& attributes) {
            pugi::xml_node rt_node = node.append_child("rt_info");
            bool has_attrs = false;
            for (auto& item : attributes) {
                if (item.second.is<ov::RuntimeAttribute>()) {
                    auto attribute_node = rt_node.append_child("attribute");
                    auto& rt_attribute = item.second.as<ov::RuntimeAttribute>();
                    const auto& type_info = rt_attribute.get_type_info();
                    attribute_node.append_attribute("name").set_value(type_info.name);
                    attribute_node.append_attribute("version").set_value(type_info.get_version().c_str());
                    rt_info::RTInfoSerializer serializer(attribute_node);
                    if (!rt_attribute.visit_attributes(serializer)) {
                        rt_node.remove_child(attribute_node);
                    } else {
                        has_attrs = true;
                    }
                }
            }
            if (!has_attrs) {
                node.remove_child(rt_node);
            }
        };

        if (version >= 11) {
            append_runtime_info(layer, node->get_rt_info());
        }

        int port_id = 0;
        // <layers/input>
        if (node->get_input_size() > 0) {
            pugi::xml_node input = layer.append_child("input");
            for (auto& i : node->inputs()) {
                // WA for LSTMCellv0, peephole input shall not be serialized
                if (i.get_index() == 6 && ov::as_type<ov::opset1::LSTMCell>(node)) {
                    port_id++;
                    continue;
                }

                pugi::xml_node port = input.append_child("port");
                port.append_attribute("id").set_value(port_id++);

                const auto& rt_info = i.get_tensor().get_rt_info();
                auto port_element_type =
                    is_fp16_compression_postponed(rt_info) ? ov::element::f16 : i.get_element_type();

                port.append_attribute("precision").set_value(get_precision_name(port_element_type).c_str());
                for (const auto& d : i.get_partial_shape()) {
                    pugi::xml_node dim = port.append_child("dim");
                    if (d.is_dynamic()) {
                        dim.append_child(pugi::xml_node_type::node_pcdata).set_value("-1");
                    } else {
                        dim.append_child(pugi::xml_node_type::node_pcdata)
                            .set_value(std::to_string(d.get_length()).c_str());
                    }
                }
                if (version >= 11)
                    append_runtime_info(port, i.get_rt_info());
            }

            if (node_type_name == "TensorIterator" || node_type_name == "Loop") {
                layer.prepend_move(input);
            }
        }
        // <layers/output>
        if (node->get_output_size() > 0) {
            auto serialize_tensor_names = [](const std::unordered_set<std::string>& names) -> std::string {
                auto sorted_names = std::vector<std::string>(names.begin(), names.end());
                std::sort(sorted_names.begin(), sorted_names.end());

                std::string serialized_names;
                for (const auto& name : sorted_names) {
                    if (!serialized_names.empty())
                        serialized_names += ",";
                    serialized_names += escape_delim(name);
                }
                return serialized_names;
            };

            if (ov::op::util::is_output(node)) {
                if (version > 10 && !deterministic) {
                    // Not serialize output names for deterministic mode (hash) computation as it is optional
                    // attribute for v11 and not affect on model structure or how it works
                    if (const auto& names = ov::descriptor::get_assigned_names(node->get_output_tensor(0));
                        !names.empty()) {
                        layer.append_attribute("output_names").set_value(serialize_tensor_names(names).c_str());
                    }
                }
            } else {
                pugi::xml_node output = layer.append_child("output");
                for (auto& o : node->outputs()) {
                    pugi::xml_node port = output.append_child("port");
                    port.append_attribute("id").set_value(port_id++);

                    const auto& rt_info = o.get_tensor().get_rt_info();
                    auto port_element_type =
                        is_fp16_compression_postponed(rt_info) ? ov::element::f16 : o.get_element_type();

                    port.append_attribute("precision").set_value(get_precision_name(port_element_type).c_str());

                    if (const auto& tensor_names = o.get_tensor().get_names(); !tensor_names.empty()) {
                        port.append_attribute("names").set_value(serialize_tensor_names(tensor_names).c_str());
                    }

                    for (const auto& d : o.get_partial_shape()) {
                        pugi::xml_node dim = port.append_child("dim");
                        if (d.is_dynamic()) {
                            dim.append_child(pugi::xml_node_type::node_pcdata).set_value("-1");
                        } else {
                            dim.append_child(pugi::xml_node_type::node_pcdata)
                                .set_value(std::to_string(d.get_length()).c_str());
                        }
                    }
                    if (version >= 11)
                        append_runtime_info(port, o.get_rt_info());
                }
                if (node_type_name == "TensorIterator" || node_type_name == "Loop") {
                    layer.insert_move_after(output, layer.first_child());
                }
            }
        }

        // fill <data> general attributes
        {
            bool compress_to_fp16 = false;
            ov::element::Type output_element_type = ov::element::dynamic;
            if (is_fp16_compression_postponed(node->get_rt_info())) {
                compress_to_fp16 = true;
                output_element_type = node->get_output_element_type(0);
            }
            // Backward compatibility: clear padding values for nodes with auto_pad
            PaddingsFixer fixed_node(node);
            XmlSerializer visitor(data,
                                  node_type_name,
                                  constant_node_write_handler,
                                  version,
                                  deterministic,
                                  compress_to_fp16,
                                  output_element_type,
                                  modified_node.data_is_temporary());
            OPENVINO_ASSERT(fixed_node.get_node()->visit_attributes(visitor), "Visitor API is not supported in ", node);
        }
        rt_info::XmlSerializer{data}.serialize(node->get_rt_info());

        if (exec_graph) {
            visit_exec_graph_node(layer, node);
        }

        const bool data_attr_size = data.attributes().begin() == data.attributes().end();
        if (data_attr_size) {
            layer.remove_child(data);
        }
    }
    // <edges>
    const std::vector<Edge> edge_mapping = create_edge_mapping(layer_ids, model);
    pugi::xml_node edges = netXml.append_child("edges");
    auto ordered_ops = model.get_ordered_ops();
    for (auto e : edge_mapping) {
        // WA for LSTMCellv0, peephole input shall not be serialized
        if (e.to_port == 6) {
            const auto& type_info = ordered_ops[e.to_layer]->get_type_info();
            if (!strcmp(type_info.name, "LSTMCell")) {
                continue;
            }
        }
        pugi::xml_node edge = edges.append_child("edge");
        edge.append_attribute("from-layer").set_value(e.from_layer);
        edge.append_attribute("from-port").set_value(e.from_port);
        edge.append_attribute("to-layer").set_value(e.to_layer);
        edge.append_attribute("to-port").set_value(e.to_port);
    }

    // Serialize rt info
    pugi::xml_node rt_info_node = netXml.append_child("rt_info");
    for (const auto& it : model.get_rt_info()) {
        // Skip IR version
        if (it.first == "version" || it.first == "__weights_path")
            continue;
        serialize_rt_info(rt_info_node, it.first, it.second);
    }
}

std::string valid_xml_path(const std::string& path) {
    OPENVINO_ASSERT(path.length() > 4, "Path for xml file is too short: \"" + path + "\"");

    const char* const extension = ".xml";
    const bool has_xml_extension = path.rfind(extension) == path.size() - std::strlen(extension);
    OPENVINO_ASSERT(has_xml_extension,
                    "Path for xml file doesn't contains file name with 'xml' extension: \"" + path + "\"");
    return path;
}

std::string provide_bin_path(const std::string& xmlPath, const std::string& binPath) {
    if (!binPath.empty()) {
        return binPath;
    }
    assert(xmlPath.size() > 4);  // should be check by valid_xml_path
    std::string bestPath = xmlPath;
    const char* const extension = "bin";
    const auto ext_size = std::strlen(extension);
    bestPath.replace(bestPath.size() - ext_size, ext_size, extension);
    return bestPath;
}

void serializeFunc(std::ostream& xml_file,
                   std::ostream& bin_file,
                   std::shared_ptr<ov::Model> model,
                   ov::pass::Serialize::Version ver,
                   bool deterministic = false) {
    auto version = static_cast<int64_t>(ver);

    auto& rt_info = model->get_rt_info();
    if (rt_info.count("version")) {
        version = rt_info.at("version").as<int64_t>();
    }

    if (version != static_cast<int64_t>(ver) && ver != ov::pass::Serialize::Version::UNSPECIFIED)
        OPENVINO_THROW("Cannot serialize Model to incompatible IR version");

    if (version == static_cast<int64_t>(ov::pass::Serialize::Version::UNSPECIFIED))
        version = static_cast<int64_t>(ov::pass::Serialize::Version::IR_V11);

    if (version != static_cast<int64_t>(ov::pass::Serialize::Version::IR_V10) &&
        version != static_cast<int64_t>(ov::pass::Serialize::Version::IR_V11)) {
        OPENVINO_THROW("Unsupported version");
    }
    std::string name = "net";
    pugi::xml_document xml_doc;
    pugi::xml_node net_node = xml_doc.append_child(name.c_str());
    ConstantWriter constant_write_handler(bin_file);
    XmlSerializer visitor(net_node, name, constant_write_handler, version, deterministic);
    visitor.on_attribute(name, model);

    xml_doc.save(xml_file);
    xml_file.flush();
    bin_file.flush();
};

}  // namespace

namespace ov {
bool pass::Serialize::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(Serialize);

    model->validate_nodes_and_infer_types();

    // TODO xxx-105807: if rt_info is set in python api as a string ['precise_0'] = '',
    //  we need to convert value to a class in order to have rt_info in the IR. The code below will convert
    // ['precise_0'] = '' into => rt_info['precise_0'] = DisableFP16Compression{}
    for (auto& node : model->get_ops())
        if (fp16_compression_is_disabled(node))
            disable_fp16_compression(node);

    if (m_xmlFile && m_binFile) {
        serializeFunc(*m_xmlFile, *m_binFile, model, m_version);
    } else {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const auto& xmlPath_ref = ov::util::string_to_wstring(m_xmlPath);
        const auto& binPath_ref = ov::util::string_to_wstring(m_binPath);
        std::string message_bin = "Can't open bin file.";
        std::string message_xml = "Can't open xml file.";
#else
        const auto& xmlPath_ref = m_xmlPath;
        const auto& binPath_ref = m_binPath;
        std::string message_bin = "Can't open bin file: \"" + binPath_ref + "\"";
        std::string message_xml = "Can't open xml file: \"" + xmlPath_ref + "\"";
#endif
        auto xmlDir = ov::util::get_directory(xmlPath_ref);
        if (xmlDir != xmlPath_ref)
            ov::util::create_directory_recursive(xmlDir);

        std::ofstream bin_file(binPath_ref, std::ios::out | std::ios::binary);
        OPENVINO_ASSERT(bin_file, message_bin);

        // create xml file
        std::ofstream xml_file(xmlPath_ref, std::ios::out);
        OPENVINO_ASSERT(xml_file, message_xml);

        try {
            serializeFunc(xml_file, bin_file, model, m_version);
        } catch (const ov::AssertFailure&) {
            // optimization decision was made to create .bin file upfront and
            // write to it directly instead of buffering its content in memory,
            // hence we need to delete it here in case of failure
            xml_file.close();
            bin_file.close();
            std::ignore = std::remove(m_xmlPath.c_str());
            std::ignore = std::remove(m_binPath.c_str());
            throw;
        }
    }

    // Return false because we didn't change ov Model
    return false;
}

pass::Serialize::Serialize(std::ostream& xmlFile, std::ostream& binFile, pass::Serialize::Version version)
    : m_xmlFile{&xmlFile},
      m_binFile{&binFile},
      m_xmlPath{},
      m_binPath{},
      m_version{version} {}

pass::Serialize::Serialize(const std::string& xmlPath, const std::string& binPath, pass::Serialize::Version version)
    : m_xmlFile{nullptr},
      m_binFile{nullptr},
      m_xmlPath{valid_xml_path(xmlPath)},
      m_binPath{provide_bin_path(xmlPath, binPath)},
      m_version{version} {}

pass::StreamSerialize::StreamSerialize(std::ostream& stream,
                                       const std::function<void(std::ostream&)>& custom_data_serializer,
                                       const std::function<std::string(const std::string&)>& cache_encrypt,
                                       Serialize::Version version)
    : m_stream(stream),
      m_custom_data_serializer(custom_data_serializer),
      m_cache_encrypt(cache_encrypt),
      m_version(version) {
    if (version != Serialize::Version::UNSPECIFIED && version != Serialize::Version::IR_V10 &&
        version != Serialize::Version::IR_V11) {
        OPENVINO_THROW("Unsupported version");
    }
}

bool pass::StreamSerialize::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(StreamSerialize);
    /*
        Format:
        [ DataHeader  ]
        [ Custom data ]
        [    Blobs    ]
        [     IR      ]
    */
    DataHeader hdr = {};

    auto writeHeader = [this](const DataHeader& hdr) {
        m_stream.write((const char*)&hdr, sizeof hdr);
    };
    auto version = static_cast<int64_t>(m_version);
    auto& rt_info = model->get_rt_info();
    if (rt_info.count("version")) {
        version = rt_info.at("version").as<int64_t>();
    }

    if (version != static_cast<int64_t>(m_version) && m_version != Serialize::Version::UNSPECIFIED)
        OPENVINO_THROW("Cannot serialize model to incompatible IR version");

    if (version == static_cast<int64_t>(Serialize::Version::UNSPECIFIED)) {
        version = static_cast<int64_t>(Serialize::Version::IR_V11);
    }

    // Header
    const size_t header_offset = m_stream.tellp();
    writeHeader(hdr);

    // Custom data
    hdr.custom_data_offset = m_stream.tellp();
    if (m_custom_data_serializer) {
        m_custom_data_serializer(m_stream);
    }

    // Blobs
    hdr.consts_offset = m_stream.tellp();
    std::string name = "net";
    pugi::xml_document xml_doc;
    pugi::xml_node net_node = xml_doc.append_child(name.c_str());
    ConstantWriter constant_write_handler(m_stream);
    XmlSerializer visitor(net_node, name, constant_write_handler, version);
    std::shared_ptr<ov::Model> fun = model;
    visitor.on_attribute(name, fun);

    // IR
    hdr.model_offset = m_stream.tellp();
    if (m_cache_encrypt) {
        std::stringstream ss;
        xml_doc.save(ss);
        auto str_encode = m_cache_encrypt(ss.str());
        m_stream.write((char*)str_encode.c_str(), str_encode.length());
    } else {
        xml_doc.save(m_stream);
    }
    m_stream.flush();

    const size_t file_size = m_stream.tellp();

    hdr.custom_data_size = hdr.consts_offset - hdr.custom_data_offset;
    hdr.consts_size = hdr.model_offset - hdr.consts_offset;
    hdr.model_size = file_size - hdr.model_offset;

    m_stream.seekp(header_offset);
    writeHeader(hdr);

    m_stream.seekp(file_size);

    // Return false because we didn't change ov Model
    return false;
}

/// -------- Hash calculation pass -------------

namespace {
// Hash combine formula from boost for uint64_t.
inline uint64_t hash_combine(uint64_t h, uint64_t k) {
    constexpr uint64_t m = 0xc6a4a7935bd1e995;
    constexpr int r = 47;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;

    return h + 0xe6546b64;
}

class OstreamHashWrapper final : public std::streambuf {
    uint64_t m_res = 0lu;

public:
    uint64_t getResult() const {
        return m_res;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        uint64_t h = ov::runtime::compute_hash(s, n);
        m_res = hash_combine(m_res, h);

        return n;
    }
};
}  // namespace

std::streamsize OstreamHashWrapperBin::xsputn(const char* s, std::streamsize n) {
    m_res = hash_combine(m_res, *reinterpret_cast<const uint64_t*>(s));
    return n;
}

bool pass::Hash::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(Hash);
    OstreamHashWrapper xmlHash;
    OstreamHashWrapperBin binHash;
    std::ostream xml(&xmlHash);
    std::ostream bin(&binHash);

    // Determinism is important for hash calculation
    serializeFunc(xml, bin, model, Serialize::Version::UNSPECIFIED, true);

    uint64_t seed = 0;
    seed = hash_combine(seed, xmlHash.getResult());
    seed = hash_combine(seed, binHash.getResult());

    m_hash = seed;
    // Return false because we didn't change OpenVINO Model
    return false;
}

pass::Hash::Hash(uint64_t& output_hash_value) : m_hash(output_hash_value) {}

}  // namespace ov
