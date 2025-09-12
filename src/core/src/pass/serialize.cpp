// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/serialize.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/compute_hash.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/xml_util/constant_writer.hpp"
#include "openvino/xml_util/xml_serialize_util.hpp"
#include "pugixml.hpp"
#include "transformations/hash.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"

namespace {
const std::filesystem::path valid_xml_path(const std::filesystem::path& path) {
    OPENVINO_ASSERT(path.extension() == ".xml",
                    "Path for xml file doesn't contains file name with 'xml' extension: \"",
                    path,
                    "\"");
    return path;
}

std::filesystem::path provide_bin_path(const std::filesystem::path& xml_path, const std::filesystem::path& bin_path) {
    if (bin_path.empty()) {
        auto path = xml_path;
        path.replace_extension(".bin");
        return path;
    } else {
        return bin_path;
    }
}

void serialize_func(std::ostream& xml_file,
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
    ov::util::ConstantWriter constant_write_handler(bin_file);
    ov::util::XmlSerializer
        visitor(net_node, name, constant_write_handler, version, deterministic, false, ov::element::dynamic, false);
    visitor.on_attribute(name, model);

    xml_doc.save(xml_file);
    xml_file.flush();
    bin_file.flush();
}

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
        serialize_func(*m_xmlFile, *m_binFile, model, m_version);
    } else {
        ov::util::create_directory_recursive(m_xmlPath);

        std::ofstream bin_file(m_binPath, std::ios::binary);
        OPENVINO_ASSERT(bin_file, "Can't open bin file: \"", m_binPath, "\"");

        // create xml file
        std::ofstream xml_file(m_xmlPath);
        OPENVINO_ASSERT(xml_file, "Can't open xml file: \"", m_xmlPath, "\"");

        try {
            serialize_func(xml_file, bin_file, model, m_version);
        } catch (const ov::AssertFailure&) {
            // optimization decision was made to create .bin file upfront and
            // write to it directly instead of buffering its content in memory,
            // hence we need to delete it here in case of failure
            xml_file.close();
            bin_file.close();
            std::ignore = std::filesystem::remove(m_xmlPath);
            std::ignore = std::filesystem::remove(m_binPath);
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

pass::Serialize::Serialize(const std::filesystem::path& xmlPath, const std::filesystem::path& binPath, Version version)
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

bool pass::StreamSerialize::use_absolute_offset() {
    return true;
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
    const auto absolute_header_offset = static_cast<size_t>(m_stream.tellp());
    const auto header_offset = use_absolute_offset() ? size_t{0} : absolute_header_offset;
    writeHeader(hdr);

    // Custom data
    hdr.custom_data_offset = static_cast<size_t>(m_stream.tellp()) - header_offset;
    if (m_custom_data_serializer) {
        m_custom_data_serializer(m_stream);
    }

    // Blobs
    hdr.consts_offset = static_cast<size_t>(m_stream.tellp()) - header_offset;
    const std::string name = "net";
    pugi::xml_document xml_doc;
    pugi::xml_node net_node = xml_doc.append_child(name.c_str());
    auto constant_write_handler = util::ConstantWriter(m_stream);
    const auto visitor = make_serializer(net_node, name, constant_write_handler, version);
    std::shared_ptr<ov::Model> fun = model;
    visitor->on_attribute(name, fun);

    // IR
    hdr.model_offset = static_cast<size_t>(m_stream.tellp()) - header_offset;
    if (m_cache_encrypt) {
        std::stringstream ss;
        xml_doc.save(ss);
        auto str_encode = m_cache_encrypt(ss.str());
        m_stream.write(str_encode.c_str(), str_encode.length());
    } else {
        xml_doc.save(m_stream);
    }
    m_stream.flush();

    const auto file_size = static_cast<size_t>(m_stream.tellp()) - header_offset;

    hdr.custom_data_size = hdr.consts_offset - hdr.custom_data_offset;
    hdr.consts_size = hdr.model_offset - hdr.consts_offset;
    hdr.model_size = file_size - hdr.model_offset;

    m_stream.seekp(absolute_header_offset);
    writeHeader(hdr);

    m_stream.seekp(file_size);

    // Return false because we didn't change ov Model
    return false;
}

std::unique_ptr<util::XmlSerializer> pass::StreamSerialize::make_serializer(
    pugi::xml_node& data,
    const std::string& node_type_name,
    util::ConstantWriter& constant_write_handler,
    int64_t version,
    bool deterministic,
    bool compress_to_fp16,
    ov::element::Type output_element_type,
    bool data_is_temporary) const {
    return std::make_unique<util::XmlSerializer>(data,
                                                 node_type_name,
                                                 constant_write_handler,
                                                 version,
                                                 deterministic,
                                                 compress_to_fp16,
                                                 output_element_type,
                                                 data_is_temporary);
}

/// -------- Hash calculation pass -------------

namespace {

class OstreamHashWrapper final : public std::streambuf {
    uint64_t m_res = 0lu;

public:
    uint64_t getResult() const {
        return m_res;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        uint64_t h = ov::runtime::compute_hash(s, n);
        m_res = util::u64_hash_combine(m_res, h);

        return n;
    }
};
}  // namespace

bool pass::Hash::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(Hash);
    OstreamHashWrapper xmlHash;
    util::OstreamHashWrapperBin binHash;
    std::ostream xml(&xmlHash);
    std::ostream bin(&binHash);

    // Determinism is important for hash calculation
    serialize_func(xml, bin, model, Serialize::Version::UNSPECIFIED, true);

    uint64_t seed = 0;
    seed = util::u64_hash_combine(seed, xmlHash.getResult());
    seed = util::u64_hash_combine(seed, binHash.get_result());

    m_hash = seed;
    // Return false because we didn't change OpenVINO Model
    return false;
}

pass::Hash::Hash(uint64_t& output_hash_value) : m_hash(output_hash_value) {}

}  // namespace ov
