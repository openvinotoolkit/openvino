// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "serializer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <ostream>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/xml_util/constant_writer.hpp"
#include "openvino/xml_util/xml_serialize_util.hpp"

namespace ov::intel_cpu {

class WeightlessWriter : public util::ConstantWriter {
public:
    explicit WeightlessWriter(util::ConstantWriter& other) : util::ConstantWriter(other), m_offset{} {}

    WeightlessWriter(std::ostream& bin_file) : util::ConstantWriter(bin_file), m_offset{} {}

    WeightlessWriter::FilePosition write([[maybe_unused]] const char* ptr,
                                         size_t size,
                                         size_t& new_size,
                                         [[maybe_unused]] bool compress_to_fp16,
                                         [[maybe_unused]] ov::element::Type src_type,
                                         [[maybe_unused]] bool ptr_is_temporary) override {
        WeightlessWriter::FilePosition offset = 0L;

        if (m_skip_weights) {
            new_size = 0LU;
            offset = m_offset;
            m_offset += size;
        } else {
            offset = util::ConstantWriter::write(ptr, size, new_size, compress_to_fp16, src_type, ptr_is_temporary);
        }

        return offset;
    }

    void skip_weights(bool skip_weights) {
        m_skip_weights = skip_weights;
    }

private:
    WeightlessWriter::FilePosition m_offset;
    bool m_skip_weights = false;
};

class XmlSerializer : public util::XmlSerializer {
public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  util::ConstantWriter& constant_write_handler,
                  int64_t version,
                  bool deterministic = false,
                  bool compress_to_fp16 = false,
                  ov::element::Type output_element_type = ov::element::dynamic,
                  bool data_is_temporary = false,
                  bool wl_mode = false)
        : util::XmlSerializer(data,
                              node_type_name,
                              constant_write_handler,
                              version,
                              deterministic,
                              compress_to_fp16,
                              output_element_type,
                              data_is_temporary),
          m_weightless_const_writer(constant_write_handler),
          m_weightless_mode(wl_mode) {}

private:
    bool append_rt_attribute(pugi::xml_node& node, const ov::RuntimeAttribute& attribute) override {
        bool result = false;
        if (const auto* wl_attr = ov::as_type<const ov::WeightlessCacheAttribute>(&attribute)) {
            m_weightless_const_writer.skip_weights(true);

            const auto& type_info = attribute.get_type_info();
            node.append_attribute("name").set_value(type_info.name);
            node.append_attribute("version").set_value(type_info.get_version().data());
            node.append_attribute("type").set_value(util::get_ir_precision_name(wl_attr->original_dtype).data());
            node.append_attribute("offset").set_value(wl_attr->bin_offset);
            node.append_attribute("size").set_value(wl_attr->original_size);

            result = true;
        } else {
            result = util::XmlSerializer::append_rt_attribute(node, attribute);
        }

        return result;
    }

    bool append_node_attributes(ov::Node& node) override {
        m_weightless_const_writer.skip_weights(
            m_weightless_mode && node.get_rt_info().count(ov::WeightlessCacheAttribute::get_type_info_static()) != 0);

        auto result = util::XmlSerializer::append_node_attributes(node);

        return result;
    }

    ov::util::ConstantWriter& get_constant_write_handler() override {
        return m_weightless_const_writer;
    }

    std::unique_ptr<util::XmlSerializer> make_visitor(pugi::xml_node& data,
                                                      const std::string& node_type_name,
                                                      util::ConstantWriter& constant_write_handler,
                                                      int64_t version,
                                                      bool deterministic,
                                                      bool compress_to_fp16,
                                                      ov::element::Type output_element_type,
                                                      bool data_is_temporary) const override {
        return std::make_unique<XmlSerializer>(data,
                                               node_type_name,
                                               constant_write_handler,
                                               version,
                                               deterministic,
                                               compress_to_fp16,
                                               output_element_type,
                                               data_is_temporary,
                                               m_weightless_mode);
    }

    WeightlessWriter m_weightless_const_writer;
    bool m_weightless_mode = false;
};

////////// ModelSerializer //////////

ModelSerializer::ModelSerializer(std::ostream& ostream, const CacheEncrypt& encrypt_fn, bool weightless_mode)
    : ov::pass::StreamSerialize(
          ostream,
          [](std::ostream& stream) {
              pugi::xml_document xml_doc;
              pugi::xml_node root = xml_doc.append_child("cnndata");
              root.append_child("outputs");
              xml_doc.save(stream);
          },
          encrypt_fn),
      m_weightless_mode(weightless_mode) {};

void ModelSerializer::operator<<(const std::shared_ptr<ov::Model>& model) {
    run_on_model(std::const_pointer_cast<ov::Model>(model->clone()));
}

bool ModelSerializer::use_absolute_offset() {
    return false;
}

std::unique_ptr<util::XmlSerializer> ModelSerializer::make_serializer(pugi::xml_node& data,
                                                                      const std::string& node_type_name,
                                                                      util::ConstantWriter& constant_write_handler,
                                                                      int64_t version,
                                                                      bool deterministic,
                                                                      bool compress_to_fp16,
                                                                      ov::element::Type output_element_type,
                                                                      bool data_is_temporary) const {
    return std::make_unique<XmlSerializer>(data,
                                           node_type_name,
                                           constant_write_handler,
                                           version,
                                           deterministic,
                                           compress_to_fp16,
                                           output_element_type,
                                           data_is_temporary,
                                           m_weightless_mode);
}

}  // namespace ov::intel_cpu
