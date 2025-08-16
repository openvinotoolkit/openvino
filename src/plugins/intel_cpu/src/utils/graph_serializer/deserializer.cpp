// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deserializer.hpp"

#include <cstddef>
#include <cstring>
#include <functional>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <variant>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "utils/codec_xor.hpp"

namespace ov::intel_cpu {
 
ModelDeserializer::ModelDeserializer(std::shared_ptr<ov::AlignedBuffer>& model_buffer,
                                     ModelBuilder fn,
                                     const CacheDecrypt& decrypt_fn,
                                     bool decript_from_string,
                                     std::string origin_weights_path)
    : m_model(model_buffer),
      m_model_builder(std::move(fn)),
      m_decript_from_string(decript_from_string),
      m_origin_weights_path(std::move(origin_weights_path)) {
    if (m_decript_from_string) {
        m_cache_decrypt.m_decrypt_str = decrypt_fn.m_decrypt_str;
    } else {
        m_cache_decrypt.m_decrypt_char = decrypt_fn.m_decrypt_char;
    }
}

ModelDeserializer::ModelDeserializer(std::istream& model_stream,
                                     ModelBuilder fn,
                                     const CacheDecrypt& decrypt_fn,
                                     bool decript_from_string,
                                     std::string origin_weights_path)
    : m_model(model_stream),
      m_model_builder(std::move(fn)),
      m_decript_from_string(decript_from_string),
      m_origin_weights_path(std::move(origin_weights_path)) {
    if (m_decript_from_string) {
        m_cache_decrypt.m_decrypt_str = decrypt_fn.m_decrypt_str;
    } else {
        m_cache_decrypt.m_decrypt_char = decrypt_fn.m_decrypt_char;
    }
}

void ModelDeserializer::set_info(pugi::xml_node& root, std::shared_ptr<ov::Model>& model) {}

void ModelDeserializer::operator>>(std::shared_ptr<ov::Model>& model) {
    std::visit(
        [&](auto&& arg) {
            process_model(model, std::forward<decltype(arg)>(arg));
        },
        m_model);
}

void ModelDeserializer::process_model(std::shared_ptr<ov::Model>& model,
                                      const std::shared_ptr<ov::AlignedBuffer>& model_buffer) {
    // Note: Don't use seekg with mmaped stream. This may affect the performance of some models.
    // Get file size before seek content.
    // Blob from cache may have other header, so need to skip this.
    auto* buffer_base = reinterpret_cast<char*>(model_buffer->get_ptr());

    const auto file_size = model_buffer->size();
    pass::StreamSerialize::DataHeader hdr = {};
    std::memcpy(reinterpret_cast<char*>(&hdr), buffer_base, sizeof hdr);

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr)) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          ((hdr.model_size = file_size - hdr.model_offset) != 0U);
    OPENVINO_ASSERT(is_valid_model, "[CPU] Could not deserialize by device xml header.");

    // Read model input/output precisions.
    pugi::xml_document xml_in_out_doc;
    if (hdr.custom_data_size > 0LU) {
        auto res = xml_in_out_doc.load_buffer(buffer_base + hdr.custom_data_offset,
                                              hdr.custom_data_size,
                                              pugi::parse_default,
                                              pugi::encoding_utf8);
        OPENVINO_ASSERT(res.status == pugi::status_ok, "[CPU] Could to deserialize custom data.");
    }

    // Map blob content
    std::shared_ptr<ov::AlignedBuffer> weights_buf;
    if (hdr.consts_size) {
        weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(buffer_base + hdr.consts_offset,
                                                                                   hdr.consts_size,
                                                                                   model_buffer);
    }

    std::shared_ptr<ov::AlignedBuffer> origin_weights_buf;
    if (!m_origin_weights_path.empty()) {
        auto mmap = ov::load_mmap_object(m_origin_weights_path);
        origin_weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mmap->data(), mmap->size(), mmap);
    }

    // XML content
    auto xml_buff = std::make_shared<std::string>();
    if (m_cache_decrypt) {
        if (m_decript_from_string) {
            xml_buff->assign(buffer_base + hdr.model_offset, hdr.model_size);
            *xml_buff = m_cache_decrypt.m_decrypt_str(*xml_buff);
        } else {
            xml_buff->reserve(hdr.model_size + 1);
            m_cache_decrypt.m_decrypt_char((*xml_buff).data(), buffer_base + hdr.model_offset, hdr.model_size);
        }
    } else {
        xml_buff->assign(buffer_base + hdr.model_offset, hdr.model_size);
    }
    std::shared_ptr<ov::AlignedBuffer> model_buf =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<std::string>>>((*xml_buff).data(), hdr.model_size, xml_buff);

    model = m_model_builder(model_buf, weights_buf, origin_weights_buf);

    // Set Info
    pugi::xml_node root = xml_in_out_doc.child("cnndata");
    set_info(root, model);
}

void ModelDeserializer::process_model(std::shared_ptr<ov::Model>& model,
                                      std::reference_wrapper<std::istream> model_stream_ref) {
    auto& model_stream = model_stream_ref.get();

    const size_t hdr_pos = model_stream.tellg();
    model_stream.seekg(0, std::istream::end);
    const size_t file_size = model_stream.tellg();
    model_stream.seekg(hdr_pos, std::istream::beg);

    pass::StreamSerialize::DataHeader hdr = {};
    model_stream.read(reinterpret_cast<char*>(&hdr), sizeof hdr);

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr) + hdr_pos) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          ((hdr.model_size = file_size - hdr.model_offset) != 0U);
    OPENVINO_ASSERT(is_valid_model, "[CPU] Could not deserialize by device xml header.");

    // read model input/output precisions
    model_stream.seekg(hdr.custom_data_offset);

    pugi::xml_document xmlInOutDoc;
    if (hdr.custom_data_size > 0) {
        std::string xmlInOutString;
        xmlInOutString.resize(hdr.custom_data_size);
        model_stream.read(const_cast<char*>(xmlInOutString.c_str()), hdr.custom_data_size);
        auto res = xmlInOutDoc.load_string(xmlInOutString.c_str());
        OPENVINO_ASSERT(res.status == pugi::status_ok,
                        "NetworkNotRead: The inputs and outputs information is invalid.");
    }

    // read blob content
    auto data_blob = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape({hdr.consts_size}));
    model_stream.seekg(hdr.consts_offset);
    if (hdr.consts_size) {
        model_stream.read(static_cast<char*>(data_blob->data(ov::element::u8)), hdr.consts_size);
    }

    std::shared_ptr<ov::AlignedBuffer> origin_weights_buf;
    if (!m_origin_weights_path.empty()) {
        auto mmap = ov::load_mmap_object(m_origin_weights_path);
        origin_weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mmap->data(), mmap->size(), mmap);
    }

    // read XML content
    auto xml_string = std::make_shared<std::string>();
    model_stream.seekg(hdr.model_offset);
    xml_string->resize(hdr.model_size);
    model_stream.read(const_cast<char*>(xml_string->data()), hdr.model_size);
    if (m_cache_decrypt) {
        if (m_decript_from_string) {
            *xml_string = m_cache_decrypt.m_decrypt_str(*xml_string);
        } else {
            m_cache_decrypt.m_decrypt_char(const_cast<char*>(xml_string->data()),
                                           xml_string->data(),
                                           xml_string->size());
        }
    }

    auto model_buf =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<std::string>>>(const_cast<char*>(xml_string->data()),
                                                                         xml_string->size(),
                                                                         xml_string);
    auto weights_buf = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::Tensor>>>(
        reinterpret_cast<char*>(data_blob->data(ov::element::u8)),
        hdr.consts_size,
        data_blob);

    model = m_model_builder(model_buf, weights_buf, origin_weights_buf);

    // Set Info
    pugi::xml_node root = xmlInOutDoc.child("cnndata");
    set_info(root, model);
};


ov::Any XmlDeserializer::parse_weightless_cache_attribute(const pugi::xml_node& node) const {
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

void XmlDeserializer::set_constant_num_buffer(ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>& adapter) {
    OPENVINO_ASSERT(get_weights() != nullptr || m_origin_weights != nullptr,
                    "Empty weights data in bin file or bin file cannot be found!");
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
        // Some test case Here is an issue as after call the code below is not executed and Const buffer is not set
        ov::Any wl_attr = parse_weightless_cache_attribute(node);
        size_t offset = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "offset"));
        auto actual_size = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "size"));
        auto original_dtype = el_type;
        if (wl_attr.is<ov::WeightlessCacheAttribute>()) {
            char* data = get_weights()->get_ptr<char>() + offset;
            auto w_size = get_weights()->size();
            auto w_so = get_weights();
            if (wl_attr.is<ov::WeightlessCacheAttribute>()) {
                const auto& wl = wl_attr.as<ov::WeightlessCacheAttribute>();
                actual_size = wl.original_size;
                offset = wl.bin_offset;
                original_dtype = wl.original_dtype;
                data = m_origin_weights->get_ptr<char>() + offset;
                w_size = m_origin_weights->size();
                w_so = m_origin_weights;
            }
            OPENVINO_ASSERT(w_size >= offset + actual_size, "Incorrect weights in bin file!");
            if (original_dtype != el_type) {
                const auto org_tensor = ov::Tensor(original_dtype, shape, data);
                auto converted_weights = std::make_shared<ov::AlignedBuffer>(
                    ov::element::get_memory_size(el_type, ov::shape_size(shape)));
                {
                    auto converted_output = ov::TensorVector{{el_type, shape, converted_weights->get_ptr()}};
                    auto convert = ov::op::v0::Convert();
                    OPENVINO_ASSERT(convert.evaluate(converted_output, {org_tensor}), "Conversion not supported");
                }
                adapter.set(converted_weights);
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
                                    ov::element::get_memory_size(el_type, ov::shape_size(shape)));
                }

                auto buffer =
                    std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(data, actual_size, w_so);
                adapter.set(buffer);
            }
        } else {
            ov::util::XmlDeserializer::set_constant_num_buffer(adapter);
        }
    }
}

}  // namespace ov::intel_cpu
