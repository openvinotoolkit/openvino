// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deserializer.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "openvino/xml_util/xml_deserialize_util.hpp"
#include "utils/codec_xor.hpp"

namespace ov::intel_cpu {

ModelDeserializer::ModelDeserializer(std::shared_ptr<ov::AlignedBuffer>& model_buffer,
                                     const std::shared_ptr<ov::ICore>& core,
                                     const CacheDecrypt& decrypt_fn,
                                     bool decript_from_string,
                                     const std::string& origin_weights_path)
    : m_model(model_buffer),
      m_core(core),
      m_decript_from_string(decript_from_string) {
    if (!origin_weights_path.empty() && std::filesystem::exists(origin_weights_path)) {
        auto mmap = ov::load_mmap_object(origin_weights_path);
        m_origin_weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mmap->data(), mmap->size(), mmap);
    }

    if (m_decript_from_string) {
        m_cache_decrypt.m_decrypt_str = decrypt_fn.m_decrypt_str;
    } else {
        m_cache_decrypt.m_decrypt_char = decrypt_fn.m_decrypt_char;
    }
}

ModelDeserializer::ModelDeserializer(std::istream& model_stream,
                                     const std::shared_ptr<ov::ICore>& core,
                                     const CacheDecrypt& decrypt_fn,
                                     bool decript_from_string,
                                     const std::string& origin_weights_path)
    : m_model(model_stream),
      m_core(core),
      m_decript_from_string(decript_from_string) {
    if (!origin_weights_path.empty() && std::filesystem::exists(origin_weights_path)) {
        auto mmap = ov::load_mmap_object(origin_weights_path);
        m_origin_weights_buf =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<MappedMemory>>>(mmap->data(), mmap->size(), mmap);
    }

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

std::shared_ptr<ov::Model> ModelDeserializer::create_ov_model(
    const std::shared_ptr<ov::AlignedBuffer>& model_buf,
    const std::shared_ptr<ov::AlignedBuffer>& weights,
    const std::shared_ptr<ov::AlignedBuffer>& origin_weights) {
    if (origin_weights == nullptr) {
        return m_core->read_model(model_buf, weights);
    }

    // Custom deserialization for weightless mode

    pugi::xml_document xml_doc;
    const auto root = [&] {
        auto res =
            xml_doc.load_buffer(model_buf->get_ptr(), model_buf->size(), pugi::parse_default, pugi::encoding_utf8);
        OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
        return xml_doc.document_element();
    }();
    const auto opsets = [] {
        std::unordered_map<std::string, ov::OpSet> opsets;
        for (const auto& [name, mk_opset] : ov::get_available_opsets()) {
            opsets[name] = mk_opset();
        }
        return opsets;
    }();
    const auto version = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(root, "version", 0));

    auto create_extensions_map = [&]() -> std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> {
        std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr> exts;
        std::vector<ov::Extension::Ptr> m_extensions;
        OV_CREATE_EXTENSION(m_extensions);
        for (const auto& ext : m_extensions) {
            if (auto base_ext = std::dynamic_pointer_cast<ov::BaseOpExtension>(ext)) {
                exts.insert({base_ext->get_type_info(), base_ext});
            }
        }
        return exts;
    }();

    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
    const auto& w = (weights != nullptr && weights->size() != 0) ? weights : origin_weights;
    XmlDeserializer visitor(root, w, origin_weights, opsets, create_extensions_map, variables, version);
    std::shared_ptr<ov::Model> model;
    visitor.on_attribute("net", model);
    model->get_rt_info()["version"] = static_cast<int64_t>(version);
    return model;
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

    model = create_ov_model(model_buf, weights_buf, m_origin_weights_buf);

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
    model_stream.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));

    // Check if model header contains valid data.
    bool is_valid_model = (hdr.custom_data_offset == sizeof(hdr)) &&
                          (hdr.custom_data_size == hdr.consts_offset - hdr.custom_data_offset) &&
                          (hdr.consts_size == hdr.model_offset - hdr.consts_offset) &&
                          ((hdr.model_size = file_size - hdr.model_offset) != 0U);
    OPENVINO_ASSERT(is_valid_model, "[CPU] Could not deserialize by device xml header.");

    // read model input/output precisions
    model_stream.seekg(hdr.custom_data_offset + hdr_pos);

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
    model_stream.seekg(hdr.consts_offset + hdr_pos);
    if (hdr.consts_size) {
        model_stream.read(static_cast<char*>(data_blob->data(ov::element::u8)), hdr.consts_size);
    }

    // read XML content
    auto xml_string = std::make_shared<std::string>();
    model_stream.seekg(hdr.model_offset + hdr_pos);
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

    model = create_ov_model(model_buf, weights_buf, m_origin_weights_buf);

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
                    const ov::element::Type original_dt(child.attribute("type").value());  // "element_type"?
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
    const auto& node = get_node();
    const auto dn = node.child("data");
    const element::Type target_dtype{ov::util::pugixml::get_str_attr(dn, "element_type")};

    // wlc -> weightless cache
    bool is_wlc_way = target_dtype != element::string && m_origin_weights != nullptr;
    ov::Any wlc;
    if (is_wlc_way) {
        wlc = parse_weightless_cache_attribute(node);
        is_wlc_way &= !wlc.empty() && wlc.is<ov::WeightlessCacheAttribute>();
    }

    if (!is_wlc_way) {
        ov::util::XmlDeserializer::set_constant_num_buffer(adapter);
        return;
    }

    const auto& wlc_attribute = wlc.as<ov::WeightlessCacheAttribute>();

    auto actual_size = wlc_attribute.original_size;
    auto offset = wlc_attribute.bin_offset;
    auto w_size = m_origin_weights->size();
    OPENVINO_ASSERT(w_size >= offset + actual_size, "Incorrect weights in bin file!");

    auto original_dtype = wlc_attribute.original_dtype;
    char* data = m_origin_weights->get_ptr<char>() + offset;

    ov::Shape shape;
    OPENVINO_ASSERT(getParameters<size_t>(dn, "shape", shape),
                    "[ CPU ] Could not get attribute 'shape' during weights deserialization.");

    if (original_dtype != target_dtype) {
        const auto org_tensor = ov::Tensor(original_dtype, shape, data);
        auto converted_weights =
            std::make_shared<ov::AlignedBuffer>(ov::util::get_memory_size(target_dtype, ov::shape_size(shape)));
        auto converted_output = ov::TensorVector{{target_dtype, shape, converted_weights->get_ptr()}};
        auto convert = op::v0::Convert();
        OPENVINO_ASSERT(convert.evaluate(converted_output, {org_tensor}), "Conversion not supported");
        adapter.set(converted_weights);
    } else {
        if (actual_size < ((ov::shape_size(shape) * target_dtype.bitwidth() + 7) >> 3)) {
            const auto type = ov::util::pugixml::get_str_attr(get_node(), "type");
            OPENVINO_THROW("Attribute and shape size are inconsistent for ",
                           type,
                           " op!",
                           actual_size,
                           ", ",
                           ((ov::shape_size(shape) * target_dtype.bitwidth() + 7) >> 3),
                           ", ",
                           ov::util::get_memory_size(target_dtype, ov::shape_size(shape)));
        }

        auto buffer =
            std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(data, actual_size, m_origin_weights);
        adapter.set(buffer);
    }
}

}  // namespace ov::intel_cpu
