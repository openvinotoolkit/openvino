// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/serialize.hpp"
#include "openvino/xml_util/xml_serialize_util.hpp"

namespace intel_npu {

/**
 * @brief Writes nothing. The visitor pattern will be used to store weights metadata instead.
 * @see WeightsPointerAttribute::visit_attributes()
 */
class WeightlessWriter : public ov::util::ConstantWriter {
public:
    explicit WeightlessWriter(ov::util::ConstantWriter& other) : ov::util::ConstantWriter(other) {}

    FilePosition write(const char*, size_t, size_t&, bool, ov::element::Type, bool) override {
        return 0;
    }
};

/**
 * @brief Overriden in order to allow marshalling models without copying weights.
 */
class XmlSerializer : public ov::util::XmlSerializer {
public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  ov::util::ConstantWriter& constant_write_handler,
                  int64_t version,
                  std::shared_ptr<WeightlessWriter> weightless_constant_writer = nullptr)
        : ov::util::XmlSerializer(data,
                                  node_type_name,
                                  constant_write_handler,
                                  version,
                                  false,
                                  false,
                                  ov::element::dynamic,
                                  false),
          m_weightless_constant_writer(weightless_constant_writer
                                           ? weightless_constant_writer
                                           : std::make_shared<WeightlessWriter>(constant_write_handler)) {}

private:
    ov::util::ConstantWriter& get_constant_write_handler() override;

    std::unique_ptr<ov::util::XmlSerializer> make_visitor(pugi::xml_node& data,
                                                          const std::string& node_type_name,
                                                          ov::util::ConstantWriter& constant_write_handler,
                                                          int64_t version,
                                                          bool,
                                                          bool,
                                                          ov::element::Type,
                                                          bool) const override;

    std::shared_ptr<WeightlessWriter> m_weightless_constant_writer = nullptr;
};

/**
 * @brief Leverages the "intel_npu::XmlSerializer" in order to allow serializing models without copying weights.
 */
class StreamSerialize : public ov::pass::StreamSerialize {
public:
    StreamSerialize(std::ostream& stream,
                    const std::function<void(std::ostream&)>& custom_data_serializer,
                    ov::pass::Serialize::Version version = ov::pass::Serialize::Version::UNSPECIFIED)
        : ov::pass::StreamSerialize(stream, custom_data_serializer, {}, version) {}

private:
    std::unique_ptr<ov::util::XmlSerializer> make_serializer(pugi::xml_node& data,
                                                             const std::string& node_type_name,
                                                             ov::util::ConstantWriter& constant_write_handler,
                                                             int64_t version,
                                                             bool,
                                                             bool,
                                                             ov::element::Type,
                                                             bool) const override {
        return std::make_unique<XmlSerializer>(data, node_type_name, constant_write_handler, version);
    }
};

}  // namespace intel_npu
