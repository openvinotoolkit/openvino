// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/serialize.hpp"
#include "openvino/xml_util/xml_serialize_util.hpp"

namespace intel_npu {

/**
 * @brief Nothing is stored. The weights are expected to be reconstruscted in some other way.
 */
class WeightlessWriter : public ov::util::ConstantWriter {
public:
    explicit WeightlessWriter(ov::util::ConstantWriter& other) : ov::util::ConstantWriter(other) {}

    FilePosition write(const char*, size_t, size_t&, bool, ov::element::Type, bool) override {
        return 0;
    }
};

/**
 * @brief Overriden in order to allow serializing models without copying weights.
 * @details Weights can be stored either as values (buffer copies, just like the parent algorithm), or as metadata
 * (memory location + buffer size in bytes). The amount of weights that are copied as values can be controlled by
 * configuring the "intel_npu::serialization_weights_size_threshold" option.
 */
class XmlSerializer : public ov::util::XmlSerializer {
public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  ov::util::ConstantWriter& constant_write_handler,
                  int64_t version)
        : ov::util::XmlSerializer(data,
                                  node_type_name,
                                  constant_write_handler,
                                  version,
                                  false,
                                  false,
                                  ov::element::dynamic,
                                  false),
          m_weightless_constant_writer(constant_write_handler),
          m_base_constant_writer(std::ref(constant_write_handler)) {}

private:
    /**
     * @brief Toggles between the two writers.
     */
    ov::util::ConstantWriter& get_constant_write_handler() override;

    /**
     * @brief Overriden in order to choose which weights writer will be used based on the occurrence of the
     * "WeightsPointerAttribute".
     */
    bool append_node_attributes(ov::Node& node) override;

    std::unique_ptr<ov::util::XmlSerializer> make_visitor(pugi::xml_node& data,
                                                          const std::string& node_type_name,
                                                          ov::util::ConstantWriter& constant_write_handler,
                                                          int64_t version,
                                                          bool,
                                                          bool,
                                                          ov::element::Type,
                                                          bool) const override;

    /**
     * @brief Writes nothing. The visitor pattern will be used in order to store weights metadata instead.
     */
    WeightlessWriter m_weightless_constant_writer;
    /**
     * @brief The base OV writer, copies the weights in a dedicated buffer.
     */
    std::reference_wrapper<ov::util::ConstantWriter> m_base_constant_writer;
    bool m_use_weightless_writer = false;
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
