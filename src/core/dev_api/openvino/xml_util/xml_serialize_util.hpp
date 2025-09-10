// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <pugixml.hpp>
#include <string>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/xml_util/constant_writer.hpp"

namespace ov::util {

static inline constexpr std::string_view rt_map_user_data_prefix{"[UserData]"};
static inline constexpr std::string_view rt_info_user_data_xml_tag{"user_data"};

OPENVINO_API std::string get_ir_precision_name(const element::Type& precision);

class OPENVINO_API XmlSerializer : public ov::AttributeVisitor {
    pugi::xml_node& m_xml_node;
    const std::string& m_node_type_name;
    std::reference_wrapper<util::ConstantWriter> m_constant_node_write_handler;
    int64_t m_version;
    bool m_deterministic;
    bool m_compress_to_fp16;
    ov::element::Type m_output_element_type;
    bool m_data_is_temporary;
    std::function<bool(pugi::xml_node& node, const ov::RuntimeAttribute& attribute)> m_custom_rt_info_append;

    template <typename T>
    std::string create_attribute_list(ov::ValueAccessor<std::vector<T>>& adapter) {
        return util::join(adapter.get());
    }

    std::vector<std::string> map_type_from_body(const pugi::xml_node& xml_node,
                                                const std::string& map_type,
                                                int64_t ir_version,
                                                const std::string& body_name = "body");

    void input_descriptions_on_adapter(
        const std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>>& input_descriptions,
        const std::vector<std::string>& parameter_mapping,
        const std::vector<std::string>& result_mapping,
        pugi::xml_node& port_map,
        const std::string& portmap_name);

    void output_descriptions_on_adapter(
        const std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>>& output_descriptions,
        const uint32_t& input_count,
        const std::vector<std::string>& result_mapping,
        pugi::xml_node& port_map,
        const std::string& portmap_name);

    void special_body_ports_on_adapter(const ov::op::v5::Loop::SpecialBodyPorts& special_body_ports,
                                       const std::vector<std::string>& parameter_mapping,
                                       const std::vector<std::string>& result_mapping,
                                       pugi::xml_node& port_map);

    virtual std::unique_ptr<XmlSerializer> make_visitor(pugi::xml_node& data,
                                                        const std::string& node_type_name,
                                                        ov::util::ConstantWriter& constant_write_handler,
                                                        int64_t version,
                                                        bool deterministic = false,
                                                        bool compress_to_fp16 = false,
                                                        ov::element::Type output_element_type = ov::element::dynamic,
                                                        bool data_is_temporary = false) const;

    void serialize(pugi::xml_node& net_xml, const ov::Model& model);

protected:
    virtual void append_rt_info(pugi::xml_node& node, ov::RTMap& attributes);
    virtual bool append_rt_attribute(pugi::xml_node& node, const ov::RuntimeAttribute& attribute);
    virtual bool append_node_attributes(ov::Node& node);
    virtual util::ConstantWriter& get_constant_write_handler() const {
        return m_constant_node_write_handler;
    }

public:
    XmlSerializer(pugi::xml_node& data,
                  const std::string& node_type_name,
                  ov::util::ConstantWriter& constant_write_handler,
                  int64_t version,
                  bool deterministic = false,
                  bool compress_to_fp16 = false,
                  ov::element::Type output_element_type = ov::element::dynamic,
                  bool data_is_temporary = false);

    void on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<bool>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<std::string>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<int64_t>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<double>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int>>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<int64_t>>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<uint64_t>>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<float>>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<std::vector<std::string>>& adapter) override;
    void on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) override;
};  // class XmlSerializer
}  // namespace ov::util
