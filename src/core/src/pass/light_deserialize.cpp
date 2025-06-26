// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/light_deserialize.hpp"

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
#include "openvino/op/binary_convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/util/avg_pool_base.hpp"
#include "openvino/op/util/deformable_convolution_base.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/max_pool_base.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/compute_hash.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "pugixml.hpp"
#include "transformations/hash.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/primitives_priority_attribute.hpp"
namespace ov {
namespace pass {
bool LightDeserialize::run_on_model(std::shared_ptr<ov::Model>& model) {
    pugi::xml_document xml_doc;
    pugi::xml_parse_result result = xml_doc.load(*m_xmlFile);
    if (!result) {
        OPENVINO_THROW("Failed to parse XML: ", result.description());
    }
    pugi::xml_node net_node = xml_doc.child("net");
    if (!net_node) {
        OPENVINO_THROW("No <net> node found in XML");
    }
    model = deserializeFunc(net_node, m_offsetConstMap);
    return true;
}

std::shared_ptr<ov::Model> LightDeserialize::deserializeFunc(
    const pugi::xml_node& net_node,
    std::map<int64_t, std::reference_wrapper<ov::ValueAccessor<void>>>& offsetConstMap) {
    std::unordered_map<int, std::shared_ptr<ov::Node>> id_node_map;
    std::vector<std::shared_ptr<ov::Node>> parameters;
    std::vector<std::shared_ptr<ov::Node>> results;
    std::vector<std::shared_ptr<ov::Node>> sinks;
    std::vector<std::shared_ptr<ov::Node>> all_nodes;
    std::unordered_map<int, size_t> layer_output_ports;

    pugi::xml_node layers_node = net_node.child("layers");
    for (pugi::xml_node layer = layers_node.child("layer"); layer; layer = layer.next_sibling("layer")) {
        int id = layer.attribute("id").as_int();
        std::string type = layer.attribute("type").as_string();
        std::string name = layer.attribute("name").as_string();

        size_t output_ports = 0;
        if (auto output = layer.child("output")) {
            for (auto port = output.child("port"); port; port = port.next_sibling("port")) {
                output_ports++;
            }
        }
        layer_output_ports[id] = output_ports;

        std::map<std::string, std::string> data_attrs;
        if (auto data = layer.child("data")) {
            for (auto attr = data.attributes_begin(); attr != data.attributes_end(); ++attr) {
                data_attrs[attr->name()] = attr->value();
            }
        }

        std::shared_ptr<ov::Node> node;
        if (type == "Parameter") {
            auto output = layer.child("output").child("port");
            std::vector<ov::Dimension> shape;
            for (auto dim = output.child("dim"); dim; dim = dim.next_sibling("dim")) {
                shape.push_back(ov::Dimension(std::stoll(dim.child_value())));
            }
            std::string precision = output.attribute("precision").as_string();
            ov::element::Type elem_type = ov::element::Type(precision);
            node = std::make_shared<ov::op::v0::Parameter>(elem_type, ov::PartialShape(shape));
            node->set_friendly_name(name);
            parameters.push_back(node);
        } else if (type == "Const" || type == "Constant") {
            int64_t offset = -1;
            size_t size = 0;
            if (data_attrs.count("offset")) {
                offset = std::stoll(data_attrs["offset"]);
            }
            if (data_attrs.count("size")) {
                size = static_cast<size_t>(std::stoull(data_attrs["size"]));
            }
            std::string precision =
                data_attrs.count("element_type") ? data_attrs["element_type"] : data_attrs["precision"];
            ov::element::Type elem_type = ov::element::Type(precision);

            std::vector<ov::Dimension> shape;
            if (auto shape_attr = data_attrs.find("shape"); shape_attr != data_attrs.end()) {
                std::stringstream ss(shape_attr->second);
                std::string dim;
                while (std::getline(ss, dim, ',')) {
                    shape.push_back(ov::Dimension(std::stoll(dim)));
                }
            } else if (auto output = layer.child("output").child("port")) {
                for (auto dim = output.child("dim"); dim; dim = dim.next_sibling("dim")) {
                    shape.push_back(ov::Dimension(std::stoll(dim.child_value())));
                }
            }

            std::shared_ptr<ov::op::v0::Constant> const_node;
            if (offsetConstMap.count(offset)) {
                auto& accessor = offsetConstMap[offset].get();
                auto* buf_adapter = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>>(&accessor);
                if (buf_adapter) {
                    auto buf = buf_adapter->get();
                    const_node = std::make_shared<ov::op::v0::Constant>(elem_type,
                                                                        ov::Shape(shape.begin(), shape.end()),
                                                                        buf->get_ptr(),
                                                                        buf->size());
                } else {
                    auto* str_adapter =
                        ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>>(&accessor);
                    if (str_adapter) {
                        auto buf = str_adapter->get();
                        const_node = std::make_shared<ov::op::v0::Constant>(elem_type,
                                                                            ov::Shape(shape.begin(), shape.end()),
                                                                            buf->get_ptr(),
                                                                            buf->size());
                    } else {
                        OPENVINO_THROW("Unsupported constant accessor type for offset ", offset);
                    }
                }
            } else {
                std::vector<uint8_t> dummy(elem_type.size() * ov::shape_size(ov::Shape(shape.begin(), shape.end())), 0);
                const_node = std::make_shared<ov::op::v0::Constant>(elem_type,
                                                                    ov::Shape(shape.begin(), shape.end()),
                                                                    dummy.data());
            }
            const_node->set_friendly_name(name);
            node = const_node;
        } else if (type == "Result") {
            node = std::make_shared<ov::op::v0::Result>(nullptr);
            node->set_friendly_name(name);
            results.push_back(node);
        } else {
            auto fw_node = std::make_shared<ov::op::util::FrameworkNode>(ov::OutputVector{}, output_ports);
            fw_node->set_friendly_name(name);
            node = fw_node;
        }
        id_node_map[id] = node;
        all_nodes.push_back(node);
    }

    pugi::xml_node edges_node = net_node.child("edges");
    for (pugi::xml_node edge = edges_node.child("edge"); edge; edge = edge.next_sibling("edge")) {
        int from_layer = edge.attribute("from-layer").as_int();
        int from_port = edge.attribute("from-port").as_int();
        int to_layer = edge.attribute("to-layer").as_int();
        int to_port = edge.attribute("to-port").as_int();

        auto from_node = id_node_map.at(from_layer);
        auto to_node = id_node_map.at(to_layer);

        if (auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(to_node)) {
            if (fw_node->inputs().size() <= static_cast<size_t>(to_port)) {
                fw_node->set_inputs(
                    ov::OutputVector(fw_node->inputs().size() + (to_port + 1 - fw_node->inputs().size())));
            }
            fw_node->input(to_port).replace_source_output(from_node->output(from_port));
        } else if (auto result_node = std::dynamic_pointer_cast<ov::op::v0::Result>(to_node)) {
            result_node->input(0).replace_source_output(from_node->output(from_port));
        } else if (auto conv_node = std::dynamic_pointer_cast<ov::op::v1::Convolution>(to_node)) {
            conv_node->input(to_port).replace_source_output(from_node->output(from_port));
        } else if (auto group_conv_node = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(to_node)) {
            group_conv_node->input(to_port).replace_source_output(from_node->output(from_port));
        } else if (auto bin_conv_node = std::dynamic_pointer_cast<ov::op::v1::BinaryConvolution>(to_node)) {
            bin_conv_node->input(to_port).replace_source_output(from_node->output(from_port));
        } else if (auto loop_node = std::dynamic_pointer_cast<ov::op::v5::Loop>(to_node)) {
            loop_node->input(to_port).replace_source_output(from_node->output(from_port));
        } else if (auto lstm_cell_node = std::dynamic_pointer_cast<ov::op::v0::LSTMCell>(to_node)) {
            lstm_cell_node->input(to_port).replace_source_output(from_node->output(from_port));
        } else if (auto param_node = std::dynamic_pointer_cast<ov::op::v0::Parameter>(to_node)) {
        } else if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(to_node)) {
        } else {
            if (to_node->inputs().size() > static_cast<size_t>(to_port)) {
                to_node->input(to_port).replace_source_output(from_node->output(from_port));
            }
        }
        if (auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(to_node)) {
            if (fw_node->inputs().size() <= static_cast<size_t>(to_port)) {
                fw_node->set_inputs(
                    ov::OutputVector(fw_node->inputs().size() + (to_port + 1 - fw_node->inputs().size())));
            }
            fw_node->input(to_port).replace_source_output(from_node->output(from_port));
        } else if (auto result_node = std::dynamic_pointer_cast<ov::op::v0::Result>(to_node)) {
            result_node->input(0).replace_source_output(from_node->output(from_port));
        }
    }

    std::vector<std::shared_ptr<ov::op::v0::Parameter>> param_nodes;
    for (auto& n : parameters) {
        auto p = std::dynamic_pointer_cast<ov::op::v0::Parameter>(n);
        if (p)
            param_nodes.push_back(p);
    }
    std::vector<std::shared_ptr<ov::op::v0::Result>> result_nodes;
    for (auto& n : results) {
        auto r = std::dynamic_pointer_cast<ov::op::v0::Result>(n);
        if (r)
            result_nodes.push_back(r);
    }
    auto model = std::make_shared<ov::Model>(result_nodes, param_nodes, net_node.attribute("name").as_string());
    return model;
}
class LightDeserialize {
public:
    // 构造函数，输入xml流和offsetConstMap
    LightDeserialize(std::istream& xmlFile,
                     std::map<int64_t, std::reference_wrapper<ov::ValueAccessor<void>>>& offsetConstMap)
        : m_xmlFile{&xmlFile},
          m_offsetConstMap(offsetConstMap) {}

    // 反序列化主入口
    bool run_on_model(std::shared_ptr<ov::Model>& model) {
        // 1. 读取xml
        pugi::xml_document xml_doc;
        pugi::xml_parse_result result = xml_doc.load(*m_xmlFile);
        if (!result) {
            OPENVINO_THROW("Failed to parse XML: ", result.description());
        }
        pugi::xml_node net_node = xml_doc.child("net");
        if (!net_node) {
            OPENVINO_THROW("No <net> node found in XML");
        }

        // 2. 反序列化
        model = deserializeFunc(net_node, m_offsetConstMap);

        return true;  // 返回true表示model已被替换
    }

private:
    std::istream* m_xmlFile;
    std::map<int64_t, std::reference_wrapper<ov::ValueAccessor<void>>>& m_offsetConstMap;

    // 反序列化函数，输入xml节点和offsetConstMap，返回Model
    static std::shared_ptr<ov::Model> deserializeFunc(
        const pugi::xml_node& net_node,
        std::map<int64_t, std::reference_wrapper<ov::ValueAccessor<void>>>& offsetConstMap) {
        // 1. 解析layers，构建节点
        std::unordered_map<int, std::shared_ptr<ov::Node>> id_node_map;
        std::vector<std::shared_ptr<ov::Node>> parameters;
        std::vector<std::shared_ptr<ov::Node>> results;
        std::vector<std::shared_ptr<ov::Node>> sinks;
        std::vector<std::shared_ptr<ov::Node>> all_nodes;

        // 记录每个layer的输出端口数量
        std::unordered_map<int, size_t> layer_output_ports;

        // 先遍历layers节点，构建所有节点对象
        pugi::xml_node layers_node = net_node.child("layers");
        for (pugi::xml_node layer = layers_node.child("layer"); layer; layer = layer.next_sibling("layer")) {
            int id = layer.attribute("id").as_int();
            std::string type = layer.attribute("type").as_string();
            std::string name = layer.attribute("name").as_string();

            // 解析输入输出端口数量
            size_t output_ports = 0;
            if (auto output = layer.child("output")) {
                for (auto port = output.child("port"); port; port = port.next_sibling("port")) {
                    output_ports++;
                }
            }
            layer_output_ports[id] = output_ports;

            // 解析data属性
            std::map<std::string, std::string> data_attrs;
            if (auto data = layer.child("data")) {
                for (auto attr = data.attributes_begin(); attr != data.attributes_end(); ++attr) {
                    data_attrs[attr->name()] = attr->value();
                }
            }

            // 构建节点
            std::shared_ptr<ov::Node> node;
            if (type == "Parameter") {
                // 解析shape和precision
                auto output = layer.child("output").child("port");
                std::vector<ov::Dimension> shape;
                for (auto dim = output.child("dim"); dim; dim = dim.next_sibling("dim")) {
                    shape.push_back(ov::Dimension(std::stoll(dim.child_value())));
                }
                std::string precision = output.attribute("precision").as_string();
                ov::element::Type elem_type = ov::element::Type(precision);
                node = std::make_shared<ov::op::v0::Parameter>(elem_type, ov::PartialShape(shape));
                node->set_friendly_name(name);
                parameters.push_back(node);
            } else if (type == "Const" || type == "Constant") {
                // 常量节点
                // 需要从offsetConstMap获取数据
                int64_t offset = -1;
                size_t size = 0;
                if (data_attrs.count("offset")) {
                    offset = std::stoll(data_attrs["offset"]);
                }
                if (data_attrs.count("size")) {
                    size = static_cast<size_t>(std::stoull(data_attrs["size"]));
                }
                std::string precision =
                    data_attrs.count("element_type") ? data_attrs["element_type"] : data_attrs["precision"];
                ov::element::Type elem_type = ov::element::Type(precision);

                std::vector<ov::Dimension> shape;
                if (auto shape_attr = data_attrs.find("shape"); shape_attr != data_attrs.end()) {
                    std::stringstream ss(shape_attr->second);
                    std::string dim;
                    while (std::getline(ss, dim, ',')) {
                        shape.push_back(ov::Dimension(std::stoll(dim)));
                    }
                } else if (auto output = layer.child("output").child("port")) {
                    for (auto dim = output.child("dim"); dim; dim = dim.next_sibling("dim")) {
                        shape.push_back(ov::Dimension(std::stoll(dim.child_value())));
                    }
                }

                std::shared_ptr<ov::op::v0::Constant> const_node;
                if (offsetConstMap.count(offset)) {
                    auto& accessor = offsetConstMap[offset].get();
                    // 这里假设accessor是AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>
                    auto* buf_adapter =
                        ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>>(&accessor);
                    if (buf_adapter) {
                        auto buf = buf_adapter->get();
                        const_node = std::make_shared<ov::op::v0::Constant>(elem_type,
                                                                            ov::Shape(shape.begin(), shape.end()),
                                                                            buf->get_ptr(),
                                                                            buf->size());
                    } else {
                        // 兼容字符串常量
                        auto* str_adapter =
                            ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>>(&accessor);
                        if (str_adapter) {
                            auto buf = str_adapter->get();
                            const_node = std::make_shared<ov::op::v0::Constant>(elem_type,
                                                                                ov::Shape(shape.begin(), shape.end()),
                                                                                buf->get_ptr(),
                                                                                buf->size());
                        } else {
                            OPENVINO_THROW("Unsupported constant accessor type for offset ", offset);
                        }
                    }
                } else {
                    // 没有offset数据，填充0
                    std::vector<uint8_t> dummy(elem_type.size() * ov::shape_size(ov::Shape(shape.begin(), shape.end())),
                                               0);
                    const_node = std::make_shared<ov::op::v0::Constant>(elem_type,
                                                                        ov::Shape(shape.begin(), shape.end()),
                                                                        dummy.data());
                }
                const_node->set_friendly_name(name);
                node = const_node;
            } else if (type == "Result") {
                // 结果节点，后续连接
                node = std::make_shared<ov::op::v0::Result>(nullptr);
                node->set_friendly_name(name);
                results.push_back(node);
            } else {
                // 其他类型节点，使用FrameworkNode占位
                auto fw_node = std::make_shared<ov::op::util::FrameworkNode>(ov::OutputVector{}, output_ports);
                fw_node->set_friendly_name(name);
                node = fw_node;
            }
            id_node_map[id] = node;
            all_nodes.push_back(node);
        }

        // 2. 解析edges，连接节点
        pugi::xml_node edges_node = net_node.child("edges");
        for (pugi::xml_node edge = edges_node.child("edge"); edge; edge = edge.next_sibling("edge")) {
            int from_layer = edge.attribute("from-layer").as_int();
            int from_port = edge.attribute("from-port").as_int();
            int to_layer = edge.attribute("to-layer").as_int();
            int to_port = edge.attribute("to-port").as_int();

            auto from_node = id_node_map.at(from_layer);
            auto to_node = id_node_map.at(to_layer);

            // 连接输出到输入
            // 这里只能处理FrameworkNode的输入，真实实现应根据type动态分派
            // 连接输出到输入，支持多种类型节点
            if (auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(to_node)) {
                // 先扩展inputs
                if (fw_node->inputs().size() <= static_cast<size_t>(to_port)) {
                    fw_node->set_inputs(
                        ov::OutputVector(fw_node->inputs().size() + (to_port + 1 - fw_node->inputs().size())));
                }
                fw_node->input(to_port).replace_source_output(from_node->output(from_port));
            } else if (auto result_node = std::dynamic_pointer_cast<ov::op::v0::Result>(to_node)) {
                // Result节点只有一个输入
                result_node->input(0).replace_source_output(from_node->output(from_port));
            } else if (auto conv_node = std::dynamic_pointer_cast<ov::op::v1::Convolution>(to_node)) {
                conv_node->input(to_port).replace_source_output(from_node->output(from_port));
            } else if (auto group_conv_node = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(to_node)) {
                group_conv_node->input(to_port).replace_source_output(from_node->output(from_port));
            } else if (auto bin_conv_node = std::dynamic_pointer_cast<ov::op::v1::BinaryConvolution>(to_node)) {
                bin_conv_node->input(to_port).replace_source_output(from_node->output(from_port));
            } else if (auto loop_node = std::dynamic_pointer_cast<ov::op::v5::Loop>(to_node)) {
                loop_node->input(to_port).replace_source_output(from_node->output(from_port));
            } else if (auto lstm_cell_node = std::dynamic_pointer_cast<ov::op::v0::LSTMCell>(to_node)) {
                lstm_cell_node->input(to_port).replace_source_output(from_node->output(from_port));
            } else if (auto param_node = std::dynamic_pointer_cast<ov::op::v0::Parameter>(to_node)) {
                // Parameter节点通常没有输入，不处理
            } else if (auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(to_node)) {
                // Constant节点没有输入，不处理
            } else {
                // 默认尝试通用方式
                if (to_node->inputs().size() > static_cast<size_t>(to_port)) {
                    to_node->input(to_port).replace_source_output(from_node->output(from_port));
                }
            }
            if (auto fw_node = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(to_node)) {
                // 先扩展inputs
                if (fw_node->inputs().size() <= static_cast<size_t>(to_port)) {
                    fw_node->set_inputs(
                        ov::OutputVector(fw_node->inputs().size() + (to_port + 1 - fw_node->inputs().size())));
                }
                fw_node->input(to_port).replace_source_output(from_node->output(from_port));
            } else if (auto result_node = std::dynamic_pointer_cast<ov::op::v0::Result>(to_node)) {
                // Result节点只有一个输入
                result_node->input(0).replace_source_output(from_node->output(from_port));
            }
            // 其他类型节点可扩展
        }

        // 3. 处理常量offsetConstMap
        // 已在构建Const节点时处理

        // 4. 组装成ov::Model
        // 收集所有Parameter和Result节点
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> param_nodes;
        for (auto& n : parameters) {
            auto p = std::dynamic_pointer_cast<ov::op::v0::Parameter>(n);
            if (p)
                param_nodes.push_back(p);
        }
        std::vector<std::shared_ptr<ov::op::v0::Result>> result_nodes;
        for (auto& n : results) {
            auto r = std::dynamic_pointer_cast<ov::op::v0::Result>(n);
            if (r)
                result_nodes.push_back(r);
        }
        auto model = std::make_shared<ov::Model>(result_nodes, param_nodes, net_node.attribute("name").as_string());
        return model;
        // 这里只做简单示例，实际应完整实现IR->Model的反序列化
        // 1. 解析layers，构建节点
        // 2. 解析edges，连接节点
        // 3. 处理常量offsetConstMap
        // 4. 组装成ov::Model
        // 这里只返回空Model，实际应完整实现
        // return std::make_shared<ov::Model>();
    }
};

}  // namespace pass
}  // namespace ov
