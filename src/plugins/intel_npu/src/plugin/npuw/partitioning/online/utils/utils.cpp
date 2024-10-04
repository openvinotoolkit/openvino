// Copyright (C) 2024 Intel Corporationov::npuw::
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "../../../logging.hpp"

using ov::npuw::online::Repeated;
using ov::npuw::online::util::ReadAttributes;

// FIXME: metadesc should be hash of layer's meta, not string
std::string ov::npuw::online::util::getMetaDesc(const std::shared_ptr<ov::Node>& ov_node) {
    std::stringstream ss;
    ss << ov_node->description() << ' ';

    for (const auto& input : ov_node->inputs()) {
        ss << input.get_element_type() << ' ' << input.get_shape() << ' ';
    }
    for (const auto& output : ov_node->outputs()) {
        ss << output.get_element_type() << ' ' << output.get_shape() << ' ';
    }

    ReadAttributes visitor_node;
    ov_node->visit_attributes(visitor_node);
    auto node_bodies = visitor_node.get_attributes_map();

    for (const auto& body : node_bodies) {
        ss << body.first << ' ' << body.second << ' ';
    }

    // FIXME: should be { self type. self inputs. self outputs. self attrs. self data }
    //        can't extract data here?
    return ss.str();
}

std::string ov::npuw::online::util::repeated_id(const std::shared_ptr<Repeated>& ptr) {
    if (!ptr) {
        OPENVINO_THROW("Online partitioning tried to convert nullptr Repeated to id!");
    }
    const void* address = static_cast<const void*>(ptr.get());
    std::stringstream ss;
    ss << address;
    return ss.str();
}

std::tuple<ov::npuw::online::PatternType, std::string, std::string> ov::npuw::online::util::parse(
    const std::string& s) {
    auto pos_col = s.find(':');
    auto pos_sl = s.find('/');

    if (pos_col == std::string::npos || pos_sl == std::string::npos) {
        LOG_WARN("Incorrect pattern: " << s << ". Please, separate a device or tag with / and pattern type with :."
                                       << " Rule " << s << " is ommited!");
        return {};
    }

    auto type = s.substr(0, pos_col);
    auto pattern = s.substr(pos_col + 1, pos_sl - pos_col - 1);
    auto device_or_tag = s.substr(pos_sl + 1, s.size() - pos_sl - 1);

    if (type != "Op" && type != "P") {
        LOG_WARN("Incorrect pattern type: " << type << ". Please, use either Op for operation or P for pattern."
                                            << " Rule " << s << " is ommited!");
        return {};
    }

    auto pattern_type = type == "Op" ? PatternType::OP : PatternType::PATTERN;
    return std::make_tuple(pattern_type, pattern, device_or_tag);
}

std::optional<ov::npuw::online::Avoid> ov::npuw::online::util::parseAvoid(const std::string& s) {
    auto parsed = parse(s);

    if (std::get<1>(parsed).empty() || std::get<2>(parsed).empty()) {
        return std::nullopt;
    }

    Avoid avoid;
    avoid.type = std::get<0>(parsed);
    avoid.pattern = std::get<1>(parsed);
    avoid.device = std::get<2>(parsed);

    return std::optional<Avoid>{avoid};
}

std::optional<ov::npuw::online::Isolate> ov::npuw::online::util::parseIsolate(const std::string& s) {
    auto parsed = parse(s);

    if (std::get<1>(parsed).empty() || std::get<2>(parsed).empty()) {
        return std::nullopt;
    }

    Isolate isolate;
    isolate.type = std::get<0>(parsed);
    isolate.pattern = std::get<1>(parsed);
    isolate.tag = std::get<2>(parsed);

    return std::optional<Isolate>{isolate};
}
