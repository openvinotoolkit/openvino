// Copyright (C) 2024 Intel Corporationov::npuw::
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "../../../logging.hpp"
#include "intel_npu/config/npuw.hpp"

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

size_t ov::npuw::online::util::getMinGraphSize(const ::intel_npu::Config& cfg) {
    std::size_t min_size = cfg.get<::intel_npu::NPUW_ONLINE_MIN_SIZE>();

    // Sanity check
    if (min_size < 10) {
        LOG_WARN("Minimum possible partitioning size is too small: " << min_size << ", using a default value of 10.");
        min_size = 10;
    }

    LOG_INFO("Online partitioning will continue until there are " << min_size << " or less subgraphs.");

    return min_size;
}

size_t ov::npuw::online::util::getMinRepBlocks(const ::intel_npu::Config& cfg) {
    std::size_t min_size = cfg.get<::intel_npu::NPUW_ONLINE_KEEP_BLOCKS>();

    return min_size;
}

size_t ov::npuw::online::util::getMinRepBlockSize(const ::intel_npu::Config& cfg) {
    std::size_t min_size = cfg.get<::intel_npu::NPUW_ONLINE_KEEP_BLOCK_SIZE>();

    return min_size;
}

std::vector<ov::npuw::online::Avoid> ov::npuw::online::util::getAvoids(const ::intel_npu::Config& cfg) {
    std::vector<ov::npuw::online::Avoid> avoids;

    std::string avoids_opt = cfg.getString<::intel_npu::NPUW_ONLINE_AVOID>();
    if (avoids_opt.empty()) {
        return {};
    }

    std::string s = std::move(avoids_opt);

    size_t pos = 0;
    size_t start = 0;
    std::string token;

    while ((pos = s.find(',', start)) != std::string::npos) {
        token = s.substr(start, pos - start);
        auto avoid_opt = util::parseAvoid(token);
        // Check that parsing was a success
        if (avoid_opt) {
            avoids.push_back(*avoid_opt);
        }
        start = pos + 1;
    }

    // Parse the tail
    auto avoid_opt = util::parseAvoid(s.substr(start, s.size() - start));
    // Check that parsing was a success
    if (avoid_opt) {
        avoids.push_back(*avoid_opt);
    }

    if (!avoids.empty()) {
        LOG_INFO("Online partitioning will avoid running subgraphs containing specified patterns on their respective "
                 "devices.");
    } else {
        LOG_WARN("Incorect pattern in OPENVINO_NPUW_AVOID!"
                 << " Please, follow the example: Op:Select/NPU,P:RMSNorm/NPU."
                 << " No avoid rules will be taken into account during execution!");
    }

    return avoids;
}

std::vector<ov::npuw::online::Isolate> ov::npuw::online::util::getIsolates(const ::intel_npu::Config& cfg) {
    return ov::npuw::online::util::getIsolates(cfg.getString<::intel_npu::NPUW_ONLINE_ISOLATE>());
}

std::vector<ov::npuw::online::Isolate> ov::npuw::online::util::getIsolates(const std::string& isolates_unparsed) {
    if (isolates_unparsed.empty()) {
        return {};
    }

    std::vector<Isolate> isolates;
    std::string s = isolates_unparsed;

    auto preset_iter = ov::npuw::online::util::ISOL_PRESETS.find(s);
    if (preset_iter != ov::npuw::online::util::ISOL_PRESETS.end()) {
        s = preset_iter->second;
    }

    size_t pos = 0;
    size_t start = 0;
    std::string token;

    while ((pos = s.find(',', start)) != std::string::npos) {
        token = s.substr(start, pos - start);
        auto isolate_opt = util::parseIsolate(token);
        // Check that parsing was a success
        if (isolate_opt) {
            isolates.push_back(*isolate_opt);
        }
        start = pos + 1;
    }

    // Parse the tail
    auto isolate_opt = util::parseIsolate(s.substr(start, s.size() - start));
    // Check that parsing was a success
    if (isolate_opt) {
        isolates.push_back(*isolate_opt);
    }

    if (!isolates.empty()) {
        LOG_INFO("Online partitioning will isolate subgraphs containing specified patterns.");
    } else {
        LOG_WARN("Incorect pattern in NPUW_ONLINE_ISOLATE! No isolate rules will be taken into account during "
                 "partitioning!");
    }

    return isolates;
}

std::vector<std::string> ov::npuw::online::util::getNoFolds(const ::intel_npu::Config& cfg) {
    return ov::npuw::online::util::getNoFolds(cfg.getString<::intel_npu::NPUW_ONLINE_NO_FOLD>());
}

std::vector<std::string> ov::npuw::online::util::getNoFolds(const std::string& nofolds_unparsed) {
    if (nofolds_unparsed.empty()) {
        return {};
    }

    std::vector<std::string> nofolds;
    std::string s = std::move(nofolds_unparsed);

    size_t pos = 0;
    size_t start = 0;
    std::string token;

    while ((pos = s.find(',', start)) != std::string::npos) {
        token = s.substr(start, pos - start);
        if (!token.empty()) {
            nofolds.push_back(token);
        }
        start = pos + 1;
    }

    // Parse the tail
    std::string tail = s.substr(start, s.size() - start);
    if (!tail.empty()) {
        nofolds.push_back(tail);
    }

    if (!nofolds.empty()) {
        LOG_INFO("Online partitioning will mark specified tags as non-foldable.");
    } else {
        LOG_WARN("Incorect pattern in NPUW_ONLINE_NO_FOLD!"
                 << " Please, follow the example: " << "compute,compute2. "
                 << "No non-fold rules will be taken into account during partitioning!");
    }

    return nofolds;
}

ov::npuw::online::PassContext::PassContext(const ::intel_npu::Config& cfg) {
    min_graph_size = ov::npuw::online::util::getMinGraphSize(cfg);
    keep_blocks = ov::npuw::online::util::getMinRepBlocks(cfg);
    keep_block_size = ov::npuw::online::util::getMinRepBlockSize(cfg);
    avoids = ov::npuw::online::util::getAvoids(cfg);
    isolates = ov::npuw::online::util::getIsolates(cfg);
    nofolds = ov::npuw::online::util::getNoFolds(cfg);
}
