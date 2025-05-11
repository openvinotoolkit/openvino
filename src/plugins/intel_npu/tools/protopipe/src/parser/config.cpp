//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "parser/config.hpp"

#include "utils/error.hpp"
#include "utils/logger.hpp"

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>  // depth

namespace fs = std::filesystem;

struct GlobalOptions {
    std::string blob_dir = ".";
    std::string model_dir = ".";
    std::string device_name = "NPU";
    std::string log_level = "NONE";
    std::string compiler_type = "DRIVER";
    std::optional<std::filesystem::path> save_validation_outputs;
};

struct Network {
    std::string tag;
    InferenceParams params;
    LayerVariantAttr<std::string> input_data;
    LayerVariantAttr<std::string> output_data;
    LayerVariantAttr<IRandomGenerator::Ptr> initializers;
    LayerVariantAttr<IAccuracyMetric::Ptr> accuracy_metrics;
};

struct InferOp {
    InferenceParams params;
    LayerVariantAttr<std::string> input_data;
    LayerVariantAttr<std::string> output_data;
    LayerVariantAttr<IRandomGenerator::Ptr> initializers;
    LayerVariantAttr<IAccuracyMetric::Ptr> accuracy_metrics;
};

struct CPUOp {
    uint64_t time_in_us;
};

struct CompoundOp {
    uint64_t repeat_count;
    InferenceParamsMap params;
    ScenarioGraph subgraph;
};

struct OpDesc {
    std::string tag;
    using OpType = std::variant<InferOp, CPUOp, CompoundOp>;
    OpType op;
};

// NB: Handles duplicating tags.
class TagsManager {
public:
    std::string add(const std::string& tag);

private:
    std::unordered_multiset<std::string> m_tags;
};

std::string TagsManager::add(const std::string& tag) {
    std::string t = tag;
    m_tags.insert(t);
    const auto c = m_tags.count(t);
    if (c > 1) {
        t += "-" + std::to_string(c);
    }
    return t;
}

static LogLevel toLogLevel(const std::string& lvl) {
    if (lvl == "NONE")
        return LogLevel::None;
    if (lvl == "INFO")
        return LogLevel::Info;
    if (lvl == "DEBUG")
        return LogLevel::Debug;
    THROW_ERROR("Unsupported log level: " << lvl);
}

static int toDepth(const std::string& prec) {
    if (prec == "FP32")
        return CV_32F;
    if (prec == "FP16")
        return CV_16F;
    if (prec == "U8")
        return CV_8U;
    if (prec == "I32")
        return CV_32S;
    throw std::logic_error("Unsupported precision type: " + prec);
}

static AttrMap<int> toDepth(const AttrMap<std::string>& attrmap) {
    AttrMap<int> depthmap;
    for (const auto& [name, str_depth] : attrmap) {
        depthmap.emplace(name, toDepth(str_depth));
    }
    return depthmap;
}

static LayerVariantAttr<int> toDepth(const LayerVariantAttr<std::string>& attr) {
    LayerVariantAttr<int> depthattr;
    if (std::holds_alternative<std::string>(attr)) {
        depthattr = toDepth(std::get<std::string>(attr));
    } else {
        depthattr = toDepth(std::get<AttrMap<std::string>>(attr));
    }
    return depthattr;
}

static std::string toPriority(const std::string& priority) {
    if (priority == "LOW") {
        return "LOW";
    }
    if (priority == "NORMAL") {
        return "MEDIUM";
    }
    if (priority == "HIGH") {
        return "HIGH";
    }
    throw std::logic_error("Unsupported model priority: " + priority);
}

static ScenarioGraph buildGraph(const std::vector<OpDesc>& op_descs,
                                const std::vector<std::vector<std::string>>& connections);

namespace YAML {

template <typename T>
struct convert<std::vector<T>> {
    static bool decode(const Node& node, std::vector<T>& vec) {
        if (!node.IsSequence()) {
            return false;
        }

        for (auto& child : node) {
            vec.push_back(child.as<T>());
        }
        return true;
    }
};

template <typename K, typename V>
struct convert<std::map<K, V>> {
    static bool decode(const Node& node, std::map<K, V>& map) {
        if (!node.IsMap()) {
            return false;
        }
        for (const auto& itr : node) {
            map.emplace(itr.first.as<K>(), itr.second.as<V>());
        }
        return true;
    }
};

template <typename T>
struct convert<LayerVariantAttr<T>> {
    static bool decode(const Node& node, LayerVariantAttr<T>& layer_attr) {
        // Note: "metric" and "random" entries in config are always presented
        //       as maps.
        //       To differ passed variants for them between "one value for all layers"
        //       and "map of layers to values", it is needed to check that map of maps
        //       is passed for them.
        if constexpr (std::is_same_v<IAccuracyMetric::Ptr, T> || std::is_same_v<IRandomGenerator::Ptr, T>) {
            if (node.IsMap() && (node.size() > 0) && node.begin()->second.IsMap()) {
                layer_attr = node.as<std::map<std::string, T>>();
            } else {
                layer_attr = node.as<T>();
            }
        } else {
            if (node.IsMap()) {
                layer_attr = node.as<std::map<std::string, T>>();
            } else {
                layer_attr = node.as<T>();
            }
        }
        return true;
    }
};

template <>
struct convert<UniformGenerator::Ptr> {
    static bool decode(const Node& node, UniformGenerator::Ptr& generator) {
        if (!node["low"]) {
            THROW_ERROR("Uniform distribution must have \"low\" attribute");
        }
        if (!node["high"]) {
            THROW_ERROR("Uniform distribution must have \"high\" attribute");
        }
        generator = std::make_shared<UniformGenerator>(node["low"].as<double>(), node["high"].as<double>());
        return true;
    }
};

template <>
struct convert<IRandomGenerator::Ptr> {
    static bool decode(const Node& node, IRandomGenerator::Ptr& generator) {
        if (!node["dist"]) {
            THROW_ERROR("\"random\" must have \"dist\" attribute!");
        }
        const auto dist = node["dist"].as<std::string>();
        if (dist == "uniform") {
            generator = node.as<UniformGenerator::Ptr>();
        } else {
            THROW_ERROR("Unsupported random distribution: \"" << dist << "\"");
        }
        return true;
    }
};

template <>
struct convert<Norm::Ptr> {
    static bool decode(const Node& node, Norm::Ptr& metric) {
        // NB: If bigger than tolerance - fail.
        if (!node["tolerance"]) {
            THROW_ERROR("Metric \"norm\" must have \"tolerance\" attribute!");
        }
        const auto tolerance = node["tolerance"].as<double>();
        metric = std::make_shared<Norm>(tolerance);
        return true;
    }
};

template <>
struct convert<Cosine::Ptr> {
    static bool decode(const Node& node, Cosine::Ptr& metric) {
        // NB: If lower than threshold - fail.
        if (!node["threshold"]) {
            THROW_ERROR("Metric \"cosine\" must have \"threshold\" attribute!");
        }
        const auto threshold = node["threshold"].as<double>();
        metric = std::make_shared<Cosine>(threshold);
        return true;
    }
};

template <>
struct convert<NRMSE::Ptr> {
    static bool decode(const Node& node, NRMSE::Ptr& metric) {
        // NB: If bigger than tolerance - fail.
        if (!node["tolerance"]) {
            THROW_ERROR("Metric \"nrmse\" must have \"tolerance\" attribute!");
        }
        const auto tolerance = node["tolerance"].as<double>();
        metric = std::make_shared<NRMSE>(tolerance);
        return true;
    }
};

template <>
struct convert<IAccuracyMetric::Ptr> {
    static bool decode(const Node& node, IAccuracyMetric::Ptr& metric) {
        const auto type = node["name"].as<std::string>();
        if (type == "norm") {
            metric = node.as<Norm::Ptr>();
        } else if (type == "cosine") {
            metric = node.as<Cosine::Ptr>();
        } else if (type == "nrmse") {
            metric = node.as<NRMSE::Ptr>();
        } else {
            THROW_ERROR("Unsupported metric type: " << type);
        }
        return true;
    }
};

template <>
struct convert<GlobalOptions> {
    static bool decode(const Node& node, GlobalOptions& opts) {
        if (node["model_dir"]) {
            if (!node["model_dir"]["local"]) {
                THROW_ERROR("\"model_dir\" must contain \"local\" key!");
            }
            opts.model_dir = node["model_dir"]["local"].as<std::string>();
        }

        if (node["blob_dir"]) {
            if (!node["blob_dir"]["local"]) {
                THROW_ERROR("\"blob_dir\" must contain \"local\" key!");
            }
            opts.blob_dir = node["blob_dir"]["local"].as<std::string>();
        }

        if (node["device_name"]) {
            opts.device_name = node["device_name"].as<std::string>();
        }

        if (node["log_level"]) {
            opts.log_level = node["log_level"].as<std::string>();
        }

        if (node["compiler_type"]) {
            opts.compiler_type = node["compiler_type"].as<std::string>();
        }

        if (node["save_validation_outputs"]) {
            const auto path = node["save_validation_outputs"].as<std::string>();
            opts.save_validation_outputs = std::make_optional(std::filesystem::path{path});
        }

        return true;
    }
};

template <>
struct convert<OpenVINOParams> {
    static bool decode(const Node& node, OpenVINOParams& params) {
        // FIXME: Worth to separate these two
        const auto name = node["name"] ? node["name"].as<std::string>() : node["path"].as<std::string>();
        fs::path path{name};
        if (path.extension() == ".xml") {
            auto bin_path = path;
            bin_path.replace_extension(".bin");
            params.path = OpenVINOParams::ModelPath{path.string(), bin_path.string()};
        } else if (path.extension() == ".blob") {
            params.path = OpenVINOParams::BlobPath{path.string()};
        } else {
            // NB: *.onnx, *.pdpd, and any other format supported in future
            params.path = OpenVINOParams::ModelPath{path.string(), "" /*weights*/};
        }
        // NB: If "device" isn't presented in config for network,
        // the device specified globally will be substitued later on
        if (node["device"]) {
            params.device = node["device"].as<std::string>();
        }

        if (node["ip"]) {
            params.input_precision = toDepth(node["ip"].as<LayerVariantAttr<std::string>>());
        }

        if (node["op"]) {
            params.output_precision = toDepth(node["op"].as<LayerVariantAttr<std::string>>());
        }

        if (node["il"]) {
            params.input_layout = node["il"].as<LayerVariantAttr<std::string>>();
        }

        if (node["ol"]) {
            params.output_layout = node["ol"].as<LayerVariantAttr<std::string>>();
        }

        if (node["iml"]) {
            params.input_model_layout = node["iml"].as<LayerVariantAttr<std::string>>();
        }

        if (node["oml"]) {
            params.output_model_layout = node["oml"].as<LayerVariantAttr<std::string>>();
        }

        if (node["reshape"]) {
            params.reshape = node["reshape"].as<LayerVariantAttr<std::vector<size_t>>> ();
        }

        if (node["config"]) {
            params.config = node["config"].as<std::map<std::string, std::string>>();
        }

        // NB: Note, it should be handled after "config" is set above
        if (node["priority"]) {
            params.config.emplace("MODEL_PRIORITY", toPriority(node["priority"].as<std::string>()));
        }

        if (node["nireq"]) {
            params.nireq = node["nireq"].as<size_t>();
        }
        return true;
    }
};

template <>
struct convert<ONNXRTParams::OpenVINO> {
    static bool decode(const Node& node, ONNXRTParams::OpenVINO& ov_ep) {
        if (node["params"]) {
            ov_ep.params_map = node["params"].as<std::map<std::string, std::string>>();
        }
        if (node["device_type"]) {
            std::string device_type = node["device_type"].as<std::string>();
            // Check if device_type already exists in params_map (collision check)
            if (ov_ep.params_map.count("device_type") > 0) {
                THROW_ERROR("Configuration error: 'device_type' has already been specified in the params.");
            } else {
                ov_ep.params_map["device_type"] = device_type;
            }
        }
        return true;
    }
};

template <>
struct convert<ONNXRTParams::EP> {
    static bool decode(const Node& node, ONNXRTParams::EP& ep) {
        const auto ep_name = node["name"].as<std::string>();
        if (ep_name == "OV") {
            ep = node.as<ONNXRTParams::OpenVINO>();
        } else {
            THROW_ERROR("Unsupported \"ep name\" value: " << ep_name);
        }
        return true;
    }
};

template <>
struct convert<ONNXRTParams> {
    static bool decode(const Node& node, ONNXRTParams& params) {
        // FIXME: Worth to separate these two
        params.model_path = node["name"] ? node["name"].as<std::string>() : node["path"].as<std::string>();
        if (node["session_options"]) {
            params.session_options = node["session_options"].as<std::map<std::string, std::string>>();
        }
        if (node["ep"]) {
            params.ep = node["ep"].as<ONNXRTParams::EP>();
        }
        if (node["opt_level"]) {
            params.opt_level = node["opt_level"].as<int>();
        }
        return true;
    }
};

template <>
struct convert<Network> {
    static bool decode(const Node& node, Network& network) {
        // NB: Take path stem as network tag
        // Note that at this point, it's fine if names aren't unique
        const auto name = node["name"].as<std::string>();
        network.tag = std::filesystem::path{name}.stem().string();
        // NB: OpenVINO is default to keep back compatibility for config syntax
        const auto framework = node["framework"] ? node["framework"].as<std::string>() : "openvino";
        if (framework == "openvino") {
            // NB: Parse OpenVINO model parameters such as path, device, precision, etc
            network.params = node.as<OpenVINOParams>();
        } else if (framework == "onnxrt") {
            network.params = node.as<ONNXRTParams>();
        } else {
            THROW_ERROR("Unsupported \"framework:\" value: " << framework);
        }

        if (node["random"]) {
            network.initializers = node["random"].as<LayerVariantAttr<IRandomGenerator::Ptr>>();
        }
        if (node["metric"]) {
            network.accuracy_metrics = node["metric"].as<LayerVariantAttr<IAccuracyMetric::Ptr>>();
        }
        if (node["input_data"]) {
            network.input_data = node["input_data"].as<LayerVariantAttr<std::string>>();
        }

        if (node["output_data"]) {
            network.output_data = node["output_data"].as<LayerVariantAttr<std::string>>();
        }
        return true;
    }
};

template <>
struct convert<CPUOp> {
    static bool decode(const Node& node, CPUOp& op) {
        // TODO: Assert there are no more options provided
        op.time_in_us = node["time_in_us"] ? node["time_in_us"].as<uint64_t>() : 0u;
        return true;
    }
};

template <>
struct convert<InferOp> {
    static bool decode(const Node& node, InferOp& op) {
        const auto framework = node["framework"] ? node["framework"].as<std::string>() : "openvino";
        if (framework == "openvino") {
            // NB: Parse OpenVINO model parameters such as path, device, precision, etc
            op.params = node.as<OpenVINOParams>();
        } else if (framework == "onnxrt") {
            op.params = node.as<ONNXRTParams>();
        } else {
            THROW_ERROR("Unsupported \"framework:\" value: " << framework);
        }

        if (node["random"]) {
            op.initializers = node["random"].as<LayerVariantAttr<IRandomGenerator::Ptr>>();
        }
        if (node["metric"]) {
            op.accuracy_metrics = node["metric"].as<LayerVariantAttr<IAccuracyMetric::Ptr>>();
        }
        if (node["input_data"]) {
            op.input_data = node["input_data"].as<LayerVariantAttr<std::string>>();
        }

        if (node["output_data"]) {
            op.output_data = node["output_data"].as<LayerVariantAttr<std::string>>();
        }
        return true;
    }
};

template <>
struct convert<OpDesc> {
    static bool decode(const Node& node, OpDesc& opdesc) {
        opdesc.tag = node["tag"].as<std::string>();
        auto type = node["type"] ? node["type"].as<std::string>() : "Infer";
        auto repeat_count = node["repeat_count"] ? node["repeat_count"].as<uint64_t>() : 1u;
        ASSERT(repeat_count > 0)
        if (repeat_count > 1u) {
            // NB: repeat_count > 1u assume that "Compound" operation will be used
            type = "Compound";
        }
        if (type == "Infer") {
            opdesc.op = node.as<InferOp>();
        } else if (type == "CPU") {
            opdesc.op = node.as<CPUOp>();
        } else if (type == "Compound") {
            std::vector<std::vector<std::string>> connections;
            if (node["connections"]) {
                connections = node["connections"].as<std::vector<std::vector<std::string>>>();
            }
            auto op_descs = node["op_desc"].as<std::vector<OpDesc>>();
            InferenceParamsMap inference_params;
            for (const auto& op_desc : op_descs) {
                if (std::holds_alternative<InferOp>(op_desc.op)) {
                    inference_params.emplace(op_desc.tag, std::get<InferOp>(op_desc.op).params);
                }
            }
            opdesc.op = CompoundOp{repeat_count, std::move(inference_params), buildGraph(op_descs, connections)};
        } else {
            THROW_ERROR("Unsupported operation type: \"" << type << "\"!");
        }
        return true;
    }
};

}  // namespace YAML

static std::vector<std::vector<Network>> parseNetworks(const YAML::Node& node) {
    ASSERT(node.IsSequence());
    TagsManager tgs_mngr;
    std::vector<std::vector<Network>> networks_list;
    for (const auto& subnode : node) {
        if (subnode.IsSequence()) {
            networks_list.push_back(subnode.as<std::vector<Network>>());
        } else {
            networks_list.push_back({subnode.as<Network>()});
        }
        // NB: Ensure all network tags are unique!
        for (auto& network : networks_list.back()) {
            network.tag = tgs_mngr.add(network.tag);
        }
    }
    return networks_list;
}

static ScenarioGraph buildGraph(const std::vector<std::vector<Network>>& networks_list, const uint32_t delay_in_us) {
    ScenarioGraph graph;
    auto src = graph.makeSource();
    std::vector<DataNode> producers = {src};
    for (uint32_t list_idx = 0; list_idx < networks_list.size(); ++list_idx) {
        auto& networks = networks_list[list_idx];
        // NB: Delay if specified, will not be added to the beginning
        // and end of the stream, ONLY between models
        if (list_idx != 0u && delay_in_us != 0u) {
            auto delay = graph.makeDelay(delay_in_us);
            for (auto p : producers) {
                graph.link(p, delay);
            }
            producers = {delay.out()};
        }
        std::vector<DataNode> curr_outs;
        curr_outs.reserve(networks.size());
        for (uint32_t net_idx = 0; net_idx < networks.size(); ++net_idx) {
            auto infer = graph.makeInfer(networks[net_idx].tag);
            for (auto p : producers) {
                graph.link(p, infer);
            }
            curr_outs.push_back(infer.out());
        }
        producers = std::move(curr_outs);
    }
    return graph;
}

static InferenceParams adjustParams(OpenVINOParams&& params, const GlobalOptions& opts, const ReplaceBy& replace_by) {
    // NB: Adjust the model path according to base directories provided for blobs & models
    auto& path = params.path;
    if (std::holds_alternative<OpenVINOParams::ModelPath>(path)) {
        auto& model_path = std::get<OpenVINOParams::ModelPath>(path);
        fs::path model_file_path{model_path.model};
        fs::path bin_file_path{model_path.bin};
        if (model_file_path.is_relative()) {
            model_path.model = (opts.model_dir / model_file_path).string();
        }
        if (!model_path.bin.empty() && bin_file_path.is_relative()) {
            model_path.bin = (opts.model_dir / bin_file_path).string();
        }
    } else {
        ASSERT(std::holds_alternative<OpenVINOParams::BlobPath>(path));
        auto& blob_path = std::get<OpenVINOParams::BlobPath>(path);
        fs::path blob_file_path{blob_path.blob};
        if (blob_file_path.is_relative()) {
            blob_path.blob = (opts.blob_dir / blob_file_path).string();
        }
    }
    // NB: Adjust device property based on opts.device_name or replace_by

    if (!replace_by.device.empty()) {
        // NB: ReplaceBy has priority - overwrite
        params.device = replace_by.device;
    } else if (params.device.empty()) {
        // NB: Otherwise, if empty - take the value from global device name
        params.device = opts.device_name;
    }

    // NB: Compiler type is only relevant for NPU device
    if (params.device == "NPU") {
        // NB: Don't overwrite compiler type if it already has been
        // specified explicitly for particular model
        if (const auto it = params.config.find("NPU_COMPILER_TYPE"); it == params.config.end()) {
            params.config.emplace("NPU_COMPILER_TYPE", opts.compiler_type);
        }
    }
    return std::move(params);
}

static InferenceParams adjustParams(ONNXRTParams&& params, const GlobalOptions& opts) {
    fs::path model_file_path{params.model_path};
    if (model_file_path.is_relative()) {
        params.model_path = (opts.model_dir / model_file_path).string();
    }
    return std::move(params);
}

static InferenceParams adjustParams(InferenceParams&& params, const GlobalOptions& opts, const ReplaceBy& replace_by) {
    if (std::holds_alternative<OpenVINOParams>(params)) {
        return adjustParams(std::get<OpenVINOParams>(std::move(params)), opts, replace_by);
    }
    ASSERT(std::holds_alternative<ONNXRTParams>(params));
    return adjustParams(std::get<ONNXRTParams>(std::move(params)), opts);
}

static StreamDesc parseStream(const YAML::Node& node, const GlobalOptions& opts, const std::string& default_name,
                              const ReplaceBy& replace_by) {
    StreamDesc stream;

    // FIXME: Create a function for the duplicate code below
    stream.name = node["name"] ? node["name"].as<std::string>() : default_name;
    stream.frames_interval_in_us = 0u;
    if (node["frames_interval_in_ms"]) {
        stream.frames_interval_in_us = node["frames_interval_in_ms"].as<uint32_t>() * 1000u;
        if (node["target_fps"]) {
            THROW_ERROR("Both \"target_fps\" and \"frames_interval_in_ms\" are defined for the stream: \""
                        << stream.name << "\"! Please specify only one of them as they are mutually exclusive.");
        }
    } else if (node["target_fps"]) {
        uint32_t target_fps = node["target_fps"].as<uint32_t>();
        stream.frames_interval_in_us = (target_fps != 0) ? (1000u * 1000u / target_fps) : 0;
    }

    if (node["target_latency_in_ms"]) {
        stream.target_latency = std::make_optional(node["target_latency_in_ms"].as<double>());
        if (stream.target_latency < 0) {
            THROW_ERROR("\"target_latency_in_ms\" is negative for the stream: \"" << stream.name << "\"!");
        }
    }
    if (node["exec_time_in_secs"]) {
        const auto exec_time_in_secs = node["exec_time_in_secs"].as<uint64_t>();
        stream.criterion = std::make_shared<TimeOut>(exec_time_in_secs * 1'000'000);
    }
    if (node["iteration_count"]) {
        const auto iteration_count = node["iteration_count"].as<uint64_t>();
        stream.criterion = std::make_shared<Iterations>(iteration_count);
    }

    auto networks_list = parseNetworks(node["network"]);
    const auto delay_in_us = node["delay_in_us"] ? node["delay_in_us"].as<uint32_t>() : 0u;
    stream.graph = buildGraph(networks_list, delay_in_us);
    // NB: Collect network parameters
    for (auto& networks : networks_list) {
        for (auto& network : networks) {
            stream.metrics_map.emplace(network.tag, std::move(network.accuracy_metrics));
            stream.initializers_map.emplace(network.tag, std::move(network.initializers));
            stream.input_data_map.emplace(network.tag, std::move(network.input_data));
            stream.output_data_map.emplace(network.tag, std::move(network.output_data));
            stream.infer_params_map.emplace(network.tag, adjustParams(std::move(network.params), opts, replace_by));
        }
    }
    return stream;
}

using DependencyMap = std::unordered_map<std::string, std::unordered_set<std::string>>;

static ScenarioGraph buildGraph(const std::vector<OpDesc>& op_descs,
                                const std::vector<std::vector<std::string>>& connections) {
    // NB: Build the graph based on list of operations and connections between them
    //
    // The algorithm is straightforward:
    // 1) For every operation create corresponding graph node
    // 2) Go though connections and create the dependency map
    // 3) Go through every operation and connect with its dependencies
    //   3.1) If operation has no dependencies, connect it directly with the source

    // NB: For the fast access to operation node by name
    std::unordered_map<std::string, OpNode> op_node_map;
    // NB: To store the list of dependencies for every operation
    std::unordered_map<std::string, std::unordered_set<std::string>> dependency_map;

    // (1) For every operation create corresponding graph node
    ScenarioGraph graph;
    for (const auto& desc : op_descs) {
        // NB: Initialize dependency list for every operation
        dependency_map[desc.tag];
        // FIXME: Implement visitor
        if (std::holds_alternative<InferOp>(desc.op)) {
            op_node_map.emplace(desc.tag, graph.makeInfer(desc.tag));
        } else if (std::holds_alternative<CompoundOp>(desc.op)) {
            const auto& compound = std::get<CompoundOp>(desc.op);
            op_node_map.emplace(
                    desc.tag, graph.makeCompound(compound.repeat_count, compound.subgraph, compound.params, desc.tag));
        } else {
            ASSERT(std::holds_alternative<CPUOp>(desc.op));
            const auto& cpu = std::get<CPUOp>(desc.op);
            op_node_map.emplace(desc.tag, graph.makeDelay(cpu.time_in_us));
        }
    }

    // (2) Go though connections and create the dependency map
    for (const auto& tags : connections) {
        if (tags.size() < 2) {
            THROW_ERROR("Connections list must be at least size of 2!");
        }
        for (uint32_t i = 1; i < tags.size(); ++i) {
            // [A, B, C] - means B depends on A, and C depends on B
            auto deps_it = dependency_map.find(tags[i]);
            if (deps_it == dependency_map.end()) {
                THROW_ERROR("Operation \"" << tags[i] << "\" hasn't been registered in op_desc list!");
            }
            if (tags[i - 1] == tags[i]) {
                THROW_ERROR("Operation \"" << tags[i] << "\" cannot be connected with itself!");
            }
            auto& dep_set = deps_it->second;
            // NB: Check if such connection already exists
            auto is_inserted = deps_it->second.emplace(tags[i - 1]).second;
            if (!is_inserted) {
                THROW_ERROR("Connection between \"" << tags[i - 1] << "\" and \"" << tags[i]
                                                    << "\" operations already exists!");
            }
        }
    }

    // (3) Go through every operation and connect with its dependencies
    auto src = graph.makeSource();
    for (const auto& [tag, deps] : dependency_map) {
        auto op = op_node_map.at(tag);
        // (3.1) If operation has no dependencies, connect it directly to the source
        if (deps.empty()) {
            graph.link(src, op);
        } else {
            for (auto dep_tag : deps) {
                auto dep = op_node_map.at(dep_tag);
                graph.link(dep.out(), op);
            }
        }
    }
    return graph;
}

static StreamDesc parseAdvancedStream(const YAML::Node& node, const GlobalOptions& opts,
                                      const std::string& default_name, const ReplaceBy& replace_by) {
    StreamDesc stream;

    // FIXME: Create a function for the duplicate code below
    stream.name = node["name"] ? node["name"].as<std::string>() : default_name;
    stream.frames_interval_in_us = 0u;
    if (node["frames_interval_in_ms"]) {
        stream.frames_interval_in_us = node["frames_interval_in_ms"].as<uint32_t>() * 1000u;
        if (node["target_fps"]) {
            THROW_ERROR("Both \"target_fps\" and \"frames_interval_in_ms\" are defined for the stream: \""
                        << stream.name << "\"! Please specify only one of them as they are mutually exclusive.");
        }
    } else if (node["target_fps"]) {
        uint32_t target_fps = node["target_fps"].as<uint32_t>();
        stream.frames_interval_in_us = (target_fps != 0) ? (1000u * 1000u / target_fps) : 0;
    }

    if (node["target_latency_in_ms"]) {
        stream.target_latency = std::make_optional(node["target_latency_in_ms"].as<double>());
        if (stream.target_latency < 0) {
            THROW_ERROR("\"target_latency_in_ms\" is negative for the stream: \"" << stream.name << "\"!");
        }
    }
    if (node["exec_time_in_secs"]) {
        const auto exec_time_in_secs = node["exec_time_in_secs"].as<uint64_t>();
        stream.criterion = std::make_shared<TimeOut>(exec_time_in_secs * 1'000'000);
    }
    if (node["iteration_count"]) {
        const auto iteration_count = node["iteration_count"].as<uint64_t>();
        stream.criterion = std::make_shared<Iterations>(iteration_count);
    }

    auto op_descs = node["op_desc"].as<std::vector<OpDesc>>();
    std::vector<std::vector<std::string>> connections;
    if (node["connections"]) {
        connections = node["connections"].as<std::vector<std::vector<std::string>>>();
    }

    for (auto& desc : op_descs) {
        if (std::holds_alternative<InferOp>(desc.op)) {
            auto&& infer = std::get<InferOp>(desc.op);
            stream.metrics_map.emplace(desc.tag, std::move(infer.accuracy_metrics));
            stream.initializers_map.emplace(desc.tag, std::move(infer.initializers));
            stream.input_data_map.emplace(desc.tag, std::move(infer.input_data));
            stream.output_data_map.emplace(desc.tag, std::move(infer.output_data));
            stream.infer_params_map.emplace(desc.tag, adjustParams(std::move(infer.params), opts, replace_by));
        }
        if (std::holds_alternative<CompoundOp>(desc.op)) {
            auto& compound = std::get<CompoundOp>(desc.op);
            InferenceParamsMap& params_map = compound.params;
            for (auto& pair : params_map) {
                pair.second = adjustParams(std::move(pair.second), opts, replace_by);
            }
        }
    }

    stream.graph = buildGraph(op_descs, connections);
    return stream;
}

static std::vector<StreamDesc> parseStreams(const YAML::Node& node, const GlobalOptions& opts,
                                            const ReplaceBy& replace_by) {
    std::vector<StreamDesc> streams;
    uint32_t stream_idx = 0;
    for (const auto& subnode : node) {
        const auto default_name = std::to_string(stream_idx);
        auto stream = subnode["op_desc"] ? parseAdvancedStream(subnode, opts, default_name, replace_by)
                                         : parseStream(subnode, opts, default_name, replace_by);
        streams.push_back(std::move(stream));
        ++stream_idx;
    }
    return streams;
}

static std::vector<ScenarioDesc> parseScenarios(const YAML::Node& node, const GlobalOptions& opts,
                                                const ReplaceBy& replace_by) {
    std::vector<ScenarioDesc> scenarios;
    for (const auto& subnode : node) {
        ScenarioDesc scenario;
        scenario.name = subnode["name"] ? subnode["name"].as<std::string>()
                                        : "multi_inference_" + std::to_string(scenarios.size());
        scenario.streams = parseStreams(subnode["input_stream_list"], opts, replace_by);

        if (opts.save_validation_outputs) {
            for (auto& stream : scenario.streams) {
                const auto& root_path = opts.save_validation_outputs.value();
                std::string stream_dir = "stream_" + stream.name;
                std::filesystem::path stream_outputs_path = root_path / scenario.name / stream_dir;
                stream.per_iter_outputs_path = std::make_optional(std::move(stream_outputs_path));
            }
        }
        scenarios.push_back(std::move(scenario));
    }
    return scenarios;
}

Config parseConfig(const YAML::Node& node, const ReplaceBy& replace_by) {
    const auto global_opts = node.as<GlobalOptions>();

    // FIXME: Perhaps should be done somewhere else...
    Logger::global_lvl = toLogLevel(global_opts.log_level);

    Config config;
    config.scenarios = parseScenarios(node["multi_inference"], global_opts, replace_by);

    ASSERT(!config.scenarios.empty());
    if (node["metric"]) {
        config.metric = node["metric"].as<IAccuracyMetric::Ptr>();
    }
    if (node["random"]) {
        config.initializer = node["random"].as<IRandomGenerator::Ptr>();
    }

    config.disable_high_resolution_timer = false;
    if (node["disable_high_resolution_waitable_timer"]) {
        config.disable_high_resolution_timer = node["disable_high_resolution_waitable_timer"].as<bool>();
    }
    return config;
}
