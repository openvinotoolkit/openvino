// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>
#include <utility>
#ifdef SNIPPETS_DEBUG_CAPS

#    include "openvino/util/common_util.hpp"
#    include "openvino/util/file_util.hpp"
#    include "snippets/lowered/linear_ir.hpp"
#    include "snippets/lowered/pass/serialize_control_flow.hpp"
#    include "snippets/lowered/pass/serialize_data_flow.hpp"
#    include "snippets/utils/debug_caps_config.hpp"

namespace ov::snippets {

class LIRPassDump {
public:
    explicit LIRPassDump(const lowered::LinearIR& linear_ir, std::string pass_name)
        : linear_ir(linear_ir),
          pass_name(std::move(pass_name)),
          debug_config(*linear_ir.get_config().debug_config) {
        dump("_in");
    }
    LIRPassDump(const lowered::LinearIR& linear_ir, std::string pass_name, std::string name_prefix)
        : linear_ir(linear_ir),
          pass_name(std::move(pass_name)),
          name_prefix(std::move(name_prefix)),
          debug_config(*linear_ir.get_config().debug_config) {
        dump("_in");
    }
    ~LIRPassDump() {
        dump("_out");
    }

private:
    void dump(const std::string&& postfix) const {
        static int num = 0;  // just to keep dumped IRs ordered in filesystem
        auto pathAndName = debug_config.dumpLIR.dir + "/";
        const bool use_subgraph_prefix =
            ov::util::to_lower(debug_config.dumpLIR.name_modifier) == std::string("subgraph_name");
        if (use_subgraph_prefix && !name_prefix.empty()) {
            pathAndName += name_prefix + "_";
        }
        pathAndName += "lir_";

        ov::util::create_directory_recursive(debug_config.dumpLIR.dir);

        if (debug_config.dumpLIR.format.filter[DebugCapsConfig::LIRFormatFilter::controlFlow]) {
            std::string xml_path =
                pathAndName + std::to_string(num) + '_' + pass_name + "_control_flow" + postfix + ".xml";
            lowered::pass::SerializeControlFlow SerializeLIR(xml_path);
            SerializeLIR.run(linear_ir);
        }
        if (debug_config.dumpLIR.format.filter[DebugCapsConfig::LIRFormatFilter::dataFlow]) {
            std::string xml_path =
                pathAndName + std::to_string(num) + '_' + pass_name + "_data_flow" + postfix + ".xml";
            lowered::pass::SerializeDataFlow SerializeLIR(xml_path);
            SerializeLIR.run(linear_ir);
        }
        num++;
    }

    const lowered::LinearIR& linear_ir;
    const std::string pass_name;
    const std::string name_prefix;
    const DebugCapsConfig& debug_config;
};

}  // namespace ov::snippets

#    define SNIPPETS_DEBUG_LIR_PASS_DUMP(_linear_ir, _pass)                                                       \
        auto dumpLIR = (_linear_ir).get_config().debug_config->dumpLIR;                                           \
        auto pass_name = std::string((_pass)->get_type_name());                                                   \
        auto dump_name = dumpLIR.passes;                                                                          \
        auto dumperPtr =                                                                                          \
            ((std::find(dump_name.begin(), dump_name.end(), ov::util::to_lower(pass_name)) != dump_name.end()) || \
             (std::find(dump_name.begin(), dump_name.end(), "all") != dump_name.end()))                           \
                ? std::unique_ptr<LIRPassDump>(new LIRPassDump(_linear_ir, pass_name))                            \
                : nullptr
#else
#    define SNIPPETS_DEBUG_LIR_PASS_DUMP(_linear_ir, _pass)
#endif  // SNIPPETS_DEBUG_CAPS