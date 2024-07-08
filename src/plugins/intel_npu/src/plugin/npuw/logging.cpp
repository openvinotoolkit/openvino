// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logging.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace {
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
const char* get_env(const std::vector<std::string>& list_to_try) {
    for (auto&& key : list_to_try) {
        const char* pstr = std::getenv(key.c_str());
        if (pstr)
            return pstr;
    }
    return nullptr;
}
#endif
}  // anonymous namespace

ov::npuw::LogLevel ov::npuw::get_log_level() {
    static LogLevel log_level = LogLevel::None;
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::once_flag flag;

    std::call_once(flag, []() {
        const auto* log_opt = get_env({"OPENVINO_NPUW_LOG_LEVEL", "OPENVINO_NPUW_LOG"});
        if (!log_opt) {
            return;
        } else if (log_opt == std::string("ERROR")) {
            log_level = ov::npuw::LogLevel::Error;
        } else if (log_opt == std::string("WARNING")) {
            log_level = ov::npuw::LogLevel::Warning;
        } else if (log_opt == std::string("INFO")) {
            log_level = ov::npuw::LogLevel::Info;
        } else if (log_opt == std::string("VERBOSE")) {
            log_level = ov::npuw::LogLevel::Verbose;
        } else if (log_opt == std::string("DEBUG")) {
            log_level = LogLevel::Debug;
        }
    });
#endif
    return log_level;
}

thread_local int ov::npuw::__logging_indent__::this_indent = 0;

ov::npuw::__logging_indent__::__logging_indent__() {
    ++this_indent;
}

ov::npuw::__logging_indent__::~__logging_indent__() {
    this_indent = std::max(0, this_indent - 1);
}

int ov::npuw::__logging_indent__::__level__() {
    return this_indent;
}

void ov::npuw::dump_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& base_path) {
    if (!tensor->is_continuous()) {
        LOG_ERROR("Failed to dump blob " << base_path << ": it is not continuous");
        return;
    }

    const auto bin_path = base_path + ".bin";
    {
        std::ofstream bin_file(bin_path, std::ios_base::out | std::ios_base::binary);
        bin_file.write(static_cast<const char*>(tensor->data()), static_cast<std::streamsize>(tensor->get_byte_size()));
        LOG_INFO("Wrote file " << bin_path << "...");
    }
    const auto meta_path = base_path + ".txt";
    {
        std::ofstream meta_file(meta_path);
        meta_file << tensor->get_element_type() << ' ' << tensor->get_shape() << std::endl;
        LOG_INFO("Wrote file " << meta_path << "...");
    }
}

static void dump_file_list(const std::string list_path, const std::vector<std::string>& base_names) {
    std::ofstream list_file(list_path);

    if (base_names.empty()) {
        return;  // That's it. But create file anyway!
    }

    auto iter = base_names.begin();
    list_file << *iter << ".bin";
    while (++iter != base_names.end()) {
        list_file << ";" << *iter << ".bin";
    }
}

void ov::npuw::dump_input_list(const std::string base_name, const std::vector<std::string>& base_input_names) {
    // dump a list of input/output files for sit.py's --inputs argument
    // note the file has no newline to allow use like
    //
    // sit.py --inputs "$(cat model_ilist.txt)"
    const auto ilist_path = base_name + "_ilist.txt";
    dump_file_list(ilist_path, base_input_names);
    LOG_INFO("Wrote input list " << ilist_path << "...");
}

void ov::npuw::dump_output_list(const std::string base_name, const std::vector<std::string>& base_output_names) {
    const auto olist_path = base_name + "_olist.txt";
    dump_file_list(olist_path, base_output_names);
    LOG_INFO("Wrote output list " << olist_path << "...");
}

void ov::npuw::dump_failure(const std::shared_ptr<ov::Model>& model, const std::string& device, const char* extra) {
    const auto model_path = "failed_" + model->get_friendly_name() + ".xml";
    const auto extra_path = "failed_" + model->get_friendly_name() + ".txt";

    ov::save_model(model, model_path);

    std::ofstream details(extra_path, std::ios_base::app);
    auto t = std::time(nullptr);
    const auto& tm = *std::localtime(&t);
    details << std::put_time(&tm, "%d-%m-%Y %H:%M:%S") << ": Failed to compile submodel for " << device << ", error:\n"
            << extra << "\n"
            << std::endl;

    LOG_INFO("Saved model to " << model_path << " with details in " << extra_path);
}
