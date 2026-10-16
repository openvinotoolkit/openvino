// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logging.hpp"

#include <atomic>
#include <cctype>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <system_error>

#include "openvino/runtime/make_tensor.hpp"  // get_tensor_impl

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

bool ov::npuw::debug_groups() {
    static bool do_debug_groups = false;
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::once_flag flag;

    std::call_once(flag, []() {
        const auto* debug_opt = get_env({"OPENVINO_NPUW_DEBUG_GROUPS"});
        if (!debug_opt) {
            return;
        }
        const std::string debug_str(debug_opt);
        do_debug_groups = (debug_str == "YES" || debug_str == "ON" || debug_str == "1");
    });
#endif
    return do_debug_groups;
}

bool ov::npuw::profiling_enabled() {
    static bool do_profiling = false;
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    static std::once_flag flag;

    std::call_once(flag, []() {
        const auto* prof_opt = get_env({"OPENVINO_NPUW_PROF"});
        if (!prof_opt) {
            return;
        }
        const std::string prof_str(prof_opt);
        do_profiling = (prof_str == "YES" || prof_str == "ON" || prof_str == "1");
    });
#endif
    return do_profiling;
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

void ov::npuw::dump_tensor(const ov::SoPtr<ov::ITensor>& input, const std::string& base_path) {
    ov::SoPtr<ov::ITensor> tensor;

    if (input->is_continuous()) {
        tensor = input;
    } else {
        // Create temporary tensor and copy data in. Dumping is never fast, anyway
        tensor = ov::get_tensor_impl(ov::Tensor(input->get_element_type(), input->get_shape()));
        NPUW_ASSERT(tensor._ptr &&
                    "temporary dump tensor has null implementation — element type or shape may be invalid");
        input->copy_to(tensor._ptr);
    }
    NPUW_ASSERT(tensor);

    const auto bin_path = base_path + ".bin";
    {
        std::ofstream bin_file(bin_path, std::ios_base::out | std::ios_base::binary);
        auto blob_size = tensor->get_byte_size();
        if (blob_size > static_cast<decltype(blob_size)>(std::numeric_limits<std::streamsize>::max())) {
            OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
        }
        bin_file.write(static_cast<const char*>(tensor->data()), static_cast<std::streamsize>(blob_size));
        LOG_INFO("Wrote file " << bin_path << "...");
    }
    const auto meta_path = base_path + ".txt";
    {
        std::ofstream meta_file(meta_path);
        meta_file << tensor->get_element_type() << ' ' << tensor->get_shape() << std::endl;
        LOG_INFO("Wrote file " << meta_path << "...");
    }
}

static void dump_file_list(const std::string& list_path, const std::vector<std::string>& base_names) {
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

void ov::npuw::dump_input_list(const std::string& base_name, const std::vector<std::string>& base_input_names) {
    // dump a list of input/output files for sit.py's --inputs argument
    // note the file has no newline to allow use like
    //
    // sit.py --inputs "$(cat model_ilist.txt)"
    const auto ilist_path = base_name + "_ilist.txt";
    dump_file_list(ilist_path, base_input_names);
    LOG_INFO("Wrote input list " << ilist_path << "...");
}

void ov::npuw::dump_output_list(const std::string& base_name, const std::vector<std::string>& base_output_names) {
    const auto olist_path = base_name + "_olist.txt";
    dump_file_list(olist_path, base_output_names);
    LOG_INFO("Wrote output list " << olist_path << "...");
}

void ov::npuw::dump_compile_options(const std::string& name,
                                    const std::string& device,
                                    const ov::AnyMap& config,
                                    const std::string& dump_dir) {
    static std::atomic<size_t> seq{0};
    const auto escape = [](const std::string& in) {
        std::string out;
        for (const char c : in) {
            if (c == '"' || c == '\\') {
                out += '\\';
                out += c;
            } else if (c == '\n') {
                out += "\\n";
            } else {
                out += c;
            }
        }
        return out;
    };
    std::string safe = name;
    for (auto& c : safe) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-' && c != '.') {
            c = '_';
        }
    }
    const auto dump_path = std::filesystem::path(dump_dir) / ("npuw_" + std::to_string(seq++) + "_" + safe + ".json");
    if (!dump_dir.empty()) {
        std::error_code error;
        std::filesystem::create_directories(dump_dir, error);
        if (error) {
            LOG_ERROR("Failed to create compile options dump directory " << dump_dir << ": " << error.message());
            return;
        }
    }
    std::ofstream out(dump_path);
    if (!out.is_open()) {
        LOG_ERROR("Failed to open compile options dump file for writing: " << dump_path.string());
        return;
    }
    out << "{\n";
    bool first = true;
    for (const auto& [key, value] : config) {
        std::ostringstream value_stream;
        value.print(value_stream);
        const auto text = value_stream.str().empty() ? "<unprintable>" : value_stream.str();
        out << (first ? "" : ",\n") << "    \"" << escape(key) << "\": \"" << escape(text) << "\"";
        first = false;
    }
    out << "\n}\n";
    if (!out) {
        LOG_ERROR("Failed to write compile options dump file: " << dump_path.string());
        return;
    }
    LOG_INFO("Dumped compile options of " << name << " (" << device << ") to " << dump_path.string());
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
