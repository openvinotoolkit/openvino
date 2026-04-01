// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "debug_helper.hpp"
#include <regex>
#ifdef __linux__
#include <unistd.h>  // write(), STDERR_FILENO
#endif
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/util/file_util.hpp"

#ifdef GPU_DEBUG_CONFIG

#include "to_string_utils.h"
#include "loop_inst.h"
#include "condition_inst.h"
#include "program_dump_graph.h"

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/convolution.hpp"
#include "intel_gpu/primitives/pooling.hpp"
#include "intel_gpu/primitives/reduce.hpp"
#include "intel_gpu/primitives/softmax.hpp"
#include "intel_gpu/primitives/mvn.hpp"
#include "intel_gpu/primitives/swiglu.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "intel_gpu/primitives/rope.hpp"
#include "intel_gpu/primitives/crop.hpp"
#include "intel_gpu/primitives/strided_slice.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/select.hpp"
#include "intel_gpu/primitives/scatter_update.hpp"
#include "intel_gpu/primitives/concatenation.hpp"
#include "intel_gpu/primitives/rms.hpp"
#include "intel_gpu/primitives/tile.hpp"
#include "intel_gpu/primitives/normalize.hpp"
#include "intel_gpu/primitives/gather_elements.hpp"
#include "intel_gpu/primitives/scatter_nd_update.hpp"
#include "intel_gpu/primitives/scatter_elements_update.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "intel_gpu/primitives/resample.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/primitives/adaptive_pooling.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"
#include "intel_gpu/primitives/col2im.hpp"
#include "intel_gpu/primitives/detection_output.hpp"

#include <iomanip>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace cldnn {

namespace {

float convert_element(int64_t i) { return static_cast<float>(i); }
float convert_element(int32_t i) { return static_cast<float>(i); }

float convert_element(float f) { return f; }

float convert_element(ov::float16 h) { return static_cast<float>(h); }

size_t get_x_pitch(const layout& layout) {
    try {
        auto tensor_x0 = tensor(batch(0), feature(0), spatial(0, 0, 0, 0));
        auto tensor_x1 = tensor(batch(0), feature(0), spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        return (x1 - x0);
    } catch (...) {
        // When spatial size of x=0, x_pitch is meaningless
        return 0;
    }
}

template <class T>
void __validate_data_range(memory::ptr mem, stream& stream, const layout& data_layout, std::string &info) {
    if (!mem)
        return;

    // Reinterpret buffer to represent actual data layout (same as log_memory_to_file)
    auto actual_mem = mem->get_engine()->reinterpret_buffer(*mem, data_layout);

    auto&& size = actual_mem->get_layout().get_tensor();
    mem_lock<T, mem_lock_type::read> lock(actual_mem, stream);
    auto mem_ptr = lock.data();
    auto x_pitch = get_x_pitch(actual_mem->get_layout());
    std::stringstream buffer;
    float val_min = std::numeric_limits<float>::max();
    float val_max = std::numeric_limits<float>::lowest();
    const bool is_memory_packed = !actual_mem->is_memory_reset_needed(actual_mem->get_layout());

    if (is_memory_packed) {
        for (size_t i = 0; i < actual_mem->count(); ++i) {
            auto val = convert_element(mem_ptr[i]);
            if (std::isinf(val) || std::isnan(val)) {
                std::string err_str = std::isinf(val) ? "inf" : "nan";
                GPU_DEBUG_COUT << err_str << " WAS FOUND: " << info << "  *********************" << std::endl;
                return;
            }
            if (val > val_max)
                val_max = val;
            if (val < val_min)
                val_min = val;
        }
    } else {
        for (ov::Dimension::value_type g = 0; g < size.group[0]; ++g) {
            for (ov::Dimension::value_type b = 0; b < size.batch[0]; ++b) {
                for (ov::Dimension::value_type f = 0; f < size.feature[0]; ++f) {
                    for (ov::Dimension::value_type w = 0; w < size.spatial[3]; ++w) {
                        for (ov::Dimension::value_type z = 0; z < size.spatial[2]; ++z) {
                            for (ov::Dimension::value_type y = 0; y < size.spatial[1]; ++y) {
                                cldnn::tensor t(cldnn::group(g), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(0, y, z, w));
                                size_t input_it = actual_mem->get_layout().get_linear_offset(t);

                                for (ov::Dimension::value_type x = 0; x < size.spatial[0]; ++x, input_it += x_pitch) {
                                    auto val = convert_element(mem_ptr[input_it]);
                                    if (std::isinf(val) || std::isnan(val)) {
                                        std::string err_str = std::isinf(val) ? "inf" : "nan";
                                        GPU_DEBUG_COUT << err_str << " WAS FOUND: " << info << "  *********************" << std::endl;
                                        return;
                                    }
                                    if (val > val_max)
                                        val_max = val;
                                    if (val < val_min)
                                        val_min = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    GPU_DEBUG_INFO << "min, max = " << val_min << ", " << val_max << "  : " << info << "  is_packed " << is_memory_packed << std::endl;
}

void validate_data_range(memory::ptr mem, stream& stream, const layout& data_layout, std::string &info) {
    auto data_type = data_layout.data_type;
    if (data_type == cldnn::data_types::f32)
        __validate_data_range<float>(mem, stream, data_layout, info);
    else if (data_type == cldnn::data_types::f16)
        __validate_data_range<ov::float16>(mem, stream, data_layout, info);
    else if (data_type == cldnn::data_types::i8)
        __validate_data_range<int8_t>(mem, stream, data_layout, info);
    else if (data_type == cldnn::data_types::u8)
        __validate_data_range<uint8_t>(mem, stream, data_layout, info);
    else
        GPU_DEBUG_INFO << "Unsupport data type for validating data range " << data_type << std::endl;
}

template <class T>
void dump(memory::ptr mem, stream& stream, std::ofstream& file_stream, bool dump_raw) {
    auto&& size = mem->get_layout().get_tensor();

    auto batch_size = std::max<ov::Dimension::value_type>(std::min<ov::Dimension::value_type>(ExecutionConfig::get_dump_batch_limit(), size.batch[0]), 1);
    tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count()
                    << ", addr: " << mem->buffer_ptr()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    } else {
        file_stream << "shape: " << tmp_size.to_string() << " ";
        file_stream << "(count: " << tmp_size.count()
                    << ", addr: " << mem->buffer_ptr()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format)
                    << ", original shape: " << size.to_string() << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    }

    if (size.count() == 0) {
        file_stream << "Empty buffer" << std::endl;
        return;
    }

    mem_lock<T, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    auto x_pitch = get_x_pitch(mem->get_layout());
    std::stringstream buffer;

    if (!dump_raw) {
        for (ov::Dimension::value_type g = 0; g < size.group[0]; ++g) {
            for (ov::Dimension::value_type b = 0; b < batch_size; ++b) {
                for (ov::Dimension::value_type f = 0; f < size.feature[0]; ++f) {
                    for (ov::Dimension::value_type w = 0; w < size.spatial[3]; ++w) {
                        for (ov::Dimension::value_type z = 0; z < size.spatial[2]; ++z) {
                            for (ov::Dimension::value_type y = 0; y < size.spatial[1]; ++y) {
                                cldnn::tensor t(cldnn::group(g), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(0, y, z, w));
                                size_t input_it = mem->get_layout().get_linear_offset(t);

                                for (ov::Dimension::value_type x = 0; x < size.spatial[0]; ++x, input_it += x_pitch) {
                                    buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[input_it]) << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < lock.size(); ++i) {
            buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[i]) << std::endl;
        }
    }
    file_stream << buffer.str();
}

void unpack(cldnn::data_types type, uint8_t input, int8_t &v0, int8_t &v1) {
    if (type == cldnn::data_types::i4) {
        char s_bit = (input & 0x08);
        char mask = s_bit > 0 ? 0xF0 : 0x00;
        v0 = (input & 0x0F) | mask;

        input >>= 4;
        s_bit = (input & 0x08);
        mask = s_bit > 0 ? 0xF0 : 0x00;
        v1 = (input & 0x0F) | mask;
    } else if (type == cldnn::data_types::u4) {
        v0 = input & 0x0F;
        v1 = input >> 4;
    } else {
        OPENVINO_ASSERT(false, "not supported unpacking");
    }
}

void dump_i4u4(cldnn::data_types type, memory::ptr mem, stream& stream, std::ofstream& file_stream, bool dump_raw) {
    auto&& size = mem->get_layout().get_tensor();

    auto batch_size = std::max<ov::Dimension::value_type>(std::min<ov::Dimension::value_type>(ExecutionConfig::get_dump_batch_limit(), size.batch[0]), 1);
    tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    } else {
        file_stream << "shape: " << tmp_size.to_string() << " ";
        file_stream << "(count: " << tmp_size.count()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format)
                    << ", original shape: " << size.to_string() << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    }

    if (size.count() == 0) {
        file_stream << "Empty buffer" << std::endl;
        return;
    }

    mem_lock<uint8_t, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    std::stringstream buffer;

    if (dump_raw) {
        for (size_t i = 0; i < lock.size(); ++i) {
            int8_t v0, v1;
            unpack(type, mem_ptr[i], v0, v1);
            buffer << std::fixed << std::setprecision(6) << static_cast<int>(v0) << std::endl;
            buffer << std::fixed << std::setprecision(6) << static_cast<int>(v1) << std::endl;
        }
    } else {
        GPU_DEBUG_COUT << " supports raw dump only" << std::endl;
    }
    file_stream << buffer.str();
}

std::string get_name_for_dump(const std::string& file_name) {
    std::string filename = file_name;
    std::replace(filename.begin(), filename.end(), '\\', '_');
    std::replace(filename.begin(), filename.end(), '/', '_');
    std::replace(filename.begin(), filename.end(), ' ', '_');
    std::replace(filename.begin(), filename.end(), ':', '_');
    return filename;
}

void log_memory_to_file(memory::ptr mem, layout data_layout, stream& stream, std::string filename, bool dump_raw) {
    std::ofstream file_stream(filename);
    if (!mem) {
        file_stream << "Empty" << std::endl;
        return;
    }

    // Reinterpret buffer to represent actual data layout
    auto actual_mem = mem->get_engine()->reinterpret_buffer(*mem, data_layout);

    auto mem_dt = actual_mem->get_layout().data_type;
    if (mem_dt == cldnn::data_types::f32)
        dump<float>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::f16)
        dump<ov::float16>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i64)
        dump<int64_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i32)
        dump<int32_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i8)
        dump<int8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::u8)
        dump<uint8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::boolean)
        dump<uint8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i4 || mem_dt == cldnn::data_types::u4)
        dump_i4u4(mem_dt, actual_mem, stream, file_stream, dump_raw);
    else
        GPU_DEBUG_COUT << "Dump for this data type is not supported: " << dt_to_str(mem_dt) << std::endl;
}

std::string get_file_path_for_binary_dump(cldnn::layout layout, const std::string& name, const std::string& dump_layers_path) {
    std::string filename;
    std::string data_type = ov::element::Type(layout.data_type).get_type_name();
    std::string format = layout.format.to_string();
    std::string tensor;
    auto dims = layout.get_dims();
    for (size_t r = 0 ; r < layout.get_rank() ; r++) {
        tensor += ("_" + to_string(dims[r]));
    }

    std::string layer_name = get_name_for_dump(name);
    filename = dump_layers_path + layer_name + "__" + data_type + "_" + tensor + "__" + format + ".bin";
    return filename;
}

bool is_target_iteration(int64_t iteration, const std::set<int64_t> dump_iteration) {
    if (iteration < 0)
        return true;

    if (dump_iteration.empty())
        return true;

    if (dump_iteration.find(iteration) == std::end(dump_iteration))
        return false;

    return true;
}

std::string get_matched_from_filelist(const std::vector<std::string>& file_names, std::string pattern) {
    for (const auto& file : file_names) {
        auto found = file.find(pattern);
        if (found != std::string::npos) {
            return file;
        }
    }

    return std::string();
}

bool is_layer_name_matched(const std::string& layer_name, const std::string& pattern) {
    auto upper_layer_name = std::string(layer_name.length(), '\0');
    std::transform(layer_name.begin(), layer_name.end(), upper_layer_name.begin(), ::toupper);
    auto upper_pattern = std::string(pattern.length(), '\0');
    std::transform(pattern.begin(), pattern.end(), upper_pattern.begin(), ::toupper);

    // Check pattern from exec_graph
    size_t pos = upper_layer_name.find(':');
    auto upper_exec_graph_name = upper_layer_name.substr(pos + 1, upper_layer_name.size());
    if (upper_exec_graph_name.compare(upper_pattern) == 0) {
        return true;
    }

    // Check pattern with regular expression
    std::regex re(upper_pattern);
    return std::regex_match(upper_layer_name, re);
}

bool is_layer_for_dumping(const ExecutionConfig& config, const std::string& layer_name) {
    const auto& dump_layers = config.get_dump_layer_names();
    if (dump_layers.empty())
        return true;

    auto iter = std::find_if(dump_layers.begin(), dump_layers.end(), [&](const std::string& dl){
        return is_layer_name_matched(layer_name, dl);
    });
    return (iter != dump_layers.end());
}

std::vector<std::string> get_filenames_for_matched_layer_loading_binaries(const ExecutionConfig& config, const std::string& id) {
    std::vector<std::string> file_names;
    if (config.get_load_dump_raw_binary().empty())
        return file_names;

    for (const auto& load_layer : config.get_load_dump_raw_binary()) {
        size_t file = load_layer.rfind(":");
        if (file != std::string::npos) {
            if (id == load_layer.substr(0, file)) {
                auto file_name_str = load_layer.substr(file + 1);
                size_t head = 0;
                size_t found = 0;
                do {
                    found = file_name_str.find(",", head);
                    if (found != std::string::npos)
                        file_names.push_back(file_name_str.substr(head, (found - head)));
                    else
                        file_names.push_back(file_name_str.substr(head));

                    head = found+1;
                    GPU_DEBUG_LOG << " Layer name loading raw dump : " << load_layer.substr(0, file) << " / the dump file : "
                                << file_names.back() << std::endl;
                } while (found != std::string::npos);

                return file_names;
            }
        }
    }

    return file_names;
}

}  // namespace

NodeDebugHelper::NodeDebugHelper(const primitive_inst& inst)
    : m_inst(inst)
    , m_stream(inst.get_network().get_stream())
    , m_network(inst.get_network())
    , m_program(inst.get_network().get_program().get())
    , m_iter(m_network.iteration) {
    const auto& config = m_network.get_config();
    // Load binary dump for input layers
    if (!config.get_load_dump_raw_binary().empty()) {
        const std::string layer_name = m_inst.id();
        auto files = get_filenames_for_matched_layer_loading_binaries(config, layer_name);
        if (!files.empty()) {
            m_stream.finish(); // Wait for stream completion before buffer assignment
            if (m_inst.is_input()) {
                // Loading binary dumps for output tensors of input-layers : only one output exists or index(dstN) exists
                auto dump_file = get_matched_from_filelist(files, "_dst0__");
                OPENVINO_ASSERT((files.size() == 1 || dump_file.length() != 0), "Unexpected binary dump for input layer");

                OPENVINO_ASSERT(files.size() == m_inst.outputs_memory_count(), "Mismatch dump file count");

                for (size_t i = 0; i < m_inst.outputs_memory_count(); i++) {
                    auto dump_file = files[0];
                    if (files.size() > 1 || m_inst.outputs_memory_count() != 1) {
                        std::string pattern = "_dst" + std::to_string(i) + "__";
                        dump_file = get_matched_from_filelist(files, pattern);
                    }
                    OPENVINO_ASSERT((dump_file.length() > 0), "Could not find expected pattern '_dst[N]__' for binary dump");
                    GPU_DEBUG_COUT << " Load binary dump : " << dump_file << " for " << layer_name << std::endl;

                    std::vector<uint8_t> bin = ov::util::load_binary(dump_file);
                    OPENVINO_ASSERT(!bin.empty(), "Failure loading binary from OV_LOAD_DUMP_RAW_BINARY : " + dump_file);

                    auto output_mem = m_inst.output_memory_ptr(i);
                    OPENVINO_ASSERT(output_mem->size() == bin.size(), "memory size mis-match for OV_LOAD_DUMP_RAW_BINARY : " + layer_name
                                    + "\n Expected size : " + to_string(output_mem->size()) + ", Binary : " + to_string(bin.size()));

                    output_mem->copy_from(m_stream, static_cast<void *>(&bin[0]), true);
                }
            } else {
                auto check_dst = get_matched_from_filelist(files, "_dst0__");
                OPENVINO_ASSERT(check_dst.length() == 0, "Expected to load binaries for inputs of " + layer_name);

                // Loading input tensors for any layer
                auto dump_file = get_matched_from_filelist(files, "_src0__");
                OPENVINO_ASSERT(dump_file.length() != 0, "Could not find expected pattern '_src[N]__' for binary dump input : " + layer_name);

                for (size_t i = 0; i < m_inst.dependencies().size(); i++) {
                    auto dump_file = files[0];
                    if (files.size() > 1 || m_inst.dependencies().size() != 1) {
                        std::string pattern = "_src" + std::to_string(i) + "__";
                        dump_file = get_matched_from_filelist(files, pattern);
                    }
                    if (dump_file.length() == 0) {
                        GPU_DEBUG_COUT  << " Skip loading for  input(" << i << ") of " << layer_name << std::endl;
                        continue;
                    }
                    OPENVINO_ASSERT((dump_file.length() > 0), "Could not find expected pattern '_src[N]__' for binary dump input");
                    GPU_DEBUG_COUT  << " Load binary dump : " << dump_file << " for input(" << i << ") of " << layer_name << std::endl;

                    std::vector<uint8_t> bin = ov::util::load_binary(dump_file);
                    OPENVINO_ASSERT(!bin.empty(), "Failure loading binary from OV_LOAD_DUMP_RAW_BINARY : " + dump_file);

                    auto input_mem = m_inst.dep_memory_ptr(i);
                    if (input_mem->size() != bin.size()) {
                        std::cout << "WARNING: memory size mis-match for OV_LOAD_DUMP_RAW_BINARY : " + layer_name
                                    << "  " << input_mem->size() << " / " << bin.size() << std::endl;
                        bin.resize(input_mem->size());
                    }

                    input_mem->copy_from(m_stream, static_cast<void *>(&bin[0]), true);
                }
            }
        }
    }

    // Dump input buffers of 'inst'
    if (config.get_dump_tensors_path().length() > 0) {
        const std::string& layer_name = inst.id();

        if (is_target_iteration(m_iter, config.get_dump_iterations()) &&
            config.get_dump_tensors() != ov::intel_gpu::DumpTensors::out && is_layer_for_dumping(config, layer_name)) {
            m_stream.finish(); // Wait for stream completion before dumping input buffers
            std::string debug_str_for_bin_load = " Command for loading : OV_LOAD_DUMP_RAW_BINARY=\"" + layer_name + ":";
            for (size_t i = 0; i < m_inst.dependencies().size(); i++) {
                std::string name = get_file_prefix() + "_src" + std::to_string(i);
                auto input_mem = m_inst.dep_memory_ptr(i);
                if (input_mem == nullptr) {
                    GPU_DEBUG_COUT  << " input_mem_" << i << " is nullptr. Nothing to dump." << std::endl;
                    continue;
                }

                auto dep = m_inst.dependencies().at(i);
                auto input_layout = dep.first->get_output_layout(dep.second);
                if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary) {
                    // Binary dump : raw
                    auto filename = get_file_path_for_binary_dump(input_layout, name, config.get_dump_tensors_path());

                    mem_lock<char, mem_lock_type::read> lock(input_mem, m_stream);
                    ov::util::save_binary(ov::util::make_path(filename), lock.data(), input_mem->size());
                    GPU_DEBUG_COUT << " Dump layer src : " << layer_name << " to " << filename << std::endl;
                    debug_str_for_bin_load += (filename + ",");
                } else {
                    const bool dump_raw = config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::text_raw;
                    GPU_DEBUG_COUT << " Dump " << (dump_raw ? "raw " : "") << name << std::endl;
                    auto filename = config.get_dump_tensors_path() + get_name_for_dump(name) + ".txt";
                    log_memory_to_file(input_mem,
                                       input_layout,
                                       m_stream,
                                       filename,
                                       dump_raw);
                }
            }

            if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary && !inst.is_input()) {
                debug_str_for_bin_load[debug_str_for_bin_load.size()-1] = '\"';
                GPU_DEBUG_COUT << debug_str_for_bin_load << std::endl;
            }
        }
    }
}


NodeDebugHelper::~NodeDebugHelper() {
    const auto& config = m_network.get_config();

    if (config.get_validate_output_buffer() && !m_network.is_internal()) {
        m_stream.finish(); // Wait for stream completion before checking output buffers
        for (size_t i = 0; i < m_inst.outputs_memory_count(); i++) {
            auto output_mem = m_inst.output_memory_ptr(i);
            std::string info = m_inst.id() + "(" + std::to_string(i) + ") at iteration " + std::to_string(m_network.get_current_iteration_num());
            validate_data_range(output_mem, m_stream, m_inst.get_output_layout(i), info);
        }
    }

    // Dump output buffers of 'inst'
    if (config.get_dump_tensors_path().length() > 0) {
        const std::string layer_name = m_inst.id();

        if (is_target_iteration(m_iter, config.get_dump_iterations()) &&
            config.get_dump_tensors() != ov::intel_gpu::DumpTensors::in &&
            is_layer_for_dumping(config, layer_name)) {
            m_stream.finish(); // Wait for stream completion before dumping output buffers
            std::string debug_str_for_bin_load = " Command for loading : OV_LOAD_DUMP_RAW_BINARY=\""
                                                    + layer_name + ":";
            for (size_t i = 0; i < m_inst.outputs_memory_count(); i++) {
                std::string name = get_file_prefix() + "_dst" + std::to_string(i);
                auto output_mem = m_inst.output_memory_ptr(i);
                if (output_mem == nullptr) {
                    GPU_DEBUG_COUT  << " output_mem is nullptr. Nothing to dump." << std::endl;
                    continue;
                }

                if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary) {
                    // Binary dump : raw
                    auto output_layout = m_inst.get_output_layout(i);
                    auto filename = get_file_path_for_binary_dump(output_layout, name, config.get_dump_tensors_path());

                    mem_lock<char, mem_lock_type::read> lock(output_mem, m_stream);
                    ov::util::save_binary(ov::util::make_path(filename), lock.data(), output_mem->size());
                    GPU_DEBUG_COUT  << " Dump layer dst : " << layer_name << " to " << filename << std::endl;
                    debug_str_for_bin_load += (filename + ",");
                } else {
                    const bool dump_raw = config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::text_raw;
                    GPU_DEBUG_COUT << " Dump " << (dump_raw ? "raw " : "") << name << std::endl;
                    auto filename = config.get_dump_tensors_path() + get_name_for_dump(name) + ".txt";
                    // Text dump
                    log_memory_to_file(output_mem,
                                       m_inst.get_output_layout(i),
                                       m_stream,
                                       filename,
                                       dump_raw);
                }
            }
            for (size_t i = 0; i < m_inst.get_intermediates_memories().size(); i++) {
                std::string name = get_file_prefix() + "_intermediates_" + std::to_string(i);
                auto output_mem = m_inst.get_intermediates_memories()[i];
                if (output_mem == nullptr || output_mem->size() == 0) {
                    GPU_DEBUG_COUT << " intermediates_mem is nullptr. Nothing to dump." << std::endl;
                    continue;
                }

                auto& output_layout = output_mem->get_layout();
                if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary) {
                    // Binary dump : raw
                    auto filename = get_file_path_for_binary_dump(output_layout, name, config.get_dump_tensors_path());

                    mem_lock<char, mem_lock_type::read> lock(output_mem, m_stream);
                    ov::util::save_binary(ov::util::make_path(filename), lock.data(), output_mem->size());
                    GPU_DEBUG_COUT << " Dump layer dst : " << layer_name << " to " << filename << std::endl;
                    debug_str_for_bin_load += (filename + ",");
                } else {
                    const bool dump_raw = config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::text_raw;
                    GPU_DEBUG_COUT << " Dump " << (dump_raw ? "raw " : "") << name << std::endl;
                    auto filename = config.get_dump_tensors_path() + get_name_for_dump(name) + ".txt";
                    // Text dump
                    log_memory_to_file(output_mem, output_layout, m_stream, filename, dump_raw);
                }
            }

            if (config.get_dump_src_after_exec()) {
                for (size_t i = 0; i < m_inst.inputs_memory_count(); i++) {
                    std::string name = get_file_prefix() + "_updated_src_" + std::to_string(i);
                    auto output_mem = m_inst.input_memory_ptr(i);
                    if (output_mem == nullptr) {
                        GPU_DEBUG_COUT << " updated_input_mem is nullptr. Nothing to dump." << std::endl;
                        continue;
                    }

                    auto& output_layout = m_inst.get_input_layout(i);
                    if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary) {
                        // Binary dump : raw
                        auto filename = get_file_path_for_binary_dump(output_layout, name, config.get_dump_tensors_path());

                        mem_lock<char, mem_lock_type::read> lock(output_mem, m_stream);
                        ov::util::save_binary(ov::util::make_path(filename), lock.data(), output_mem->size());
                        GPU_DEBUG_COUT << " Dump layer dst : " << layer_name << " to " << filename << std::endl;
                        debug_str_for_bin_load += (filename + ",");
                    } else {
                        const bool dump_raw = config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::text_raw;
                        GPU_DEBUG_COUT << " Dump " << (dump_raw ? "raw " : "") << name << std::endl;
                        auto filename = config.get_dump_tensors_path() + get_name_for_dump(name) + ".txt";
                        // Text dump
                        log_memory_to_file(output_mem, output_layout, m_stream, filename, dump_raw);
                    }
                }
            }

            if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary && m_inst.is_input()) {
                debug_str_for_bin_load[debug_str_for_bin_load.size()-1] = '\"';
                GPU_DEBUG_COUT << debug_str_for_bin_load << std::endl;;
            }
        }
    }
}

NetworkDebugHelper::NetworkDebugHelper(const network& net)
    : m_network(net)
    , m_iter(net.iteration) {
    auto net_id = m_network.get_id();
    const auto& config = m_network.get_config();
    if (config.get_dump_memory_pool()) {
        auto& iters = config.get_dump_iterations();
        if (iters.empty() || iters.find(m_iter) != iters.end()) {
            GPU_DEBUG_COUT << "============================================================================" << std::endl;
            GPU_DEBUG_COUT << "Start network execution (net_id : " << net_id << ", iter :" << m_iter << ")" << std::endl;
            if (m_iter == 0 && net_id > 0) {
                dump_memory_pool(config.get_dump_memory_pool_path(), m_iter);
                GPU_DEBUG_COUT << "============================================================================" << std::endl;
            }
        }
    } else {
        GPU_DEBUG_TRACE << "============================================================================" << std::endl;
        GPU_DEBUG_TRACE << "Start network execution (net_id : " << net_id << ", iter :" << m_iter << ")" << std::endl;
    }

    if (config.get_list_layers()) {
        for (auto& inst : m_network._exec_order) {
            GPU_DEBUG_COUT << inst->id() << std::endl;
            if (inst->get_node().is_type<loop>()) {
                auto& loop_node = inst->get_node().as<loop>();
                for (auto& prim : loop_node.get_body_program()->get_processing_order()) {
                    GPU_DEBUG_COUT << "\t" << prim->id() << std::endl;
                }
            } else if (inst->get_node().is_type<condition>()) {
                auto& cond_node = inst->get_node().as<condition>();
                GPU_DEBUG_COUT << "* Branch_True" << std::endl;
                for (auto& prim : cond_node.get_branch_true().inner_program->get_processing_order()) {
                    GPU_DEBUG_COUT << "\t" << prim->id() << std::endl;
                }
                GPU_DEBUG_COUT << "* Branch_False" << std::endl;
                for (auto& prim : cond_node.get_branch_false().inner_program->get_processing_order()) {
                    GPU_DEBUG_COUT << "\t" << prim->id() << std::endl;
                }
            }
        }
        if (!m_network.is_internal())
            exit(0);
    }
}

NetworkDebugHelper::~NetworkDebugHelper() {
    auto prog = m_network.get_program().get();
    auto net_id = m_network.get_id();
    const auto& config = prog->get_config();
    // print '-data_shape' option for benchmark_app
    if (config.get_print_input_data_shapes() || config.get_verbose() >= 4) {
        std::stringstream data_shape_str;
        auto add_string = [&data_shape_str](std::string str) {
            data_shape_str << ((data_shape_str.rdbuf()->in_avail() == 0) ? " -data_shape " : ",") << str;
        };

        for (auto& inst : m_network._exec_order) {
            auto name = inst->id();
            auto pos = name.find(':');
            auto type = name.substr(0, pos);
            name.erase(0, pos + 1);
            if (inst->is_input() && type == "parameter") {
                add_string(name + inst->get_output_layout().get_partial_shape().to_string());
            }
        }

        GPU_DEBUG_COUT << "[program:" << std::setw(2) << ((prog != nullptr) ? prog->get_id() : 0)
                       << "|network:" << std::setw(2) << net_id << "|iter:" << std::setw(4) << m_iter <<  "] benchmark_app cmd: "
                       << data_shape_str.str() << std::endl;
    }

    if (!config.get_dump_graphs_path().empty() && is_target_iteration(m_iter, config.get_dump_iterations())) {
        auto get_fixed_str = [](int value, int length = 2) -> std::string {
            std::ostringstream ss;
            ss << std::setw(length) << std::setfill('0') << std::to_string(value);
            return ss.str();
        };
        std::string path = get_dir_path(m_network.get_config());
        if (!path.empty()) {
            std::ofstream ofs(path + "cldnn_program_exec_p" + get_fixed_str(prog->get_id()) + "_n" + get_fixed_str(net_id)
                              + "_" + get_fixed_str(m_iter, 5) + ".graph");
            dump_graph_init(ofs, *prog, [this](const primitive_id& id) -> std::shared_ptr<const primitive_inst> {
                return m_network.get_primitive(id);
            });
        }
    }

    if (config.get_dump_memory_pool()) {
        auto& iters = config.get_dump_iterations();
        if (iters.empty() || iters.find(m_iter) != iters.end()) {
            dump_memory_pool(config.get_dump_memory_pool_path(), m_iter);
            GPU_DEBUG_COUT << "============================================================================" << std::endl;
        }
    }

    m_network.iteration++;
}

void NetworkDebugHelper::dump_memory_pool(std::string dump_path, int64_t curr_iter) const {
    m_network.get_memory_pool().dump(m_network.get_id(), curr_iter, dump_path);
    auto get_constants_mem_size = [&](allocation_type type) -> size_t {
        size_t mem_size = 0;
        for (auto& prim : m_network._primitives) {
            if (prim.second->get_node().is_constant()) {
                for (size_t i = 0; i < prim.second->outputs_memory_count(); i++) {
                    if (prim.second->output_memory_ptr(i)->get_allocation_type() == type)
                        mem_size += prim.second->output_memory_ptr(i)->size();
                }
            }
        }
        return mem_size;
    };
    auto get_variables_mem_size = [&](allocation_type type) -> size_t {
        size_t mem_size = 0;
        for (auto& var : m_network.get_variables()) {
            if (var.second->get_memory() && var.second->get_memory()->get_allocation_type() == type)
                mem_size += var.second->get_actual_mem_size();
        }
        return mem_size;
    };
    auto get_mb_size = [&](int64_t size) -> std::string {
        if (size == 0) return "0 MB";
        return std::to_string(static_cast<float>(size) / (1024 * 1024)) + " MB";
    };
    int64_t usm_host_const_mem_size     = get_constants_mem_size(allocation_type::usm_host);
    int64_t usm_device_const_mem_size   = get_constants_mem_size(allocation_type::usm_device);
    int64_t usm_host_var_mem_size       = get_variables_mem_size(allocation_type::usm_host);
    int64_t usm_device_var_mem_size     = get_variables_mem_size(allocation_type::usm_device);
    int64_t host_mem_size               = m_network.get_engine().get_used_device_memory(allocation_type::usm_host);
    int64_t device_mem_size             = m_network.get_engine().get_used_device_memory(allocation_type::usm_device);
    int64_t usm_host_mem_pool_size      = m_network.get_memory_pool().get_total_mem_pool_size(allocation_type::usm_host);
    int64_t usm_host_etc_size           = host_mem_size - usm_host_mem_pool_size
                                            - usm_host_const_mem_size - usm_host_var_mem_size;
    int64_t usm_device_mem_pool_size    = m_network.get_memory_pool().get_total_mem_pool_size(allocation_type::usm_device);
    int64_t usm_device_etc_size         = device_mem_size - usm_device_mem_pool_size
                                            - usm_device_const_mem_size - usm_device_var_mem_size;
    GPU_DEBUG_COUT << "------------------------------------------------------------------------" << std::endl;
    GPU_DEBUG_COUT << "Memory statistics for (net_id:" << m_network.get_id() << ", iter:" << curr_iter << ")" << std::endl;
    GPU_DEBUG_COUT << " Total host mem size     : " << get_mb_size(host_mem_size)               << std::endl;
    GPU_DEBUG_COUT << " * Memory pool           : " << get_mb_size(usm_host_mem_pool_size)      << std::endl;
    GPU_DEBUG_COUT << " * Constant              : " << get_mb_size(usm_host_const_mem_size)     << std::endl;
    GPU_DEBUG_COUT << " * Variable              : " << get_mb_size(usm_host_var_mem_size)       << std::endl;
    GPU_DEBUG_COUT << " * ETC                   : " << get_mb_size(usm_host_etc_size)           << std::endl;
    GPU_DEBUG_COUT << " Total device mem size   : " << get_mb_size(device_mem_size)             << std::endl;
    GPU_DEBUG_COUT << " * Memory pool           : " << get_mb_size(usm_device_mem_pool_size)    << std::endl;
    GPU_DEBUG_COUT << " * Constant              : " << get_mb_size(usm_device_const_mem_size)   << std::endl;
    GPU_DEBUG_COUT << " * Variable              : " << get_mb_size(usm_device_var_mem_size)     << std::endl;
    GPU_DEBUG_COUT << " * ETC                   : " << get_mb_size(usm_device_etc_size)         << std::endl;
    GPU_DEBUG_COUT << "------------------------------------------------------------------------" << std::endl;
}

// ============================================================================
// Bench kernel verbose logging infrastructure
// Controlled by ov::intel_gpu::bench_verbose debug option:
//   =1  Output execution log without timing
//   =2  Output execution log with timing (host-side wall-clock, no -pc needed)
// Format: ov_gpu_bench,exec,type=TYPE,id=ID,impl=IMPL,kernel=KERNEL,
//         in0=DT:DxDx...,in1=DT:DxDx...,out0=DT:DxDx...,fused=OP+OP,time=US
// ============================================================================
namespace {

static int get_bench_verbose_level() {
    static int level = ExecutionConfig::get_bench_verbose();
    return level;
}

static std::string activation_func_name(cldnn::activation_func func) {
    switch (func) {
        case cldnn::activation_func::none: return "none";
        case cldnn::activation_func::relu: return "relu";
        case cldnn::activation_func::relu_negative_slope: return "relu_negative_slope";
        case cldnn::activation_func::gelu: return "gelu";
        case cldnn::activation_func::gelu_tanh: return "gelu_tanh";
        case cldnn::activation_func::swish: return "swish";
        case cldnn::activation_func::hswish: return "hswish";
        case cldnn::activation_func::abs: return "abs";
        case cldnn::activation_func::clamp: return "clamp";
        case cldnn::activation_func::elu: return "elu";
        case cldnn::activation_func::exp: return "exp";
        case cldnn::activation_func::floor: return "floor";
        case cldnn::activation_func::ceil: return "ceil";
        case cldnn::activation_func::hard_sigmoid: return "hard_sigmoid";
        case cldnn::activation_func::hsigmoid: return "hsigmoid";
        case cldnn::activation_func::hyperbolic_tan: return "tanh";
        case cldnn::activation_func::linear: return "linear";
        case cldnn::activation_func::log: return "log";
        case cldnn::activation_func::log2: return "log2";
        case cldnn::activation_func::logistic: return "sigmoid";
        case cldnn::activation_func::mish: return "mish";
        case cldnn::activation_func::negative: return "negative";
        case cldnn::activation_func::pow: return "pow";
        case cldnn::activation_func::reciprocal: return "reciprocal";
        case cldnn::activation_func::selu: return "selu";
        case cldnn::activation_func::sign: return "sign";
        case cldnn::activation_func::sin: return "sin";
        case cldnn::activation_func::cos: return "cos";
        case cldnn::activation_func::softrelu: return "softrelu";
        case cldnn::activation_func::softplus: return "softplus";
        case cldnn::activation_func::softsign: return "softsign";
        case cldnn::activation_func::sqrt: return "sqrt";
        case cldnn::activation_func::square: return "square";
        case cldnn::activation_func::tan: return "tan";
        case cldnn::activation_func::erf: return "erf";
        case cldnn::activation_func::round_half_to_even: return "round";
        default: return "act_unknown";
    }
}

static std::string eltwise_mode_name(cldnn::eltwise_mode mode) {
    switch (mode) {
        case cldnn::eltwise_mode::sum: return "sum";
        case cldnn::eltwise_mode::sub: return "sub";
        case cldnn::eltwise_mode::prod: return "prod";
        case cldnn::eltwise_mode::div: return "div";
        case cldnn::eltwise_mode::max: return "max";
        case cldnn::eltwise_mode::min: return "min";
        case cldnn::eltwise_mode::pow: return "pow";
        case cldnn::eltwise_mode::mod: return "mod";
        case cldnn::eltwise_mode::eq: return "eq";
        case cldnn::eltwise_mode::ne: return "ne";
        case cldnn::eltwise_mode::lt: return "lt";
        case cldnn::eltwise_mode::le: return "le";
        case cldnn::eltwise_mode::gt: return "gt";
        case cldnn::eltwise_mode::ge: return "ge";
        case cldnn::eltwise_mode::squared_diff: return "squared_diff";
        case cldnn::eltwise_mode::logic_and: return "logic_and";
        case cldnn::eltwise_mode::logic_or: return "logic_or";
        case cldnn::eltwise_mode::logic_xor: return "logic_xor";
        case cldnn::eltwise_mode::floor_mod: return "floor_mod";
        default: return "eltwise_unknown";
    }
}

static const char* auto_broadcast_type_name(ov::op::AutoBroadcastType type) {
    switch (type) {
        case ov::op::AutoBroadcastType::NONE: return "none";
        case ov::op::AutoBroadcastType::NUMPY: return "numpy";
        case ov::op::AutoBroadcastType::PDPD: return "pdpd";
        default: return "unknown";
    }
}

static std::string layout_to_bench_str(const layout& l) {
    std::stringstream ss;
    ss << ov::element::Type(l.data_type).get_type_name() << ":";
    auto ps = l.get_partial_shape();
    for (size_t d = 0; d < ps.size(); d++) {
        if (d > 0) ss << "x";
        if (ps[d].is_dynamic())
            ss << "?";
        else
            ss << ps[d].get_length();
    }
    // Append layout format (e.g. ":bfyx", ":b_fs_yx_fsv16")
    ss << ":" << l.format.to_string();
    return ss.str();
}

static std::string fused_ops_to_bench_str(const std::vector<cldnn::fused_primitive_desc>& fused_desc) {
    if (fused_desc.empty()) return "none";
    std::stringstream ss;
    for (size_t i = 0; i < fused_desc.size(); i++) {
        if (i > 0) ss << "+";
        const auto& fd = fused_desc[i];
        auto type_str = fd.desc->type_string();
        if (fd.is_type<activation>()) {
            auto act = fd.typed_desc<activation>();
            ss << "activation_" << activation_func_name(act->activation_function);
            // Append alpha:beta when non-zero (e.g. swish:1, relu_negative_slope:0.1, linear:0.5:0.3)
            float a = act->additional_params.a;
            float b = act->additional_params.b;
            // Some activations have implicit alpha (e.g. swish always uses beta=1 in OpenVINO)
            if (act->activation_function == cldnn::activation_func::swish && a == 0.0f)
                a = 1.0f;
            if (a != 0.0f || b != 0.0f) {
                ss << ":" << a;
                if (b != 0.0f)
                    ss << ":" << b;
            }
        } else if (fd.is_type<eltwise>()) {
            auto elt = fd.typed_desc<eltwise>();
            ss << "eltwise_" << eltwise_mode_name(elt->mode);
            // Append 2nd input dt from external dependency (e.g. eltwise_sum:f16)
            bool found_ext = false;
            for (const auto& inp : fd.inputs) {
                if (inp.m_type == FusedInputType::EXTERNAL) {
                    ss << ":" << ov::element::Type(inp.m_element_type).get_type_name();
                    found_ext = true;
                    break;
                }
            }
            // Fallback: use output layout dt if no external input found
            if (!found_ext) {
                ss << ":" << ov::element::Type(fd.output_layout.data_type).get_type_name();
            }
        } else {
            ss << type_str;
        }
    }
    return ss.str();
}

}  // anonymous namespace

PrimitiveInstDebugHelper::PrimitiveInstDebugHelper(primitive_inst& inst)
    : m_inst(inst) {
    start_bench_timing();
}

PrimitiveInstDebugHelper::~PrimitiveInstDebugHelper() {
    stop_bench_timing();
    dump_bench_kernel_verbose();
}

void PrimitiveInstDebugHelper::start_bench_timing() {
    if (get_bench_verbose_level() >= 2) {
        m_inst.get_network().get_stream().finish();  // full GPU sync before timing
        m_bench_start_time = std::chrono::high_resolution_clock::now();
        m_bench_time_us = 0.0;
    }
}

void PrimitiveInstDebugHelper::stop_bench_timing() {
    if (get_bench_verbose_level() >= 2) {
        auto ev = m_inst._impl_params->out_event;
        if (ev) {
            m_inst.get_network().get_stream().wait_for_events({ev});
        } else {
            m_inst.get_network().get_stream().finish();
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        m_bench_time_us = std::chrono::duration<double, std::micro>(end_time - m_bench_start_time).count();
    }
}

void PrimitiveInstDebugHelper::dump_bench_kernel_verbose() const {
    const int bench_level = get_bench_verbose_level();
    if (bench_level <= 0) return;

    // One-time header with device info
    static bool header_printed = false;
    if (!header_printed) {
        header_printed = true;
        const auto& dev_info = m_inst.get_network().get_engine().get_device_info();
        auto gpu_id = m_inst.get_network().get_config().get_property(ov::device::id.name()).as<std::string>();
        std::string hdr = "ov_gpu_bench,info,device=" + dev_info.dev_name
                        + ",driver=" + dev_info.driver_version
                        + ",gpu_id=" + gpu_id + "\n";
        auto ret = ::write(STDERR_FILENO, hdr.data(), hdr.size());
        (void)ret;
    }

    // Cache gpu_id for each exec line
    static std::string cached_gpu_id;
    if (cached_gpu_id.empty()) {
        cached_gpu_id = m_inst.get_network().get_config().get_property(ov::device::id.name()).as<std::string>();
    }

    const auto& params = *m_inst._impl_params;
    std::stringstream ss;
    ss << "ov_gpu_bench,exec";
    ss << ",device=gpu." << cached_gpu_id;
    ss << ",type=" << params.desc->type_string();
    ss << ",id=" << m_inst.id();

    // impl type
    if (m_inst._impl && m_inst._impl->m_manager) {
        ss << ",impl=" << m_inst._impl->m_manager->get_impl_type();
    } else {
        ss << ",impl=unknown";
    }

    // kernel name
    ss << ",kernel=" << (m_inst._impl ? m_inst._impl->get_kernel_name() : "unknown");

    // Input layouts
    for (size_t i = 0; i < params.input_layouts.size(); i++) {
        ss << ",in" << i << "=" << layout_to_bench_str(params.input_layouts[i]);
    }

    // Output layouts
    for (size_t i = 0; i < params.output_layouts.size(); i++) {
        ss << ",out" << i << "=" << layout_to_bench_str(params.output_layouts[i]);
    }

    // Fused ops
    ss << ",fused=" << fused_ops_to_bench_str(params.fused_desc);

    // Primitive-specific attributes
    auto type_str = params.desc->type_string();
    if (type_str == "reorder") {
        auto prim = std::static_pointer_cast<const reorder>(params.desc);
        if (prim->truncate) {
            ss << ",truncate=1";
        }
    } else if (type_str == "gemm") {
        auto prim = std::static_pointer_cast<const gemm>(params.desc);
        ss << ",transpose_a=" << prim->transpose_input0;
        ss << ",transpose_b=" << prim->transpose_input1;
        if (!prim->input0_transpose_order.empty()) {
            ss << ",order0=";
            for (size_t i = 0; i < prim->input0_transpose_order.size(); i++) {
                if (i > 0) ss << ":";
                ss << prim->input0_transpose_order[i];
            }
        }
        if (!prim->input1_transpose_order.empty()) {
            ss << ",order1=";
            for (size_t i = 0; i < prim->input1_transpose_order.size(); i++) {
                if (i > 0) ss << ":";
                ss << prim->input1_transpose_order[i];
            }
        }
    } else if (type_str == "scaled_dot_product_attention") {
        auto prim = std::static_pointer_cast<const scaled_dot_product_attention>(params.desc);
        ss << ",is_causal=" << prim->is_causal;
        auto dump_order = [&](const std::string& name, const std::vector<int64_t>& order) {
            ss << "," << name << "=";
            for (size_t i = 0; i < order.size(); i++) {
                if (i > 0) ss << ":";
                ss << order[i];
            }
        };
        dump_order("order_q", prim->input_q_transpose_order);
        dump_order("order_k", prim->input_k_transpose_order);
        dump_order("order_v", prim->input_v_transpose_order);
        dump_order("order_out", prim->output_transpose_order);
        if (prim->scale_val.has_value()) {
            ss << ",scale_val=" << prim->scale_val.value();
        }
    } else if (type_str == "fully_connected") {
        auto prim = std::static_pointer_cast<const fully_connected>(params.desc);
        ss << ",compressed=" << prim->compressed_weights;
        ss << ",dynamic_quantized=" << prim->dynamic_quantized_activation;
        ss << ",dynamic_quantized_zp=" << prim->dynamic_quantized_activation_zp;
        ss << ",dynamic_quantized_precomputed_reduction=" << prim->dynamic_quantized_precomputed_reduction;
        ss << ",fc_input_size=" << prim->input_size;
        ss << ",fc_weights_rank=" << prim->weights_rank;

        // Emit bench attrs so converter can reconstruct FC scales/zero-points.
        // Input convention: in0=src, in1=weights, then optional wei_scale/wei_zp/src_scale/src_zp.
        auto is_int_dt = [](cldnn::data_types dt) {
            return dt == cldnn::data_types::i8 || dt == cldnn::data_types::u8 ||
                   dt == cldnn::data_types::i4 || dt == cldnn::data_types::u4 ||
                   dt == cldnn::data_types::i32 || dt == cldnn::data_types::u32 ||
                   dt == cldnn::data_types::i64 || dt == cldnn::data_types::u64;
        };

        std::vector<std::string> scale_attrs;
        std::vector<std::string> zp_attrs;
        size_t attr_input_idx = 2;

        if (prim->compressed_weights && params.input_layouts.size() > attr_input_idx) {
            auto wei_scale_dt = dt_to_str(params.input_layouts[attr_input_idx].data_type);
            scale_attrs.push_back(std::string("wei:per_oc:") + wei_scale_dt);
            attr_input_idx++;

            if (params.input_layouts.size() > attr_input_idx &&
                is_int_dt(params.input_layouts[attr_input_idx].data_type)) {
                auto wei_zp_dt = dt_to_str(params.input_layouts[attr_input_idx].data_type);
                zp_attrs.push_back(std::string("wei:per_oc:") + wei_zp_dt);
                attr_input_idx++;
            }
        }

        if (prim->dynamic_quantized_activation && params.input_layouts.size() > attr_input_idx) {
            auto src_scale_dt = dt_to_str(params.input_layouts[attr_input_idx].data_type);
            scale_attrs.push_back(std::string("src:per_token:") + src_scale_dt);
            attr_input_idx++;

            if (prim->dynamic_quantized_activation_zp && params.input_layouts.size() > attr_input_idx) {
                auto src_zp_dt = dt_to_str(params.input_layouts[attr_input_idx].data_type);
                zp_attrs.push_back(std::string("src:per_token:") + src_zp_dt);
            }
        }

        if (!scale_attrs.empty()) {
            ss << ",attr_scales=";
            for (size_t i = 0; i < scale_attrs.size(); ++i) {
                if (i > 0) ss << "+";
                ss << scale_attrs[i];
            }
        }
        if (!zp_attrs.empty()) {
            ss << ",attr_zero_points=";
            for (size_t i = 0; i < zp_attrs.size(); ++i) {
                if (i > 0) ss << "+";
                ss << zp_attrs[i];
            }
        }
    } else if (type_str == "convolution") {
        auto prim = std::static_pointer_cast<const convolution>(params.desc);
        ss << ",groups=" << prim->groups;
        auto dump_vec = [&](const std::string& name, const auto& vec) {
            ss << "," << name << "=";
            for (size_t i = 0; i < vec.size(); i++) {
                if (i > 0) ss << "x";
                ss << vec[i];
            }
        };
        dump_vec("strides", prim->stride);
        dump_vec("dilations", prim->dilation);
        dump_vec("padding_begin", prim->padding_begin);
        dump_vec("padding_end", prim->padding_end);
        ss << ",grouped_weights_shape=" << prim->grouped_weights_shape;
    } else if (type_str == "pooling") {
        auto prim = std::static_pointer_cast<const pooling>(params.desc);
        ss << ",pool_mode=" << static_cast<int>(prim->mode);
        auto dump_vec = [&](const std::string& name, const auto& vec) {
            ss << "," << name << "=";
            for (size_t i = 0; i < vec.size(); i++) {
                if (i > 0) ss << "x";
                ss << vec[i];
            }
        };
        dump_vec("kernel", prim->size);
        dump_vec("pool_strides", prim->stride);
        dump_vec("pads_begin", prim->pads_begin);
        dump_vec("pads_end", prim->pads_end);
        ss << ",rounding_type=" << static_cast<int>(prim->rounding_type);
    } else if (type_str == "reduce") {
        auto prim = std::static_pointer_cast<const reduce>(params.desc);
        ss << ",reduce_mode=" << static_cast<int>(prim->mode);
        ss << ",keep_dims=" << prim->keep_dims;
        ss << ",reduce_axes=";
        for (size_t i = 0; i < prim->axes.size(); i++) {
            if (i > 0) ss << ":";
            ss << prim->axes[i];
        }
    } else if (type_str == "softmax") {
        auto prim = std::static_pointer_cast<const softmax>(params.desc);
        ss << ",axis=" << prim->dimension;
    } else if (type_str == "mvn") {
        auto prim = std::static_pointer_cast<const mvn>(params.desc);
        ss << ",normalize_variance=" << prim->normalize_variance;
        ss << ",epsilon=" << prim->epsilon;
        ss << ",eps_inside_sqrt=" << prim->eps_inside_sqrt;
        if (!prim->reduction_axes.empty()) {
            ss << ",mvn_reduction_axes=";
            for (size_t i = 0; i < prim->reduction_axes.size(); ++i) {
                if (i > 0) ss << ":";
                ss << prim->reduction_axes[i];
            }
        }
    } else if (type_str == "eltwise") {
        auto prim = std::static_pointer_cast<const eltwise>(params.desc);
        ss << ",eltwise_mode=" << static_cast<int>(prim->mode);
        ss << ",pythondiv=" << prim->m_pythondiv;
        if (!prim->coefficients.empty()) {
            ss << ",eltwise_coefficients=";
            for (size_t i = 0; i < prim->coefficients.size(); ++i) {
                if (i > 0) ss << ":";
                ss << prim->coefficients[i];
            }
        }
        if (!prim->stride.empty()) {
            ss << ",eltwise_stride=";
            for (size_t i = 0; i < prim->stride.size(); ++i) {
                if (i > 0) ss << ";";
                for (size_t j = 0; j < prim->stride[i].raw.size(); ++j) {
                    if (j > 0) ss << ":";
                    ss << prim->stride[i].raw[j];
                }
            }
        }
        ss << ",eltwise_broadcast_type=" << auto_broadcast_type_name(prim->broadcast_spec.m_type);
        ss << ",eltwise_broadcast_axis=" << prim->broadcast_spec.m_axis;
    } else if (type_str == "swiglu") {
        auto prim = std::static_pointer_cast<const swiglu>(params.desc);
        ss << ",glu_type=" << static_cast<int>(prim->glu_type);
        ss << ",split_axis=" << prim->axis;
        ss << ",split_length=" << prim->glu_stride;
        ss << ",gate_idx=" << prim->gate_idx;
    } else if (type_str == "gather") {
        auto prim = std::static_pointer_cast<const gather>(params.desc);
        ss << ",gather_axis=" << prim->axis;
        ss << ",batch_dim=" << prim->batch_dim;
        ss << ",support_neg_ind=" << prim->support_neg_ind;
    } else if (type_str == "rope") {
        auto prim = std::static_pointer_cast<const rope>(params.desc);
        ss << ",head_cnt=" << prim->config.head_cnt;
        ss << ",head_size=" << prim->config.head_size;
        ss << ",rotary_ndims=" << prim->config.rotary_ndims;
        ss << ",is_interleaved=" << prim->config.is_interleaved;
        ss << ",is_chatglm=" << prim->config.is_chatglm;
        ss << ",is_qwen=" << prim->config.is_qwen;
        ss << ",input_trans0213=" << prim->config.input_trans0213;
        ss << ",output_trans0213=" << prim->config.output_trans0213;
        ss << ",use_rope_cache=" << prim->config.use_rope_cache;
        ss << ",support_2d_rope=" << prim->config.support_2d_rope;
        ss << ",support_3d_rope=" << prim->config.support_3d_rope;
        ss << ",is_ltx_video=" << prim->config.is_ltx_video;
        ss << ",gather_position_arg_id=" << prim->config.gather_position_arg_id;
        ss << ",slice_start=" << prim->config.slice_start;
        ss << ",slice_stop=" << prim->config.slice_stop;
        ss << ",gather_rank=" << prim->gather_rank;
    } else if (type_str == "crop") {
        auto prim = std::static_pointer_cast<const crop>(params.desc);
        auto off = prim->offsets;
        ss << ",offsets=" << off.batch[0] << ":" << off.feature[0]
           << ":" << off.spatial[1] << ":" << off.spatial[0];
    } else if (type_str == "strided_slice") {
        auto prim = std::static_pointer_cast<const strided_slice>(params.desc);
        auto dump_ivec = [&](const std::string& name, const std::vector<int64_t>& v) {
            ss << "," << name << "=";
            for (size_t i = 0; i < v.size(); i++) {
                if (i > 0) ss << ":";
                ss << v[i];
            }
        };
        dump_ivec("ss_begin", prim->begin);
        dump_ivec("ss_end", prim->end);
        dump_ivec("ss_strides", prim->strides);
        dump_ivec("begin_mask", prim->begin_mask);
        dump_ivec("end_mask", prim->end_mask);
        dump_ivec("shrink_axis_mask", prim->shrink_axis_mask);
        dump_ivec("new_axis_mask", prim->new_axis_mask);
    } else if (type_str == "concatenation") {
        auto prim = std::static_pointer_cast<const concatenation>(params.desc);
        ss << ",concat_axis=" << prim->axis;
    } else if (type_str == "scatter_update") {
        auto prim = std::static_pointer_cast<const scatter_update>(params.desc);
        ss << ",gather_axis=" << prim->axis;
    } else if (type_str == "rms") {
        auto prim = std::static_pointer_cast<const rms>(params.desc);
        ss << ",epsilon=" << prim->epsilon;
    } else if (type_str == "tile") {
        auto prim = std::static_pointer_cast<const tile>(params.desc);
        ss << ",tile_repeats=";
        for (size_t i = 0; i < prim->repeats.size(); ++i) {
            if (i) ss << ":";
            ss << prim->repeats[i];
        }
    } else if (type_str == "normalize") {
        auto prim = std::static_pointer_cast<const normalize>(params.desc);
        ss << ",across_spatial=" << prim->across_spatial;
        ss << ",epsilon=" << prim->epsilon;
    } else if (type_str == "gather_elements") {
        auto prim = std::static_pointer_cast<const gather_elements>(params.desc);
        ss << ",gather_axis=" << prim->axis;
    } else if (type_str == "scatter_nd_update") {
        auto prim = std::static_pointer_cast<const scatter_nd_update>(params.desc);
        ss << ",indices_rank=" << prim->indices_rank;
    } else if (type_str == "scatter_elements_update") {
        auto prim = std::static_pointer_cast<const scatter_elements_update>(params.desc);
        ss << ",axis=" << prim->axis;
        ss << ",scatter_mode=" << static_cast<int>(prim->mode);
        ss << ",scatter_use_init_val=" << prim->use_init_val;
    } else if (type_str == "group_normalization") {
        auto prim = std::static_pointer_cast<const group_normalization>(params.desc);
        ss << ",num_groups=" << prim->num_groups;
        ss << ",epsilon=" << prim->epsilon;
    } else if (type_str == "quantize") {
        auto prim = std::static_pointer_cast<const quantize>(params.desc);
        ss << ",levels=" << prim->levels;
    } else if (type_str == "deconvolution") {
        auto prim = std::static_pointer_cast<const deconvolution>(params.desc);
        ss << ",groups=" << prim->groups;
        ss << ",grouped_weights_shape=" << (prim->grouped_weights_shape ? 1 : 0);
        ss << ",strides=";
        for (size_t i = 0; i < prim->stride.size(); ++i) { if (i) ss << "x"; ss << prim->stride[i]; }
        ss << ",dilations=";
        for (size_t i = 0; i < prim->dilations.size(); ++i) { if (i) ss << "x"; ss << prim->dilations[i]; }
        ss << ",padding_begin=";
        for (size_t i = 0; i < prim->pads_begin.size(); ++i) { if (i) ss << "x"; ss << prim->pads_begin[i]; }
        ss << ",padding_end=";
        for (size_t i = 0; i < prim->pads_end.size(); ++i) { if (i) ss << "x"; ss << prim->pads_end[i]; }
    } else if (type_str == "resample") {
        auto prim = std::static_pointer_cast<const resample>(params.desc);
        ss << ",resample_mode=" << static_cast<int>(prim->operation_type);
        ss << ",resample_sizes=";
        for (size_t i = 0; i < prim->sizes.size(); ++i) { if (i) ss << ":"; ss << prim->sizes[i]; }
    } else if (type_str == "permute") {
        auto prim = std::static_pointer_cast<const permute>(params.desc);
        ss << ",permute_order=";
        for (size_t i = 0; i < prim->permute_order.size(); ++i) { if (i) ss << ":"; ss << prim->permute_order[i]; }
    } else if (type_str == "adaptive_pooling") {
        auto prim = std::static_pointer_cast<const adaptive_pooling>(params.desc);
        ss << ",adaptive_pool_mode=" << static_cast<int>(prim->mode);
        ss << ",adaptive_pool_out=";
        for (size_t i = 0; i < prim->output_size.raw.size(); ++i) {
            if (i) ss << ":";
            ss << prim->output_size.raw[i];
        }
    } else if (type_str == "arg_max_min") {
        auto prim = std::static_pointer_cast<const arg_max_min>(params.desc);
        ss << ",topk_mode=" << static_cast<int>(prim->mode);
        ss << ",top_k=" << prim->top_k;
        ss << ",axis=" << prim->axis;
    } else if (type_str == "col2im") {
        auto prim = std::static_pointer_cast<const col2im>(params.desc);
        ss << ",strides=";
        for (size_t i = 0; i < prim->stride.size(); ++i) { if (i) ss << "x"; ss << prim->stride[i]; }
        ss << ",dilations=";
        for (size_t i = 0; i < prim->dilation.size(); ++i) { if (i) ss << "x"; ss << prim->dilation[i]; }
        ss << ",col2im_padding_begin=";
        for (size_t i = 0; i < prim->padding_begin.size(); ++i) { if (i) ss << "x"; ss << prim->padding_begin[i]; }
        ss << ",col2im_padding_end=";
        for (size_t i = 0; i < prim->padding_end.size(); ++i) { if (i) ss << "x"; ss << prim->padding_end[i]; }
        ss << ",col2im_output_shape=";
        for (size_t i = 0; i < prim->output_shape.size(); ++i) { if (i) ss << "x"; ss << prim->output_shape[i]; }
        ss << ",col2im_kernel_shape=";
        for (size_t i = 0; i < prim->kernel_shape.size(); ++i) { if (i) ss << "x"; ss << prim->kernel_shape[i]; }
    } else if (type_str == "detection_output") {
        auto prim = std::static_pointer_cast<const detection_output>(params.desc);
        ss << ",det_num_classes=" << prim->num_classes;
        ss << ",det_keep_top_k=" << prim->keep_top_k;
        ss << ",det_share_location=" << (prim->share_location ? 1 : 0);
        ss << ",det_background_label_id=" << prim->background_label_id;
        ss << ",det_nms_threshold=" << prim->nms_threshold;
        ss << ",det_top_k=" << prim->top_k;
        ss << ",det_eta=" << prim->eta;
        ss << ",det_code_type=" << static_cast<int>(prim->code_type);
        ss << ",det_variance_encoded=" << (prim->variance_encoded_in_target ? 1 : 0);
        ss << ",det_confidence_threshold=" << prim->confidence_threshold;
        ss << ",det_prior_info_size=" << prim->prior_info_size;
        ss << ",det_prior_coordinates_offset=" << prim->prior_coordinates_offset;
        ss << ",det_prior_is_normalized=" << (prim->prior_is_normalized ? 1 : 0);
        ss << ",det_input_width=" << prim->input_width;
        ss << ",det_input_height=" << prim->input_height;
        ss << ",det_decrease_label_id=" << (prim->decrease_label_id ? 1 : 0);
        ss << ",det_clip_before_nms=" << (prim->clip_before_nms ? 1 : 0);
        ss << ",det_clip_after_nms=" << (prim->clip_after_nms ? 1 : 0);
        ss << ",det_objectness_score=" << prim->objectness_score;
    } else {
        ss << ",unsupported=" << type_str;
    }

    // Timing: use pre-recorded host-side wall-clock time (no ov::enable_profiling needed)
    ss << ",time=" << std::fixed << std::setprecision(3) << m_bench_time_us;

    // Use single write() syscall to avoid interleaving with other threads' output
    std::string msg = ss.str() + "\n";
    auto ret = ::write(STDERR_FILENO, msg.data(), msg.size());
    (void)ret;
}

}  // namespace cldnn

#endif // GPU_DEBUG_CONFIG
