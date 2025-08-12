// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/config/npuw.hpp"

using namespace intel_npu;
using namespace ov::intel_npu;

//
// register
//

void intel_npu::registerNPUWOptions(OptionsDesc& desc) {
    desc.add<NPU_USE_NPUW>();
    desc.add<NPUW_DEVICES>();
    desc.add<NPUW_SUBMODEL_DEVICE>();
    desc.add<NPUW_ONLINE_PIPELINE>();
    desc.add<NPUW_ONLINE_AVOID>();
    desc.add<NPUW_ONLINE_ISOLATE>();
    desc.add<NPUW_ONLINE_NO_FOLD>();
    desc.add<NPUW_ONLINE_MIN_SIZE>();
    desc.add<NPUW_ONLINE_KEEP_BLOCKS>();
    desc.add<NPUW_ONLINE_KEEP_BLOCK_SIZE>();
    desc.add<NPUW_ONLINE_DUMP_PLAN>();
    desc.add<NPUW_PLAN>();
    desc.add<NPUW_FOLD>();
    desc.add<NPUW_CWAI>();
    desc.add<NPUW_DQ>();
    desc.add<NPUW_DQ_FULL>();
    desc.add<NPUW_PMM>();
    desc.add<NPUW_SLICE_OUT>();
    desc.add<NPUW_SPATIAL>();
    desc.add<NPUW_SPATIAL_NWAY>();
    desc.add<NPUW_SPATIAL_DYN>();
    desc.add<NPUW_HOST_GATHER>();
    desc.add<NPUW_CACHE_ROPE>();
    desc.add<NPUW_F16IC>();
    desc.add<NPUW_DCOFF_TYPE>();
    desc.add<NPUW_DCOFF_SCALE>();
    desc.add<NPUW_FUNCALL_FOR_ALL>();
    desc.add<NPUW_PARALLEL_COMPILE>();
    desc.add<NPUW_FUNCALL_ASYNC>();
    desc.add<NPUW_UNFOLD_IREQS>();
    desc.add<NPUW_WEIGHTS_BANK>();
    desc.add<NPUW_WEIGHTS_BANK_ALLOC>();
    desc.add<NPUW_CACHE_DIR>();
    desc.add<NPUW_ACC_CHECK>();
    desc.add<NPUW_ACC_THRESH>();
    desc.add<NPUW_ACC_DEVICE>();
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    desc.add<NPUW_DUMP_FULL>();
    desc.add<NPUW_DUMP_SUBS>();
    desc.add<NPUW_DUMP_SUBS_ON_FAIL>();
    desc.add<NPUW_DUMP_IO>();
    desc.add<NPUW_DUMP_IO_ITERS>();
#endif
}

void intel_npu::registerNPUWLLMOptions(OptionsDesc& desc) {
    desc.add<NPUW_LLM>();
    desc.add<NPUW_LLM_BATCH_DIM>();
    desc.add<NPUW_LLM_SEQ_LEN_DIM>();
    desc.add<NPUW_LLM_MAX_PROMPT_LEN>();
    desc.add<NPUW_LLM_MIN_RESPONSE_LEN>();
    desc.add<NPUW_LLM_MAX_LORA_RANK>();
    desc.add<NPUW_LLM_OPTIMIZE_V_TENSORS>();
    desc.add<NPUW_LLM_PREFILL_CHUNK_SIZE>();
    desc.add<NPUW_LLM_PREFILL_HINT>();
    desc.add<NPUW_LLM_GENERATE_HINT>();
    desc.add<NPUW_LLM_SHARED_HEAD>();
}

std::string ov::npuw::s11n::anyToString(const ov::Any& var) {
#define HNDL(anyt, t)                                                \
    auto type = static_cast<int>(AnyType::anyt);                     \
    stream.write(reinterpret_cast<const char*>(&type), sizeof type); \
    auto var_as = var.as<t>();                                       \
    stream.write(reinterpret_cast<const char*>(&var_as), sizeof var_as);

    std::stringstream stream;
    // FIXME: figure out a proper way to serialize Any (for config)
    if (var.is<std::string>()) {
        auto type = static_cast<int>(AnyType::STRING);
        stream.write(reinterpret_cast<const char*>(&type), sizeof type);
        auto str = var.as<std::string>();
        auto var_size = str.size();
        stream.write(reinterpret_cast<const char*>(&var_size), sizeof var_size);
        stream.write(&str[0], str.size());
    } else if (var.is<const char*>()) {
        // FIXME: handle properly
        auto type = static_cast<int>(AnyType::CHARS);
        stream.write(reinterpret_cast<const char*>(&type), sizeof type);
        auto str = std::string(var.as<const char*>());
        auto var_size = str.size();
        stream.write(reinterpret_cast<const char*>(&var_size), sizeof var_size);
        stream.write(&str[0], str.size());
    } else if (var.is<std::size_t>()) {
        HNDL(SIZET, std::size_t)
    } else if (var.is<int>()) {
        HNDL(INT, int)
    } else if (var.is<int64_t>()) {
        HNDL(INT64, int64_t)
    } else if (var.is<uint32_t>()) {
        HNDL(UINT32, uint32_t)
    } else if (var.is<uint64_t>()) {
        HNDL(UINT64, uint64_t)
    } else if (var.is<float>()) {
        HNDL(FLOAT, float)
    } else if (var.is<bool>()) {
        HNDL(BOOL, bool)
    } else if (var.is<ov::CacheMode>()) {
        HNDL(CACHE_MODE, ov::CacheMode)
    } else if (var.is<ov::element::Type>()) {
        HNDL(ELEMENT_TYPE, ov::element::Type)
    } else if (var.is<ov::AnyMap>()) {
        auto any_map = var.as<ov::AnyMap>();
        auto str = ov::npuw::s11n::anyMapToString(any_map);
        auto type = static_cast<int>(AnyType::ANYMAP);
        stream.write(reinterpret_cast<const char*>(&type), sizeof type);
        auto var_size = str.size();
        stream.write(reinterpret_cast<const char*>(&var_size), sizeof var_size);
        stream.write(&str[0], str.size());
    } else if (var.is<ov::hint::PerformanceMode>()) {
        HNDL(PERFMODE, ov::hint::PerformanceMode)
    } else {
        OPENVINO_THROW("Unsupported type of ov::Any to convert to string!");
    }
#undef HNDL
    return stream.str();
}

ov::Any ov::npuw::s11n::stringToAny(const std::string& var) {
    std::stringstream stream(var);
    int type_int;
    stream.read(reinterpret_cast<char*>(&type_int), sizeof type_int);
    AnyType type = static_cast<AnyType>(type_int);

#define HNDL(t)                                             \
    t val;                                                  \
    stream.read(reinterpret_cast<char*>(&val), sizeof val); \
    return ov::Any(val);

    if (type == AnyType::STRING) {
        std::string val;
        std::size_t size = 0;
        stream.read(reinterpret_cast<char*>(&size), sizeof size);
        val.resize(size);
        stream.read(&val[0], size);
        return ov::Any(val);
    } else if (type == AnyType::CHARS) {
        // FIXME: handle properly
        std::string val;
        std::size_t size = 0;
        stream.read(reinterpret_cast<char*>(&size), sizeof size);
        val.resize(size);
        stream.read(&val[0], size);
        return ov::Any(val);
    } else if (type == AnyType::SIZET) {
        HNDL(std::size_t)
    } else if (type == AnyType::INT) {
        HNDL(int)
    } else if (type == AnyType::INT64) {
        HNDL(int64_t)
    } else if (type == AnyType::UINT32) {
        HNDL(uint32_t)
    } else if (type == AnyType::UINT64) {
        HNDL(uint64_t)
    } else if (type == AnyType::FLOAT) {
        HNDL(float)
    } else if (type == AnyType::BOOL) {
        HNDL(bool)
    } else if (type == AnyType::CACHE_MODE) {
        HNDL(ov::CacheMode)
    } else if (type == AnyType::ELEMENT_TYPE) {
        HNDL(ov::element::Type)
    } else if (type == AnyType::ANYMAP) {
        std::string val;
        std::size_t size = 0;
        stream.read(reinterpret_cast<char*>(&size), sizeof size);
        val.resize(size);
        stream.read(&val[0], size);
        return ov::Any(ov::npuw::s11n::stringToAnyMap(val));
    } else if (type == AnyType::PERFMODE) {
        HNDL(ov::hint::PerformanceMode)
    } else {
        OPENVINO_THROW("Unsupported type of ov::Any to convert from string!");
    }
#undef HNDL
    return {};
}

std::string ov::npuw::s11n::anyMapToString(const ov::AnyMap& var) {
    std::stringstream stream;
    auto map_size = var.size();
    stream.write(reinterpret_cast<const char*>(&map_size), sizeof map_size);
    for (const auto& el : var) {
        // key
        auto str = el.first;
        auto var_size = str.size();
        stream.write(reinterpret_cast<const char*>(&var_size), sizeof var_size);
        stream.write(&str[0], str.size());
        // value
        auto any = el.second;
        auto strv = ov::npuw::s11n::anyToString(any);
        auto varv_size = strv.size();
        stream.write(reinterpret_cast<const char*>(&varv_size), sizeof varv_size);
        stream.write(&strv[0], strv.size());
    }
    return stream.str();
}

ov::AnyMap ov::npuw::s11n::stringToAnyMap(const std::string& var) {
    std::stringstream stream(var);
    ov::AnyMap res;
    std::size_t size = 0;
    stream.read(reinterpret_cast<char*>(&size), sizeof size);
    for (std::size_t i = 0; i < size; ++i) {
        // key
        std::size_t ksize = 0;
        stream.read(reinterpret_cast<char*>(&ksize), sizeof ksize);
        std::string kval;
        kval.resize(ksize);
        stream.read(&kval[0], ksize);
        // value
        std::size_t vsize = 0;
        stream.read(reinterpret_cast<char*>(&vsize), sizeof vsize);
        std::string vval;
        vval.resize(vsize);
        stream.read(&vval[0], vsize);

        auto any_val = ov::npuw::s11n::stringToAny(vval);
        res[kval] = any_val;
    }
    return res;
}
