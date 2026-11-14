// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vm_runtime/virtual_machine.h"

#include <gflags/gflags.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

DEFINE_string(path, "", "[Required] Path to the bytecode file");
DEFINE_validator(path, [](const char* flagname, const std::string& value) {
    if (value.empty()) {
        std::cerr << "Error: the path to the bytecode file must be provided via the --" << flagname << " argument"
                  << std::endl;
        return false;
    }
    if (!std::filesystem::exists(value)) {
        std::cerr << "Error: the specified bytecode file does not exist: " << value << std::endl;
        return false;
    }
    return true;
});
DEFINE_string(mode, "run",
              "[Optional] Execution mode: 'run', 'print', 'print-full'. The 'print-full' mode includes the "
              "content of binary sections, such as constants or kernels");
DEFINE_validator(mode, [](const char* /*flagname*/, const std::string& value) {
    if (value != "run" && value != "print" && value != "print-full") {
        std::cerr << "Error: invalid execution mode '" << value << "'. Valid options are: run, print, print-full."
                  << std::endl;
        return false;
    }
    return true;
});
DEFINE_string(function, NPU_VM_MAIN_INFERENCE_FUNCTION_NAME,
              "[Optional] Name of the function to execute (default 'main'). Used if --mode=run");
DEFINE_validator(function, [](const char* flagname, const std::string& value) {
    if (FLAGS_mode == "run" && value.empty()) {
        std::cerr << "Error: the name of the entrypoint function must be provided via the --" << flagname
                  << " argument when --mode=run" << std::endl;
        return false;
    }
    return true;
});
DEFINE_string(params, "",
              "[Optional] Comma-separated list of typed parameters to pass to the called function, in the format "
              "'type:value,...' where type is one of: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, buffer. "
              "For buffer, value has the format `byteSize|hexData`, where `byteSize` is the size of the buffer and "
              "`hexData` is the content in hexadecimal format; if the content should have the same value repeated, a "
              "single hex byte can be provided (e.g. `00` for a zero-initialized buffer)."
              "If only a part of the hex data is given, the rest will be padded with zero"
              "Example: --params 'i32:42,f32:3.14,i64:100,buffer:4|00010203'. Used if --mode=run");

namespace {

struct Values {
    std::vector<npu_vm_value> values;
    std::vector<std::vector<uint8_t>> ownedBuffers;
};

constexpr int HEX_BASE = 16;
constexpr size_t HEX_CHARS_PER_BYTE = 2;

template <typename T>
std::optional<T> parseFromChars(std::string_view input, int base = 10) {
    if constexpr (std::is_integral_v<T>) {
        const auto* begin = input.data();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) - std::from_chars requires pointers
        const auto* end = begin + input.size();
        T parsed{};
        const auto result = std::from_chars(begin, end, parsed, base);
        if (result.ec != std::errc{} || result.ptr != end) {
            return std::nullopt;
        }
        return parsed;
    } else if constexpr (std::is_floating_point_v<T>) {
        // std::from_chars(float/double) is not available on all C++17 standard libraries
        std::string inputStr(input);
        const auto* begin = inputStr.c_str();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) - checking full token consumption
        const auto* end = begin + inputStr.size();
        char* endPtr = nullptr;
        errno = 0;
        T parsed{};
        if constexpr (std::is_same_v<T, float>) {
            parsed = std::strtof(begin, &endPtr);
        } else if constexpr (std::is_same_v<T, double>) {
            parsed = std::strtod(begin, &endPtr);
        } else {
            parsed = static_cast<T>(std::strtold(begin, &endPtr));
        }
        if (endPtr == begin || endPtr != end || errno == ERANGE) {
            return std::nullopt;
        }
        return parsed;
    } else {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>, "Unsupported type for parseFromChars");
        return std::nullopt;
    }
}

template <typename T>
bool parseTypedValue(std::string_view valueStr, std::string_view typeName, T& outValue) {
    const auto parsed = parseFromChars<T>(valueStr);
    if (!parsed.has_value()) {
        std::cerr << "Error: Invalid value '" << valueStr << "' for type '" << typeName << "'" << std::endl;
        return false;
    }
    outValue = *parsed;
    return true;
}

std::optional<uint8_t> parseHexByte(std::string_view hexByte) {
    auto parsed = parseFromChars<unsigned int>(hexByte, HEX_BASE);
    if (!parsed.has_value() || *parsed > std::numeric_limits<uint8_t>::max()) {
        return std::nullopt;
    }
    return static_cast<uint8_t>(*parsed);
}

std::optional<Values> parseParams(const std::string& paramsStr) {
    Values params;
    if (paramsStr.empty()) {
        return params;
    }
    std::istringstream ss(paramsStr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const auto colonPos = token.find(':');
        if (colonPos == std::string::npos) {
            std::cerr << "Error: Invalid parameter format '" << token << "'. Expected 'type:value'" << std::endl;
            return std::nullopt;
        }
        const auto type = token.substr(0, colonPos);
        const auto valueStr = token.substr(colonPos + 1);
        npu_vm_value value{};
        if (type == "i8") {
            if (!parseTypedValue(valueStr, "i8", value.i8)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "u8") {
            if (!parseTypedValue(valueStr, "u8", value.u8)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "i16") {
            if (!parseTypedValue(valueStr, "i16", value.i16)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "u16") {
            if (!parseTypedValue(valueStr, "u16", value.u16)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "i32") {
            if (!parseTypedValue(valueStr, "i32", value.i32)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "u32") {
            if (!parseTypedValue(valueStr, "u32", value.u32)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "i64") {
            if (!parseTypedValue(valueStr, "i64", value.i64)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "u64") {
            if (!parseTypedValue(valueStr, "u64", value.u64)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "f32") {
            if (!parseTypedValue(valueStr, "f32", value.f32)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "f64") {
            if (!parseTypedValue(valueStr, "f64", value.f64)) {
                return std::nullopt;
            }
            params.values.emplace_back(value);
        } else if (type == "buffer") {
            const auto pipePos = valueStr.find('|');
            if (pipePos == std::string::npos) {
                std::cerr << "Error: Buffer parameter must specify size and content in the format 'size|hexdata'"
                          << std::endl;
                return std::nullopt;
            }

            const auto sizeStr = valueStr.substr(0, pipePos);
            const auto contentStr = valueStr.substr(pipePos + 1);
            const auto byteSize = parseFromChars<size_t>(sizeStr);
            if (!byteSize.has_value()) {
                std::cerr << "Error: Invalid buffer size '" << sizeStr << "'" << std::endl;
                return std::nullopt;
            }
            if (*byteSize == 0) {
                std::cerr << "Error: Buffer parameter size must be greater than 0" << std::endl;
                return std::nullopt;
            }
            if (contentStr.empty()) {
                std::cerr << "Error: Buffer parameter content cannot be empty. Use '00' for zero-initialized buffer."
                          << std::endl;
                return std::nullopt;
            }
            if (contentStr.size() % HEX_CHARS_PER_BYTE != 0 || contentStr.size() / HEX_CHARS_PER_BYTE > *byteSize) {
                std::cerr << "Error: Invalid buffer content size or format" << std::endl;
                return std::nullopt;
            }
            const auto isSplat = contentStr.size() == HEX_CHARS_PER_BYTE;

            params.ownedBuffers.emplace_back(*byteSize, uint8_t{0});
            auto& buffer = params.ownedBuffers.back();
            if (isSplat) {
                const auto byte = parseHexByte(contentStr);
                if (!byte.has_value()) {
                    std::cerr << "Error: Invalid hexadecimal buffer content '" << contentStr << "'" << std::endl;
                    return std::nullopt;
                }
                std::fill(buffer.begin(), buffer.end(), *byte);
            } else {
                for (size_t i = 0; i < contentStr.size(); i += HEX_CHARS_PER_BYTE) {
                    const auto hexPair = std::string_view(contentStr).substr(i, HEX_CHARS_PER_BYTE);
                    const auto byte = parseHexByte(hexPair);
                    if (!byte.has_value()) {
                        std::cerr << "Error: Invalid hexadecimal byte in buffer content at position " << i << std::endl;
                        return std::nullopt;
                    }
                    buffer.at(i / HEX_CHARS_PER_BYTE) = *byte;
                }
            }
            value.buffer.data = buffer.data();
            if (buffer.size() > std::numeric_limits<uint32_t>::max()) {
                std::cerr << "Error: Buffer size exceeds maximum supported size of "
                          << std::numeric_limits<uint32_t>::max() << " bytes" << std::endl;
                return std::nullopt;
            }
            value.buffer.size = static_cast<uint32_t>(buffer.size());
            params.values.emplace_back(value);
        } else {
            std::cerr << "Error: Unknown parameter type '" << type
                      << "'. Valid types are: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, buffer" << std::endl;
            return std::nullopt;
        }
    }
    return params;
}

std::optional<Values> createResultStorage(npu_vm_function_info* functionInfo) {
    const auto& resultTypes = functionInfo->result_types;
    Values results;
    results.values.reserve(functionInfo->num_results);
    for (size_t i = 0; i < functionInfo->num_results; ++i) {
        npu_vm_value value{};
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) - using C API with raw pointers
        const auto resultType = resultTypes[i];
        switch (resultType) {
        case npu_vm_type_int8: {
            value.i8 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_uint8: {
            value.u8 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_int16: {
            value.i16 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_uint16: {
            value.u16 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_int32: {
            value.i32 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_uint32: {
            value.u32 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_int64: {
            value.i64 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_uint64: {
            value.u64 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_float32: {
            value.f32 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_float64: {
            value.f64 = 0;
            results.values.emplace_back(value);
            break;
        }
        case npu_vm_type_buffer: {
            value.buffer.data = nullptr;
            value.buffer.size = 0;
            results.values.emplace_back(value);
            break;
        }
        default:
            std::cerr << "Error: Unknown result type " << static_cast<int>(resultType) << " for result " << i
                      << std::endl;
            return std::nullopt;
        }
    }
    return results;
}

template <typename T>
void printSpecificResult(size_t i, const T& value, std::string_view typeStr) {
    std::cout << "Result[" << i << "] (" << typeStr << "): " << value << std::endl;
}

void printResults(npu_vm_function_info* functionInfo, const Values& results) {
    const auto& resultTypes = functionInfo->result_types;
    if (results.values.size() != functionInfo->num_results) {
        std::cerr << "Warning: Number of results returned (" << results.values.size()
                  << ") does not match the function's declared number of result types (" << functionInfo->num_results
                  << "). Attempting to print results anyway." << std::endl;
    }
    for (size_t i = 0; i < functionInfo->num_results; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) - using C API with raw pointers
        const auto resultType = resultTypes[i];
        switch (resultType) {
        case npu_vm_type_int8:
            printSpecificResult(i, static_cast<int32_t>(results.values.at(i).i8), "i8");
            break;
        case npu_vm_type_uint8:
            printSpecificResult(i, static_cast<uint32_t>(results.values.at(i).u8), "u8");
            break;
        case npu_vm_type_int16:
            printSpecificResult(i, results.values.at(i).i16, "i16");
            break;
        case npu_vm_type_uint16:
            printSpecificResult(i, results.values.at(i).u16, "u16");
            break;
        case npu_vm_type_int32:
            printSpecificResult(i, results.values.at(i).i32, "i32");
            break;
        case npu_vm_type_uint32:
            printSpecificResult(i, results.values.at(i).u32, "u32");
            break;
        case npu_vm_type_int64:
            printSpecificResult(i, results.values.at(i).i64, "i64");
            break;
        case npu_vm_type_uint64:
            printSpecificResult(i, results.values.at(i).u64, "u64");
            break;
        case npu_vm_type_float32:
            printSpecificResult(i, results.values.at(i).f32, "f32");
            break;
        case npu_vm_type_float64:
            printSpecificResult(i, results.values.at(i).f64, "f64");
            break;
        case npu_vm_type_buffer: {
            std::cout << "Result[" << i << "] (buffer): ";
            const auto bufferData = static_cast<const uint8_t*>(results.values.at(i).buffer.data);
            const auto bufferSize = results.values.at(i).buffer.size;
            if (bufferData == nullptr || bufferSize == 0) {
                std::cout << "null or empty buffer" << std::endl;
                break;
            }
            for (size_t j = 0; j < bufferSize; ++j) {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) - using C API with raw pointers
                std::cout << std::hex << static_cast<int>(bufferData[j]) << " ";
            }
            std::cout << std::dec << std::endl;
            break;
        }
        default:
            std::cout << "Result[" << i << "] (unknown): unknown" << std::endl;
            break;
        };
    }
}

struct VMContext {
    npu_vm_module* module{nullptr};
    npu_vm_engine* engine{nullptr};
    npu_vm_function_info* functionInfo{nullptr};
};

void freeResources(const VMContext& context, const std::optional<Values>& results = std::nullopt) {
    if (context.engine != nullptr) {
        npu_vm_destroy_engine(context.engine);
    }
    if (context.module != nullptr) {
        npu_vm_destroy_module(context.module);
    }
    if (context.functionInfo != nullptr) {
        if (results.has_value()) {
            const auto& resultTypes = context.functionInfo->result_types;
            if (context.functionInfo->num_results <= results->values.size()) {
                for (size_t i = 0; i < context.functionInfo->num_results; ++i) {
                    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) - using C API with raw pointers
                    const auto resultType = resultTypes[i];
                    if (resultType == npu_vm_type_buffer) {
                        const auto& value = results->values.at(i);
                        if (value.buffer.data != nullptr) {
                            // NOLINTNEXTLINE(cppcoreguidelines-owning-memory, cppcoreguidelines-no-malloc)
                            free(value.buffer.data);
                        }
                    }
                }
            }
        }
        npu_vm_destroy_function_info(context.functionInfo);
    }
}

int cleanupAndExitSuccess(const VMContext& context = {}, const std::optional<Values>& results = std::nullopt) {
    freeResources(context, results);
    return 0;
}

int cleanupAndExitFailure(const VMContext& context = {}, const std::optional<Values>& results = std::nullopt) {
    freeResources(context, results);
    return 1;
}

}  // namespace

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const auto bytecodeFile = FLAGS_path;
    std::ifstream input(bytecodeFile, std::ios::binary);
    if (!input) {
        std::cerr << "Error: Failed to open bytecode file: " << bytecodeFile << std::endl;
        return 1;
    }
    std::cout << "Loading bytecode from file: " << bytecodeFile << std::endl;
    std::vector<uint8_t> bytecode((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::cout << "Bytecode size: " << bytecode.size() << " bytes" << std::endl;
    input.close();

    const auto printOnly = FLAGS_mode == "print" || FLAGS_mode == "print-full";
    if (printOnly) {
        std::cout << "File content:" << std::endl;
        if (npu_vm_print(bytecode.data(), static_cast<uint32_t>(bytecode.size()), FLAGS_mode == "print-full",
                         /*indent_level=*/1) != NPU_VM_SUCCESS) {
            std::cerr << "Error: Failed to print bytecode." << std::endl;
            return cleanupAndExitFailure();
        }
        return cleanupAndExitSuccess();
    }

    VMContext context;
    try {
        if (npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &context.module) !=
            NPU_VM_SUCCESS) {
            std::cerr << "Error: Failed to parse bytecode." << std::endl;
            return cleanupAndExitFailure();
        }
        if (npu_vm_get_function_info(context.module, FLAGS_function.c_str(), &context.functionInfo) != NPU_VM_SUCCESS) {
            std::cerr << "Error: No function named '" << FLAGS_function << "' found in the bytecode." << std::endl;
            return cleanupAndExitFailure(context);
        }

        auto arguments = parseParams(FLAGS_params);
        if (!arguments) {
            return cleanupAndExitFailure(context);
        }
        auto results = createResultStorage(context.functionInfo);
        if (!results.has_value()) {
            return cleanupAndExitFailure(context);
        }
        if (npu_vm_new_engine(&context.engine) != NPU_VM_SUCCESS) {
            std::cerr << "Error: Failed to create VM engine." << std::endl;
            return cleanupAndExitFailure(context, results);
        }
        if (npu_vm_load_module(context.engine, /*load_params*/ nullptr, context.module) != NPU_VM_SUCCESS) {
            std::cerr << "Error: Failed to load module into VM engine." << std::endl;
            return cleanupAndExitFailure(context, results);
        }
        if (npu_vm_call_with_results(context.engine, FLAGS_function.c_str(),
                                     static_cast<uint32_t>(arguments->values.size()), arguments->values.data(),
                                     static_cast<uint32_t>(results->values.size()),
                                     results->values.data()) != NPU_VM_SUCCESS) {
            std::cerr << "Error: Bytecode execution failed." << std::endl;
            return cleanupAndExitFailure(context, results);
        }
        printResults(context.functionInfo, *results);
        return cleanupAndExitSuccess(context, results);
    } catch (const std::exception& ex) {
        std::cerr << "Error during bytecode execution: " << ex.what() << std::endl;
        return cleanupAndExitFailure(context);
    }
}
