//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NPU_VIRTUAL_MACHINE_H
#define NPU_VIRTUAL_MACHINE_H

#include "vm_runtime/npu_vm_runtime.hpp"

#if defined(__cplusplus)
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifndef NPU_VM_APICALL
#if defined(_WIN32)
#define NPU_VM_APICALL __cdecl
#else
#define NPU_VM_APICALL
#endif  // defined(_WIN32)
#endif  // NPU_VM_APICALL

#ifndef NPU_VM_EXPORT
#if defined(_WIN32)
#define NPU_VM_EXPORT __declspec(dllexport)
#else
#define NPU_VM_EXPORT __attribute__((visibility("default")))
#endif  // defined(_WIN32)
#endif  // NPU_VM_EXPORT

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN

typedef enum _npu_vm_result {
    NPU_VM_SUCCESS = 0,
    NPU_VM_ERROR_INVALID_NULL_POINTER = 0x80000001,
    NPU_VM_ERROR_MODULE_NOT_FOUND = 0x80000002,
    NPU_VM_ERROR_FUNCTION_NOT_FOUND = 0x80000003,
    NPU_VM_ERROR_FUNCTION_NAME_TOO_LONG = 0x80000004,
    NPU_VM_ERROR_FUNCTION_CALL_FAILED = 0x80000005,
    NPU_VM_ERROR_INFERENCE_FAILED = 0x80000006,
    NPU_VM_ERROR_OUTPUT_SHAPE_PREDICTION_FAILED = 0x80000007,
    NPU_VM_ERROR_MEMORY_ALLOCATION_FAILED = 0x80000008,
    NPU_VM_ERROR_UNKNOWN = 0x8ffffffe,
} npu_vm_result;

typedef enum _npu_vm_type {
    npu_vm_type_int8 = 0,
    npu_vm_type_int16 = 1,
    npu_vm_type_int32 = 2,
    npu_vm_type_int64 = 3,
    npu_vm_type_uint8 = 4,
    npu_vm_type_uint16 = 5,
    npu_vm_type_uint32 = 6,
    npu_vm_type_uint64 = 7,
    npu_vm_type_float32 = 8,
    npu_vm_type_float64 = 9,
    npu_vm_type_buffer = 10
} npu_vm_type;

typedef union _npu_vm_value {
    int8_t i8;
    int16_t i16;
    int32_t i32;
    int64_t i64;
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    float f32;
    double f64;
    struct {
        uint8_t* data;
        uint32_t size;
    } buffer;
} npu_vm_value;

typedef struct _npu_vm_function_info {
    char* name;  // Null-terminated function name string
    uint32_t num_params;
    npu_vm_type* param_types;
    uint32_t num_results;
    npu_vm_type* result_types;
} npu_vm_function_info;

typedef struct _npu_vm_command_queue_desc {
    ze_command_queue_priority_t priority;
    ze_command_queue_workload_type_t workload_type;
    uint32_t option_flags;
} npu_vm_command_queue_desc;

typedef struct _npu_vm_module_load_params {
    npu_vm_runtime_create_execution_context_params_t* create_execution_context_params;
    /// Stores the number concurrent command lists that can be executed in parallel.
    /// If the value is 1, the runtime will consider that the compiled schedule uses
    /// the max number of tiles
    uint32_t concurrentJobCount;

    // If true, the runtime will create an internal command queue for the engine to use.
    // If false, the runtime will use the command queue provided in create_execution_context_params
    bool internalCommandQueue;

    // If true, the runtime will use counter-based events for synchronization between subgraphs.
    bool counterBasedEventSupported;
} npu_vm_module_load_params;

typedef struct _npu_vm_module npu_vm_module;
typedef struct _npu_vm_engine npu_vm_engine;

#define NPU_VM_MAIN_INFERENCE_FUNCTION_NAME "main"
#define NPU_VM_OUTPUT_SHAPE_PREDICTION_FUNCTION_NAME "output_shape"

/// Print the bytecode binary in a human-readable format. The output is printed to stdout
/// @param bytecode Pointer to the bytecode binary data
/// @param bytecode_size Size of the bytecode binary in bytes
/// @param print_full If true, large constants are printed in full. Non-zero values are treated as true
/// @param indent_level The indentation level for pretty-printing nested structures
/// @return NPU_VM_SUCCESS if printing succeeded, or an appropriate error code if parsing the bytecode failed
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_print(const uint8_t* bytecode, uint32_t bytecode_size, int print_full,
                                                        uint32_t indent_level);

/// Deserializes a bytecode binary and stores this information inside a `npu_vm_module` for later execution. The
/// resulting module instance references the provided bytecode data, so the caller must ensure that the bytecode data
/// remains valid for the lifetime of the module.
/// @param bytecode Pointer to the bytecode binary data
/// @param bytecode_size Size of the bytecode binary in bytes
/// @param module_out Output parameter where to store the deserialized module. The caller is responsible for freeing the
/// memory allocated for the output module, by calling `npu_vm_destroy_module()`
/// @return NPU_VM_SUCCESS if parsing succeeded, or an appropriate error code if parsing failed
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_parse_module(const uint8_t* bytecode, uint32_t bytecode_size,
                                                               npu_vm_module** module_out);

/// Destroy an instance of `npu_vm_module` and free the memory allocated for its fields
/// @param module The `npu_vm_module` structure to free
/// @return NPU_VM_SUCCESS if the memory was successfully freed, or an appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_destroy_module(npu_vm_module* module);

/// Returns information about a function by name. The information is stored in the info_out parameter
/// @param module The module instance
/// @param name The name of the function to query
/// @param info_out Output parameter where to store the function information. The caller is responsible for freeing the
/// memory allocated for the output structure, by calling `npu_vm_destroy_function_info()`
/// @return NPU_VM_SUCCESS if the function was found and the information was successfully stored in `info_out`, or an
/// appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_get_function_info(npu_vm_module* module, const char* name,
                                                                    npu_vm_function_info** info_out);

/// Destroy an instance of `npu_vm_function_info` and free the memory allocated for its fields
/// @param info The `npu_vm_function_info` structure to free
/// @return NPU_VM_SUCCESS if the memory was successfully freed, or an appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_destroy_function_info(npu_vm_function_info* info);

/// Creates a new engine instance. This instance can be used to load a module and execute its functions. The caller is
/// responsible for destroying the created engine instance by calling `npu_vm_destroy_engine()`
/// @param engine_out Output parameter where to store the created engine instance
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_new_engine(npu_vm_engine** engine_out);

/// Destroys an engine instance
/// @param engine The engine instance to destroy
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_destroy_engine(npu_vm_engine* engine);

/// Load a module into the engine instance. The loaded module can then be executed by calling its functions through the
/// engine. The state of the engine is reset when a new module is loaded
/// @param engine The engine instance where to load the module
/// @param module The module to load into the engine
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_load_module(npu_vm_engine* engine, npu_vm_module_load_params* params,
                                                              npu_vm_module* module);

/// Resets the internal state of the engine, to prepare for a new inference execution
/// @param engine The engine instance to reset
/// @param resetExecutionContext Whether to reset the execution context. Non-zero values are treated as true
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_reset_state(npu_vm_engine* engine, int resetExecutionContext);

/// Calls the function with the given name and the specified arguments
/// @param engine The engine instance
/// @param name The name of the function to call
/// @param num_arguments The number of arguments provided in the `arguments` array. This must match the function's
/// parameter count, and the types of the provided arguments must match the function's parameter types
/// @param arguments The argument values to pass to the function
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
/// @details The called function must not return any values; otherwise use `vm_call_with_results`.
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_call(npu_vm_engine* engine, const char* name, uint32_t num_arguments,
                                                       const npu_vm_value* arguments);

/// Calls the function with the given name and the specified arguments, then returns the results in the provided results
/// storage
/// @param engine The engine instance
/// @param name The name of the function to call
/// @param num_arguments The number of arguments provided in the `arguments` array. This must match the function's
/// parameter count, and the types of the provided arguments must match the function's parameter types
/// @param arguments The argument values to pass to the function
/// @param num_results The number of results expected from the function, and the size of the `results` array. This must
/// match the function's result count, and the types of the expected results must match the function's result types
/// @param results The storage where to write the function's result values
/// @details In case the result types include buffers, the VM can either use the provided buffers, or allocate memory
/// internally and set the `data` and `size` fields in the corresponding `npu_vm_value` entries. If the data pointer of
/// a buffer is null and the size is zero, the VM will allocate the buffer internally; if the data pointer is not null,
/// the VM will write the result to the provided buffer, and the caller is responsible for ensuring that the buffer has
/// sufficient size to hold the result. In case the VM allocates the buffer internally, the caller is responsible for
/// freeing the allocated memory after the call, by using `free()`
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_call_with_results(npu_vm_engine* engine, const char* name,
                                                                    uint32_t num_arguments,
                                                                    const npu_vm_value* arguments, uint32_t num_results,
                                                                    npu_vm_value* results);

/// Calls the main inference function with the specified execution parameters
/// @param engine The engine instance
/// @param params The execution parameters from the VM Runtime API, including input and output buffer information
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
/// @details The main inference function is expected to have the name 'main' and to have the same number and types of
/// parameters as specified in the execution parameters (i.e. buffers)
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_infer(npu_vm_engine* engine, npu_vm_runtime_execute_params_t* params);

/// Calls the main inference function with the specified execution parameters
/// @param engine The engine instance
/// @param params The execution parameters from the VM Runtime API, including input and output buffer information
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
/// @details The main inference function is expected to have the name 'main' and to have the same number and types of
/// parameters as specified in the execution parameters (i.e. buffers)
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_infer2(npu_vm_engine* engine,
                                                         npu_vm_runtime_execute_params2_t* params);

/// Calls the output shape prediction function with the specified execution parameters
/// @param engine The engine instance
/// @param params The execution parameters from the VM Runtime API, specifying the model's input buffer information
/// needed for output shape prediction, and the result buffers where to store the predicted output shapes
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
/// @details The output shape prediction function is expected to have the name 'output_shape' and to have the
/// same number and types of parameters as specified in the output shape prediction execution parameters (i.e. buffers)
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL
npu_vm_predict_output_shape(npu_vm_engine* engine, npu_vm_runtime_predict_output_shape_params_t2* params);

/// Returns true if host code execution is required for the given execution parameters
/// @param params The execution parameters from the VM Runtime API, including input and output buffer information
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
NPU_VM_EXPORT bool NPU_VM_APICALL npu_vm_host_code_execution_required(npu_vm_engine* engine,
                                                                      npu_vm_runtime_execute_params2_t* params);

/// Submit recorded command lists
/// @param engine The engine instance
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL
npu_vm_submit_recorded_command_lists(npu_vm_engine* engine, npu_vm_runtime_execute_params2_t* params);

/// Host synchronizes the engine, blocking until all previously submitted commands have completed execution.
/// @param engine The engine instance
/// @return NPU_VM_SUCCESS if the function executed and finalized successfully, or an appropriate error code otherwise
NPU_VM_EXPORT npu_vm_result NPU_VM_APICALL npu_vm_host_sync(npu_vm_engine* engine);

// NOLINTEND

#ifdef __cplusplus
}
#endif

#endif  // NPU_VIRTUAL_MACHINE_H
