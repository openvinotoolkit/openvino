/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "api/C/cldnn.h"
#include "api_impl.h"
#include "engine_impl.h"
#include "topology_impl.h"
#include "program_impl.h"
#include "primitive_type.h"
#include "network_impl.h"
#include "memory_impl.h"
#include "primitive_inst.h"
#include <string>
#include <vector>

namespace cldnn {
last_err& last_err::instance() {
    thread_local static last_err _instance;
    return _instance;
}
}  // namespace cldnn

#define SHOULD_NOT_BE_NULL(arg, msg_prefix) \
    if (arg == nullptr)                     \
        throw std::invalid_argument(std::string(msg_prefix) + " should not be null.");
#define SHOULD_NOT_EQUAL_0(arg, msg_prefix) \
    if (arg == 0)                           \
        throw std::invalid_argument(std::string(msg_prefix) + " should not equals 0.");

extern "C" {

#ifndef CLDNN_VERSION_MAJOR
#define CLDNN_VERSION_MAJOR (0)
#endif

#ifndef CLDNN_VERSION_MINOR
#define CLDNN_VERSION_MINOR (0)
#endif

#ifndef CLDNN_VERSION_BUILD
#define CLDNN_VERSION_BUILD (0)
#endif

#ifndef CLDNN_VERSION_REVISION
#define CLDNN_VERSION_REVISION (0)
#endif

cldnn_version cldnn_get_version(cldnn_status* status) {
    return exception_handler<cldnn_version>(CLDNN_ERROR, status, {}, []() -> cldnn_version {
        return {CLDNN_VERSION_MAJOR, CLDNN_VERSION_MINOR, CLDNN_VERSION_BUILD, CLDNN_VERSION_REVISION};
    });
}

cldnn_topology cldnn_create_topology(cldnn_status* status) {
    return exception_handler<cldnn_topology>(CLDNN_ERROR, status, nullptr, [&]() {
        return api_cast(new cldnn::topology_impl());
    });
}

void cldnn_add_primitive(cldnn_topology topology, const CLDNN_PRIMITIVE_DESC(primitive) * dto, cldnn_status* status) {
    return exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(topology, "Topology");
        SHOULD_NOT_BE_NULL(dto, "Primitive");
        SHOULD_NOT_BE_NULL(dto->id, "Primitive id");
        SHOULD_NOT_BE_NULL(dto->type, "Primitive type");
        api_cast(topology)->add(dto->type->from_dto(dto));
    });
}

void cldnn_change_input_layout(cldnn_topology topology,
                               cldnn_primitive_id id,
                               cldnn_layout new_layout,
                               cldnn_status* status) {
    return exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(topology, "Topology");
        SHOULD_NOT_BE_NULL(id, "Input layout id");
        if (new_layout.format < cldnn_format_any || new_layout.format >= cldnn_format_format_num)
            throw std::invalid_argument("Unknown format of layout.");
        if (new_layout.data_type != cldnn_data_type::cldnn_f16 && new_layout.data_type != cldnn_data_type::cldnn_f32 &&
            new_layout.data_type != cldnn_data_type::cldnn_i8 && new_layout.data_type != cldnn_data_type::cldnn_bin &&
            new_layout.data_type != cldnn_data_type::cldnn_u8 && new_layout.data_type != cldnn_data_type::cldnn_i32 &&
            new_layout.data_type != cldnn_data_type::cldnn_i64)
            throw std::invalid_argument("Unknown data_type of layout.");
        api_cast(topology)->change_input_layout(id, (layout) new_layout);
    });
}

static void primitive_id_vector_to_char_array(char* names,
                                              size_t size,
                                              size_t* size_ret,
                                              cldnn_status* status,
                                              const std::vector<primitive_id>& vec) {
    *size_ret = std::accumulate(std::begin(vec),
                                std::end(vec),
                                size_t(1),  // final zero symbol
                                [](size_t acc, const cldnn::primitive_id& id) {
                                    return acc + id.size() + 1;  // plus zero symbol
                                });

    if (size < *size_ret) {
        if (status)
            *status = CLDNN_INVALID_ARG;
        return;
    }

    size_t i = 0;
    for (auto& id : vec) {
        // workaround for Microsoft VC++
#if defined _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
        i += id.copy(names + i, size - i - 2);
#if defined _MSC_VER
#pragma warning(pop)
#endif
        names[i++] = 0;  // plus zero symbol
        assert(i < size);
    }
    names[i] = 0;  // final zero symbol
}

void cldnn_get_primitive_ids(cldnn_topology topology, char* ids, size_t size, size_t* size_ret, cldnn_status* status) {
    return exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(topology, "Topology");
        auto ids_size = api_cast(topology)->get_primitives().size();
        SHOULD_NOT_EQUAL_0(ids_size, "Primitives number");
        auto&& primitives_ids = api_cast(topology)->get_primitives_id();
        primitive_id_vector_to_char_array(ids, size, size_ret, status, primitives_ids);
    });
}

void cldnn_retain_topology(cldnn_topology topology, cldnn_status* status) {
    return exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(topology, "Topology");
        api_cast(topology)->add_ref();
    });
}
void cldnn_release_topology(cldnn_topology topology, cldnn_status* status) {
    return exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(topology, "Topology");
        api_cast(topology)->release();
    });
}

uint32_t cldnn_get_engine_count(/*cldnn_engine_type*/ int32_t type, cldnn_status* status) {
    if (type == cldnn_engine_type::cldnn_engine_ocl) {
        if (status)
            *status = CLDNN_SUCCESS;
        return 1;
    } else {
        if (status)
            *status = CLDNN_DEVICE_ERROR;
        return 0;
    }
}

void cldnn_release_pending_memory(cldnn_engine engine, uint16_t stream_id, cldnn_status* status) {
    return exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(engine, "engine");
        api_cast(engine)->release_pending_memory(stream_id);
    });
}

cldnn_engine cldnn_create_engine(/*cldnn_engine_type*/ int32_t type,
                                 uint32_t engine_num,
                                 const cldnn_engine_configuration* configuration,
                                 cldnn_status* status) {
    if (engine_num > 0 || (type != cldnn_engine_type::cldnn_engine_ocl)) {
        if (status)
            *status = CLDNN_DEVICE_ERROR;
        return nullptr;
    }

    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]() {
        return api_cast(new cldnn::engine_impl(configuration ? cldnn::engine_configuration(*configuration)
                                                             : cldnn::engine_configuration()));
    });
}

void cldnn_retain_engine(cldnn_engine engine, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        api_cast(engine)->add_ref();
    });
}

void cldnn_release_engine(cldnn_engine engine, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        api_cast(engine)->release();
    });
}

cldnn_engine_info cldnn_get_engine_info(cldnn_engine engine, cldnn_status* status) {
    return exception_handler<cldnn_engine_info>(CLDNN_ERROR,
                                                status,
                                                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                                [&]() -> cldnn_engine_info {
                                                    SHOULD_NOT_BE_NULL(engine, "Engine");
                                                    auto info = api_cast(engine)->get_engine_info();
                                                    cldnn_engine_info res = {info.cores_count,
                                                            info.core_frequency,
                                                            info.max_work_group_size,
                                                            info.max_local_mem_size,
                                                            info.max_global_mem_size,
                                                            info.max_alloc_mem_size,
                                                            info.max_image2d_width,
                                                            info.max_image2d_height,
                                                            info.supports_fp16,
                                                            info.supports_fp16_denorms,
                                                            info.supports_subgroups_short,
                                                            info.supports_image,
                                                            info.supports_imad,
                                                            info.supports_immad
                                                            };
                                                    strncpy(res.ocl_device_name, info.dev_name.c_str(), CLDNN_API_STRING_SIZE_MAX);
                                                    strncpy(res.ocl_driver_version, info.driver_version.c_str(), CLDNN_API_STRING_SIZE_MAX);
                                                    return res;
                                                });
}

/*cldnn_engine_type*/ int32_t cldnn_get_engine_type(cldnn_engine engine, cldnn_status* status) {
    return exception_handler<int32_t>(CLDNN_ERROR, status, cldnn_engine_ocl, [&]() {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        return static_cast<int32_t>(api_cast(engine)->type());
    });
}

int64_t cldnn_get_max_used_device_memory_size(cldnn_engine engine, cldnn_status* status) {
    return exception_handler<int32_t>(CLDNN_ERROR, status, cldnn_engine_ocl, [&]() {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        return static_cast<int32_t>(api_cast(engine)->get_max_used_device_memory());
    });
}

int64_t cldnn_get_temp_used_device_memory_size(cldnn_engine engine, cldnn_status* status) {
    return exception_handler<int32_t>(CLDNN_ERROR, status, cldnn_engine_ocl, [&]() {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        return static_cast<int32_t>(api_cast(engine)->get_used_device_memory());
    });
}

cldnn_event cldnn_create_user_event(cldnn_engine engine, uint16_t stream_id, cldnn_status* status) {
    return exception_handler<cldnn_event>(CLDNN_ERROR, status, nullptr, [&]() {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        event_impl* e = api_cast(engine)->create_user_event(stream_id).detach();
        return api_cast(e);
    });
}

CLDNN_API int32_t cldnn_is_user_event(cldnn_event event, cldnn_status* status) {
    return exception_handler<int32_t>(CLDNN_ERROR, status, 0, [&]() {
        SHOULD_NOT_BE_NULL(event, "Event");
        auto user_ev = dynamic_cast<user_event*>(api_cast(event));
        return (user_ev != nullptr);
    });
}

void cldnn_retain_event(cldnn_event event, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(event, "Event");
        api_cast(event)->add_ref();
    });
}

void cldnn_release_event(cldnn_event event, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(event, "Event");
        api_cast(event)->release();
    });
}

void cldnn_wait_for_event(cldnn_event event, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(event, "Event");
        api_cast(event)->wait();
    });
}

void cldnn_set_event(cldnn_event event, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(event, "Event");
        if (auto user_ev = dynamic_cast<user_event*>(api_cast(event)))
            user_ev->set();
        else
            throw std::invalid_argument("Event passed to cldnn_set_event should be an user event");
    });
}

void cldnn_add_event_handler(cldnn_event event, cldnn_event_handler handler, void* param, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(handler, "Handler");
        SHOULD_NOT_BE_NULL(event, "Event");
        api_cast(event)->add_event_handler(handler, param);
    });
}

void cldnn_get_event_profiling_info(cldnn_event event,
                                    cldnn_profiling_interval* profiling,
                                    size_t size,
                                    size_t* size_ret,
                                    cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(event, "Event");
        if (!profiling && !size_ret) {
            if (status)
                *status = CLDNN_INVALID_ARG;
            return;
        }
        auto& profiling_info = api_cast(event)->get_profiling_info();
        if (size_ret)
            *size_ret = profiling_info.size();
        if (profiling != nullptr) {
            if (size != profiling_info.size()) {
                if (status)
                    *status = CLDNN_INVALID_ARG;
                return;
            }
            size_t i = 0;
            for (auto& info : profiling_info) {
                profiling[i].name = info.name;
                profiling[i].nanoseconds = info.nanoseconds;
                ++i;
            }
        }
    });
}

void cldnn_get_primitives_info(cldnn_network network,
                               const cldnn_primitive_info** info,
                               size_t size,
                               size_t* size_ret,
                               cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        if (!info && !size_ret) {
            if (status)
                *status = CLDNN_INVALID_ARG;
            return;
        }
        auto& primitives_info = api_cast(network)->get_primitives_info();
        if (size_ret)
            *size_ret = primitives_info.size();

        if (info != nullptr) {
            if (size != primitives_info.size()) {
                if (status)
                    *status = CLDNN_INVALID_ARG;
                return;
            }
            size_t i = 0;
            for (auto& pi : primitives_info) {
                info[i] = pi.get_dto();
                ++i;
            }
        }
    });
}

void cldnn_get_optimizer_passes_info(cldnn_network network,
                                     const cldnn_primitive_info** info,
                                     int* pass_sizes,
                                     char* pass_names,
                                     size_t total_size,
                                     size_t* total_size_ret,
                                     size_t* pass_count_ret,
                                     size_t* pass_names_total_size_ret,
                                     cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        if ((!info || !pass_sizes || !pass_names) &&
            (!total_size_ret || !pass_count_ret || !pass_names_total_size_ret)) {
            if (status)
                *status = CLDNN_INVALID_ARG;
            return;
        }

        auto& opt_passes_info = api_cast(network)->get_optimizer_passes_info();
        size_t pi_total_size = 0;
        size_t names_total_size = 0;
        std::vector<primitive_id> names;
        for (auto& step : opt_passes_info) {
            pi_total_size += step.second.size();
            names_total_size += step.first.size() + 1;
            names.push_back(step.first);
        }

        if (total_size_ret && pass_count_ret && pass_names_total_size_ret) {
            *total_size_ret = pi_total_size;
            *pass_count_ret = opt_passes_info.size();

            primitive_id_vector_to_char_array(pass_names, 0, pass_names_total_size_ret, status, names);
            // Function should return invalid arg when it is used to get output size, so reset it to success
            *status = CLDNN_SUCCESS;
        }

        if (info != nullptr && pass_sizes != nullptr && pass_names != nullptr) {
            if (total_size != pi_total_size) {
                if (status)
                    *status = CLDNN_INVALID_ARG;
                return;
            }

            primitive_id_vector_to_char_array(pass_names,
                                              *pass_names_total_size_ret,
                                              pass_names_total_size_ret,
                                              status,
                                              names);

            if (*status != CLDNN_SUCCESS)
                return;

            size_t step_idx = 0;
            size_t global_off = 0;
            for (auto& step : opt_passes_info) {
                for (auto& pi : step.second) {
                    info[global_off] = pi.get_dto();
                    global_off++;
                }
                pass_sizes[step_idx++] = static_cast<int>(step.second.size());
            }
        }
    });
}

cldnn_program cldnn_build_program(cldnn_engine engine,
                                  cldnn_topology topology,
                                  cldnn_build_option* options,
                                  size_t options_num,
                                  cldnn_status* status) {
    return exception_handler<cldnn_program>(CLDNN_ERROR, status, nullptr, [&]() {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        SHOULD_NOT_BE_NULL(topology, "Topology");
        cldnn::build_options options_obj(cldnn::array_ref<cldnn_build_option>(options, options_num));

        cldnn::program_impl* prog = api_cast(engine)->build_program(*api_cast(topology), options_obj).detach();
        return api_cast(prog);
    });
}

void cldnn_retain_program(cldnn_program program, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(program, "Program");
        api_cast(program)->add_ref();
    });
}

void cldnn_release_program(cldnn_program program, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(program, "Program");
        api_cast(program)->release();
    });
}

cldnn_network cldnn_allocate_network(cldnn_program program, uint16_t stream_id, cldnn_status* status) {
    return exception_handler<cldnn_network>(CLDNN_ERROR, status, nullptr, [&]() {
        SHOULD_NOT_BE_NULL(program, "Program");
        network_impl* p = api_cast(program)->get_engine().allocate_network(*api_cast(program), stream_id).detach();
        return api_cast(p);
    });
}

cldnn_network cldnn_build_network(cldnn_engine engine,
                                  cldnn_topology topology,
                                  cldnn_build_option* options,
                                  size_t options_num,
                                  cldnn_status* status) {
    cldnn_program program = cldnn_build_program(engine, topology, options, options_num, status);
    if (!program)
        return nullptr;

    cldnn_network network = cldnn_allocate_network(program, 0, status);
    cldnn_release_program(program, nullptr);
    return network;
}

void cldnn_retain_network(cldnn_network network, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        api_cast(network)->add_ref();
    });
}

void cldnn_release_network(cldnn_network network, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        api_cast(network)->release();
    });
}

void cldnn_set_network_input(cldnn_network network, cldnn_primitive_id id, cldnn_memory mem, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(mem, "Mem");
        auto mem_size = api_cast(mem)->size();
        SHOULD_NOT_BE_NULL(network, "Network");
        SHOULD_NOT_BE_NULL(id, "Id");
        SHOULD_NOT_EQUAL_0(mem_size, "Memory size");
        api_cast(network)->set_input_data(id, *api_cast(mem));
    });
}

void cldnn_set_learning_rate(cldnn_network network, float lr, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() { api_cast(network)->set_learning_rate(lr); });
}

float cldnn_get_learning_rate(cldnn_network network, cldnn_status* status) {
    return exception_handler<float>(CLDNN_ERROR, status, 0, [&]() { return api_cast(network)->get_learning_rate(); });
}

cldnn_engine cldnn_get_network_engine(cldnn_network network, cldnn_status* status) {
    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        refcounted_obj_ptr<cldnn::engine_impl> ptr{&api_cast(network)->get_engine()};
        return api_cast(ptr.detach());
    });
}

cldnn_program cldnn_get_network_program(cldnn_network network, cldnn_status* status) {
    return exception_handler<cldnn_program>(CLDNN_ERROR, status, nullptr, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        refcounted_obj_ptr<cldnn::program_impl> ptr{
            const_cast<cldnn::program_impl*>(&api_cast(network)->get_program())};
        return api_cast(ptr.detach());
    });
}

void cldnn_get_primitive_info(cldnn_network network,
                              cldnn_primitive_id prim_id,
                              char* info,
                              size_t size,
                              size_t* size_ret,
                              cldnn_status* status) {
    return exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        const auto& prim_info = api_cast(network)->get_primitive_info(prim_id);
        *size_ret = prim_info.size() + 1;

        if (size < *size_ret) {
            if (status)
                *status = CLDNN_INVALID_ARG;
            return;
        }

        size_t i = 0;
        for (const auto c : prim_info) {
            info[i++] = c;
            assert(i < size);
        }
        info[i] = 0;  // final zero symbol
    });
}

void cldnn_get_network_output_names(cldnn_network network,
                                    char* names,
                                    size_t size,
                                    size_t* size_ret,
                                    cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        auto&& output_ids = api_cast(network)->get_output_ids();
        SHOULD_NOT_EQUAL_0(output_ids.size(), "Output size");
        primitive_id_vector_to_char_array(names, size, size_ret, status, output_ids);
    });
}

void cldnn_get_network_executed_primitive_names(cldnn_network network,
                                                char* names,
                                                size_t size,
                                                size_t* size_ret,
                                                cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        auto&& primitive_ids = api_cast(network)->get_executed_primitive_ids();
        primitive_id_vector_to_char_array(names, size, size_ret, status, primitive_ids);
    });
}

void cldnn_get_network_all_primitive_names(cldnn_network network,
                                           char* names,
                                           size_t size,
                                           size_t* size_ret,
                                           cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        auto&& primitive_ids = api_cast(network)->get_all_primitive_ids();
        SHOULD_NOT_EQUAL_0(primitive_ids.size(), "Primitives size");
        primitive_id_vector_to_char_array(names, size, size_ret, status, primitive_ids);
    });
}

void cldnn_get_network_all_primitive_org_names(cldnn_network network,
                                               char* names,
                                               size_t size,
                                               size_t* size_ret,
                                               cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        auto&& primitive_ids = api_cast(network)->get_all_primitive_org_ids();
        SHOULD_NOT_EQUAL_0(primitive_ids.size(), "Primitives size");
        primitive_id_vector_to_char_array(names, size, size_ret, status, primitive_ids);
    });
}

void cldnn_execute_network(cldnn_network network, cldnn_event* dependencies, size_t deps_num, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        std::vector<cldnn::refcounted_obj_ptr<cldnn::event_impl>> deps;
        deps.reserve(deps_num);
        for (size_t i = 0; i < deps_num; i++) {
            deps.emplace_back(api_cast(dependencies[i]));
        }

        api_cast(network)->execute(deps);
    });
}

cldnn_network_output cldnn_get_network_output(cldnn_network network, const char* name, cldnn_status* status) {
    cldnn_network_output error_result = {nullptr, nullptr};
    return exception_handler<cldnn_network_output>(CLDNN_ERROR, status, error_result, [&]() -> cldnn_network_output {
        SHOULD_NOT_BE_NULL(network, "Network");
        SHOULD_NOT_BE_NULL(name, "ID of primitive");
        cldnn::primitive_id id(name);
        auto event = api_cast(network)->get_primitive_event(id);
        auto& mem_result = api_cast(network)->get_primitive(id)->output_memory();
        refcounted_obj_ptr<cldnn::memory_impl> mem_ptr{&mem_result};
        return {api_cast(event.detach()), api_cast(mem_ptr.detach())};
    });
}

cldnn_memory cldnn_get_network_output_memory(cldnn_network network, const char* name, cldnn_status* status) {
    cldnn_memory error_result = nullptr;
    return exception_handler<cldnn_memory>(CLDNN_ERROR, status, error_result, [&]() -> cldnn_memory {
        SHOULD_NOT_BE_NULL(network, "Network");
        SHOULD_NOT_BE_NULL(name, "ID of primitive");
        cldnn::primitive_id id(name);
        auto& mem_result = api_cast(network)->get_primitive(id)->output_memory();
        refcounted_obj_ptr<cldnn::memory_impl> mem_ptr{&mem_result};
        return api_cast(mem_ptr.detach());
    });
}

cldnn_event cldnn_get_network_output_event(cldnn_network network, const char* name, cldnn_status* status) {
    cldnn_event error_result = nullptr;
    return exception_handler<cldnn_event>(CLDNN_ERROR, status, error_result, [&]() -> cldnn_event {
        SHOULD_NOT_BE_NULL(network, "Network");
        SHOULD_NOT_BE_NULL(name, "ID of primitive");
        cldnn::primitive_id id(name);
        auto event = api_cast(network)->get_primitive_event(id);
        return api_cast(event.detach());
    });
}

cldnn_memory cldnn_allocate_memory(cldnn_engine engine, cldnn_layout layout, uint16_t stream_id, cldnn_status* status) {
    return exception_handler<cldnn_memory>(CLDNN_ERROR, status, nullptr, [&]() {
        SHOULD_NOT_BE_NULL(engine, "Engine");
        if (layout.format < cldnn_format_any || layout.format >= cldnn_format_format_num)
            throw std::invalid_argument("Unknown format of layout.");
        if (layout.data_type != cldnn_data_type::cldnn_f16 && layout.data_type != cldnn_data_type::cldnn_f32 &&
            layout.data_type != cldnn_data_type::cldnn_i8 && layout.data_type != cldnn_data_type::cldnn_u8 &&
            layout.data_type != cldnn_data_type::cldnn_bin && layout.data_type != cldnn_data_type::cldnn_i32 &&
            layout.data_type != cldnn_data_type::cldnn_i64)
            throw std::invalid_argument("Unknown data_type of layout.");

        cldnn::memory_impl* mem_ptr = api_cast(engine)->allocate_memory((cldnn::layout)layout, stream_id).detach();
        return api_cast(mem_ptr);
    });
}

cldnn_memory cldnn_attach_memory(cldnn_layout layout,
                                 void* pointer,
                                 size_t size,
                                 uint16_t stream_id,
                                 cldnn_status* status) {
    return exception_handler<cldnn_memory>(CLDNN_ERROR, status, nullptr, [&]() {
        cldnn::layout layout_obj(layout);
        if (layout_obj.bytes_count() > size)
            throw std::invalid_argument("buffer size does not match layout size");
        return api_cast(new cldnn::simple_attached_memory(layout_obj, pointer, stream_id));
    });
}

CLDNN_API int32_t cldnn_is_the_same_buffer(cldnn_memory mem1, cldnn_memory mem2, cldnn_status* status) {
    return static_cast<int32_t>(exception_handler<bool>(CLDNN_ERROR, status, false, [&]() {
        SHOULD_NOT_BE_NULL(mem1, "Memory");
        SHOULD_NOT_BE_NULL(mem2, "Memory");

        if (mem1 == mem2)
            return true;

        if (api_cast(mem1)->get_engine() != api_cast(mem2)->get_engine())
            return false;

        // memories were allocated by the user so just check if pointers match
        if (!api_cast(mem1)->get_engine())
            return api_cast(mem1)->lock() == api_cast(mem2)->lock();

        // memories were allocated by the engine so let it decide whether they refer to the same buffer
        return api_cast(mem1)->get_engine()->is_the_same_buffer(*api_cast(mem1), *api_cast(mem2));
    }));
}

void cldnn_retain_memory(cldnn_memory memory, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        api_cast(memory)->add_ref();
    });
}

void cldnn_release_memory(cldnn_memory memory, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        api_cast(memory)->release();
    });
}

void* cldnn_lock_memory(cldnn_memory memory, cldnn_status* status) {
    return exception_handler<void*>(CLDNN_ERROR, status, nullptr, [&]() {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        return api_cast(memory)->lock();
    });
}

void cldnn_unlock_memory(cldnn_memory memory, cldnn_status* status) {
    exception_handler(CLDNN_ERROR, status, [&]() {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        api_cast(memory)->unlock();
    });
}

cldnn_layout cldnn_get_memory_layout(cldnn_memory memory, cldnn_status* status) {
    cldnn_layout error_result = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, {0, 0, 0, 0});

    return exception_handler<cldnn_layout>(CLDNN_ERROR, status, error_result, [&]() {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        auto memory_size = api_cast(memory)->size();
        SHOULD_NOT_EQUAL_0(memory_size, "Memory size");
        return api_cast(memory)->get_layout();
    });
}

uint16_t cldnn_get_memory_stream_id(cldnn_memory memory, cldnn_status* status) {
    return exception_handler<uint16_t>(CLDNN_ERROR, status, 0, [&]() {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        return api_cast(memory)->get_stream_id();
    });
}

uint16_t cldnn_get_network_stream_id(cldnn_network network, cldnn_status* status) {
    return exception_handler<uint16_t>(CLDNN_ERROR, status, 0, [&]() {
        SHOULD_NOT_BE_NULL(network, "Network");
        return api_cast(network)->get_stream_id();
    });
}

cldnn_engine cldnn_get_memory_engine(cldnn_memory memory, cldnn_status* status) {
    return exception_handler<cldnn_engine>(CLDNN_ERROR, status, nullptr, [&]() {
        SHOULD_NOT_BE_NULL(memory, "Memory");
        auto engine = api_cast(memory)->get_engine();
        return api_cast(engine.detach());
    });
}

const char* cldnn_get_last_error_message() {
    try {
        return cldnn::last_err::instance().get_last_error_message().c_str();
    } catch (...) {
        return "Reading error message failed.";
    }
}

CLDNN_API uint16_t cldnn_float_to_half(float value, cldnn_status* status) {
    return exception_handler<uint16_t>(CLDNN_ERROR, status, 0, [&]() { return cldnn::float_to_half(value); });
}

CLDNN_API float cldnn_half_to_float(uint16_t value, cldnn_status* status) {
    return exception_handler<float>(CLDNN_ERROR, status, 0.0f, [&]() { return cldnn::half_to_float(value); });
}

} /* extern "C" */

#define PRIMITIVE_TYPE_ID_CALL_IMPL(PType)                                                       \
    namespace cldnn {                                                                            \
    primitive_type_id PType##_type_id();                                                         \
    }                                                                                            \
    extern "C" CLDNN_API cldnn_primitive_type_id cldnn_##PType##_type_id(cldnn_status* status) { \
        return exception_handler<cldnn_primitive_type_id>(CLDNN_ERROR, status, nullptr, []() {   \
            return cldnn::PType##_type_id();                                                     \
        });                                                                                      \
    }

PRIMITIVE_TYPE_ID_CALL_IMPL(activation)
PRIMITIVE_TYPE_ID_CALL_IMPL(activation_grad)
PRIMITIVE_TYPE_ID_CALL_IMPL(arg_max_min)
PRIMITIVE_TYPE_ID_CALL_IMPL(average_unpooling)
PRIMITIVE_TYPE_ID_CALL_IMPL(batch_norm)
PRIMITIVE_TYPE_ID_CALL_IMPL(batch_norm_grad)
PRIMITIVE_TYPE_ID_CALL_IMPL(border)
PRIMITIVE_TYPE_ID_CALL_IMPL(broadcast)
PRIMITIVE_TYPE_ID_CALL_IMPL(convolution)
PRIMITIVE_TYPE_ID_CALL_IMPL(crop)
PRIMITIVE_TYPE_ID_CALL_IMPL(data)
PRIMITIVE_TYPE_ID_CALL_IMPL(embed)
PRIMITIVE_TYPE_ID_CALL_IMPL(mutable_data)
PRIMITIVE_TYPE_ID_CALL_IMPL(deconvolution)
PRIMITIVE_TYPE_ID_CALL_IMPL(concatenation)
PRIMITIVE_TYPE_ID_CALL_IMPL(eltwise)
PRIMITIVE_TYPE_ID_CALL_IMPL(fully_connected)
PRIMITIVE_TYPE_ID_CALL_IMPL(fused_conv_bn_scale)
PRIMITIVE_TYPE_ID_CALL_IMPL(fused_conv_eltwise)
PRIMITIVE_TYPE_ID_CALL_IMPL(input_layout)
PRIMITIVE_TYPE_ID_CALL_IMPL(lookup_table)
PRIMITIVE_TYPE_ID_CALL_IMPL(lrn)
PRIMITIVE_TYPE_ID_CALL_IMPL(max_unpooling)
PRIMITIVE_TYPE_ID_CALL_IMPL(permute)
PRIMITIVE_TYPE_ID_CALL_IMPL(pooling)
PRIMITIVE_TYPE_ID_CALL_IMPL(reorder)
PRIMITIVE_TYPE_ID_CALL_IMPL(reshape)
PRIMITIVE_TYPE_ID_CALL_IMPL(scale)
PRIMITIVE_TYPE_ID_CALL_IMPL(scale_grad_input)
PRIMITIVE_TYPE_ID_CALL_IMPL(scale_grad_weights)
PRIMITIVE_TYPE_ID_CALL_IMPL(softmax)
PRIMITIVE_TYPE_ID_CALL_IMPL(region_yolo)
PRIMITIVE_TYPE_ID_CALL_IMPL(reorg_yolo)
PRIMITIVE_TYPE_ID_CALL_IMPL(proposal)
PRIMITIVE_TYPE_ID_CALL_IMPL(roi_pooling)
PRIMITIVE_TYPE_ID_CALL_IMPL(prior_box)
PRIMITIVE_TYPE_ID_CALL_IMPL(detection_output)
PRIMITIVE_TYPE_ID_CALL_IMPL(detection_output_sort)
PRIMITIVE_TYPE_ID_CALL_IMPL(normalize)
PRIMITIVE_TYPE_ID_CALL_IMPL(generic_layer)
PRIMITIVE_TYPE_ID_CALL_IMPL(custom_gpu_primitive)
PRIMITIVE_TYPE_ID_CALL_IMPL(split)
PRIMITIVE_TYPE_ID_CALL_IMPL(upsampling)
PRIMITIVE_TYPE_ID_CALL_IMPL(convolution_grad_weights)
PRIMITIVE_TYPE_ID_CALL_IMPL(apply_adam)
PRIMITIVE_TYPE_ID_CALL_IMPL(mvn)
PRIMITIVE_TYPE_ID_CALL_IMPL(fully_connected_grad_input)
PRIMITIVE_TYPE_ID_CALL_IMPL(fully_connected_grad_weights)
PRIMITIVE_TYPE_ID_CALL_IMPL(lstm)
PRIMITIVE_TYPE_ID_CALL_IMPL(lstm_gemm)
PRIMITIVE_TYPE_ID_CALL_IMPL(lstm_elt)
PRIMITIVE_TYPE_ID_CALL_IMPL(softmax_loss_grad)
PRIMITIVE_TYPE_ID_CALL_IMPL(tile)
PRIMITIVE_TYPE_ID_CALL_IMPL(gemm)
PRIMITIVE_TYPE_ID_CALL_IMPL(select)
PRIMITIVE_TYPE_ID_CALL_IMPL(index_select)
PRIMITIVE_TYPE_ID_CALL_IMPL(condition)
PRIMITIVE_TYPE_ID_CALL_IMPL(pyramid_roi_align)
PRIMITIVE_TYPE_ID_CALL_IMPL(contract)
PRIMITIVE_TYPE_ID_CALL_IMPL(one_hot)
PRIMITIVE_TYPE_ID_CALL_IMPL(gather)
PRIMITIVE_TYPE_ID_CALL_IMPL(depth_to_space)
PRIMITIVE_TYPE_ID_CALL_IMPL(shuffle_channels)
PRIMITIVE_TYPE_ID_CALL_IMPL(strided_slice)
PRIMITIVE_TYPE_ID_CALL_IMPL(reverse_sequence)
PRIMITIVE_TYPE_ID_CALL_IMPL(binary_convolution)
PRIMITIVE_TYPE_ID_CALL_IMPL(quantize)
PRIMITIVE_TYPE_ID_CALL_IMPL(lstm_dynamic)
PRIMITIVE_TYPE_ID_CALL_IMPL(lstm_dynamic_input)
PRIMITIVE_TYPE_ID_CALL_IMPL(lstm_dynamic_timeloop)
PRIMITIVE_TYPE_ID_CALL_IMPL(reduce)
PRIMITIVE_TYPE_ID_CALL_IMPL(deformable_interp)
PRIMITIVE_TYPE_ID_CALL_IMPL(deformable_conv)
