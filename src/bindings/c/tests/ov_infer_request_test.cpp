// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <mutex>

#include "ov_test.hpp"

inline void get_tensor_info(ov_model_t* model, bool input, char** name, ov_shape_t* shape, ov_element_type_e* type) {
    ov_output_const_port* port = nullptr;
    if (input) {
        OV_EXPECT_OK(ov_model_const_input(model, &port));
    } else {
        OV_EXPECT_OK(ov_model_const_output(model, &port));
    }
    EXPECT_NE(nullptr, port);

    OV_EXPECT_OK(ov_port_get_any_name(port, name));
    EXPECT_NE(nullptr, *name);

    OV_ASSERT_OK(ov_const_port_get_shape(port, shape));
    OV_EXPECT_OK(ov_port_get_element_type(port, type));

    ov_partial_shape_t p_shape;
    p_shape.dims = nullptr;
    OV_EXPECT_OK(ov_port_get_partial_shape(port, &p_shape));
    ov_partial_shape_free(&p_shape);

    ov_output_const_port_free(port);
}

class ov_infer_request : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        auto device_name = GetParam();
        core = nullptr;
        in_tensor_name = nullptr;
        model = nullptr;
        input_tensor = nullptr;
        output_tensor = nullptr;
        compiled_model = nullptr;
        infer_request = nullptr;

        OV_EXPECT_OK(ov_core_create(&core));
        EXPECT_NE(nullptr, core);

        OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
        EXPECT_NE(nullptr, model);

        ov_shape_t tensor_shape = {0, nullptr};
        ov_element_type_e tensor_type;
        get_tensor_info(model, true, &in_tensor_name, &tensor_shape, &tensor_type);

        OV_EXPECT_OK(ov_tensor_create(tensor_type, tensor_shape, &input_tensor));
        EXPECT_NE(nullptr, input_tensor);

        OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
        EXPECT_NE(nullptr, compiled_model);

        OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
        EXPECT_NE(nullptr, infer_request);

        ov_shape_free(&tensor_shape);
    }
    void TearDown() override {
        ov_tensor_free(input_tensor);
        ov_tensor_free(output_tensor);
        ov_free(in_tensor_name);
        ov_infer_request_free(infer_request);
        ov_compiled_model_free(compiled_model);
        ov_model_free(model);
        ov_core_free(core);
    }

public:
    ov_core_t* core;
    ov_model_t* model;
    ov_compiled_model_t* compiled_model;
    ov_infer_request_t* infer_request;
    char* in_tensor_name;
    ov_tensor_t* input_tensor;
    ov_tensor_t* output_tensor;
    static std::mutex m;
    static bool ready;
    static std::condition_variable condVar;
};
bool ov_infer_request::ready = false;
std::mutex ov_infer_request::m;
std::condition_variable ov_infer_request::condVar;

class ov_infer_request_ppp : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        auto device_name = GetParam();
        output_tensor = nullptr;
        input_tensor = nullptr;
        core = nullptr;
        model = nullptr;
        preprocess = nullptr;
        input_info = nullptr;
        input_tensor_info = nullptr;
        ov_layout_t* layout = nullptr;
        input_process = nullptr;
        input_model = nullptr;
        ov_layout_t* model_layout = nullptr;
        compiled_model = nullptr;
        infer_request = nullptr;

        OV_EXPECT_OK(ov_core_create(&core));
        EXPECT_NE(nullptr, core);

        OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
        EXPECT_NE(nullptr, model);

        OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
        EXPECT_NE(nullptr, preprocess);

        OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info_by_index(preprocess, 0, &input_info));
        EXPECT_NE(nullptr, input_info);

        OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(input_info, &input_tensor_info));
        EXPECT_NE(nullptr, input_tensor_info);

        ov_shape_t shape = {0, nullptr};
        int64_t dims[4] = {1, 224, 224, 3};
        OV_EXPECT_OK(ov_shape_create(4, dims, &shape));

        ov_element_type_e type = U8;
        OV_EXPECT_OK(ov_tensor_create(type, shape, &input_tensor));
        OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_from(input_tensor_info, input_tensor));
        OV_EXPECT_OK(ov_shape_free(&shape));

        const char* layout_desc = "NHWC";
        OV_EXPECT_OK(ov_layout_create(layout_desc, &layout));
        OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_layout(input_tensor_info, layout));
        ov_layout_free(layout);

        OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(input_info, &input_process));
        EXPECT_NE(nullptr, input_process);
        OV_EXPECT_OK(
            ov_preprocess_preprocess_steps_resize(input_process, ov_preprocess_resize_algorithm_e::RESIZE_LINEAR));

        OV_EXPECT_OK(ov_preprocess_input_info_get_model_info(input_info, &input_model));
        EXPECT_NE(nullptr, input_model);

        const char* model_layout_desc = "NCHW";
        OV_EXPECT_OK(ov_layout_create(model_layout_desc, &model_layout));
        OV_EXPECT_OK(ov_preprocess_input_model_info_set_layout(input_model, model_layout));
        ov_layout_free(model_layout);

        OV_EXPECT_OK(ov_preprocess_prepostprocessor_build(preprocess, &model));
        EXPECT_NE(nullptr, model);

        OV_EXPECT_OK(ov_core_compile_model(core, model, device_name.c_str(), 0, &compiled_model));
        EXPECT_NE(nullptr, compiled_model);

        OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
        EXPECT_NE(nullptr, infer_request);
    }
    void TearDown() override {
        ov_tensor_free(output_tensor);
        ov_tensor_free(input_tensor);
        ov_infer_request_free(infer_request);
        ov_compiled_model_free(compiled_model);
        ov_preprocess_input_model_info_free(input_model);
        ov_preprocess_preprocess_steps_free(input_process);
        ov_preprocess_input_tensor_info_free(input_tensor_info);
        ov_preprocess_input_info_free(input_info);
        ov_preprocess_prepostprocessor_free(preprocess);
        ov_model_free(model);
        ov_core_free(core);
    }

public:
    ov_core_t* core;
    ov_model_t* model;
    ov_compiled_model_t* compiled_model;
    ov_infer_request_t* infer_request;
    ov_tensor_t* input_tensor;
    ov_tensor_t* output_tensor;
    ov_preprocess_prepostprocessor_t* preprocess;
    ov_preprocess_input_info_t* input_info;
    ov_preprocess_input_tensor_info_t* input_tensor_info;
    ov_preprocess_preprocess_steps_t* input_process;
    ov_preprocess_input_model_info_t* input_model;
};

INSTANTIATE_TEST_SUITE_P(device_name, ov_infer_request, ::testing::Values("CPU"));
INSTANTIATE_TEST_SUITE_P(device_name, ov_infer_request_ppp, ::testing::Values("CPU"));

TEST_P(ov_infer_request, set_tensor) {
    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, in_tensor_name, input_tensor));
}

TEST_P(ov_infer_request, set_input_tensor_by_index) {
    OV_EXPECT_OK(ov_infer_request_set_input_tensor_by_index(infer_request, 0, input_tensor));
}

TEST_P(ov_infer_request, set_input_tensor) {
    OV_EXPECT_OK(ov_infer_request_set_input_tensor(infer_request, input_tensor));
}

TEST_P(ov_infer_request, set_output_tensor_by_index) {
    OV_EXPECT_OK(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
    EXPECT_NE(nullptr, output_tensor);
    OV_EXPECT_OK(ov_infer_request_set_output_tensor_by_index(infer_request, 0, output_tensor));
}

TEST_P(ov_infer_request, set_output_tensor) {
    OV_EXPECT_OK(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
    EXPECT_NE(nullptr, output_tensor);
    OV_EXPECT_OK(ov_infer_request_set_output_tensor(infer_request, output_tensor));
}

TEST_P(ov_infer_request, set_tensor_error_handling) {
    OV_EXPECT_NOT_OK(ov_infer_request_set_tensor(nullptr, in_tensor_name, input_tensor));
    OV_EXPECT_NOT_OK(ov_infer_request_set_tensor(infer_request, nullptr, input_tensor));
    OV_EXPECT_NOT_OK(ov_infer_request_set_tensor(infer_request, in_tensor_name, nullptr));
}

TEST_P(ov_infer_request, get_tensor) {
    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, in_tensor_name, &input_tensor));
    EXPECT_NE(nullptr, input_tensor);
}

TEST_P(ov_infer_request, get_input_tensor_by_index) {
    OV_EXPECT_OK(ov_infer_request_get_input_tensor_by_index(infer_request, 0, &output_tensor));
}

TEST_P(ov_infer_request, get_input_tensor) {
    OV_EXPECT_OK(ov_infer_request_get_input_tensor(infer_request, &output_tensor));
}

TEST_P(ov_infer_request, get_output_tensor_by_index) {
    OV_EXPECT_OK(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
}

TEST_P(ov_infer_request, get_output_tensor) {
    OV_EXPECT_OK(ov_infer_request_get_output_tensor(infer_request, &output_tensor));
}

TEST_P(ov_infer_request, get_tensor_error_handling) {
    OV_EXPECT_NOT_OK(ov_infer_request_get_tensor(nullptr, in_tensor_name, &input_tensor));
    OV_EXPECT_NOT_OK(ov_infer_request_get_tensor(infer_request, nullptr, &input_tensor));
    OV_EXPECT_NOT_OK(ov_infer_request_get_tensor(infer_request, in_tensor_name, nullptr));
}

TEST_P(ov_infer_request, infer) {
    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, in_tensor_name, input_tensor));

    OV_ASSERT_OK(ov_infer_request_infer(infer_request));

    char* out_tensor_name = nullptr;
    ov_shape_t tensor_shape = {0, nullptr};
    ov_element_type_e tensor_type;
    get_tensor_info(model, false, &out_tensor_name, &tensor_shape, &tensor_type);

    OV_EXPECT_OK(ov_infer_request_get_tensor(infer_request, out_tensor_name, &output_tensor));
    EXPECT_NE(nullptr, output_tensor);

    ov_shape_free(&tensor_shape);
    ov_free(out_tensor_name);
}

TEST_P(ov_infer_request, cancel) {
    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, in_tensor_name, input_tensor));

    OV_EXPECT_OK(ov_infer_request_cancel(infer_request));
}

TEST_P(ov_infer_request_ppp, infer_ppp) {
    OV_EXPECT_OK(ov_infer_request_set_input_tensor_by_index(infer_request, 0, input_tensor));

    OV_EXPECT_OK(ov_infer_request_infer(infer_request));

    OV_EXPECT_OK(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
    EXPECT_NE(nullptr, output_tensor);
}

TEST(ov_infer_request, infer_error_handling) {
    OV_EXPECT_NOT_OK(ov_infer_request_infer(nullptr));
}

TEST_P(ov_infer_request, infer_async) {
    OV_EXPECT_OK(ov_infer_request_set_input_tensor_by_index(infer_request, 0, input_tensor));

    OV_ASSERT_OK(ov_infer_request_start_async(infer_request));

    if (!HasFatalFailure()) {
        OV_EXPECT_OK(ov_infer_request_wait(infer_request));

        OV_EXPECT_OK(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
        EXPECT_NE(nullptr, output_tensor);
    }
}

TEST_P(ov_infer_request_ppp, infer_async_ppp) {
    OV_EXPECT_OK(ov_infer_request_set_input_tensor_by_index(infer_request, 0, input_tensor));

    OV_ASSERT_OK(ov_infer_request_start_async(infer_request));

    if (!HasFatalFailure()) {
        OV_EXPECT_OK(ov_infer_request_wait(infer_request));

        OV_EXPECT_OK(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
        EXPECT_NE(nullptr, output_tensor);
    }
}

inline void infer_request_callback(void* args) {
    ov_infer_request_t* infer_request = (ov_infer_request_t*)args;
    ov_tensor_t* out_tensor = nullptr;

    OV_EXPECT_OK(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &out_tensor));
    EXPECT_NE(nullptr, out_tensor);

    ov_tensor_free(out_tensor);

    std::lock_guard<std::mutex> lock(ov_infer_request::m);
    ov_infer_request::ready = true;
    ov_infer_request::condVar.notify_one();
}

TEST_P(ov_infer_request, infer_request_set_callback) {
    OV_EXPECT_OK(ov_infer_request_set_input_tensor_by_index(infer_request, 0, input_tensor));

    ov_callback_t callback;
    callback.callback_func = infer_request_callback;
    callback.args = infer_request;

    OV_EXPECT_OK(ov_infer_request_set_callback(infer_request, &callback));

    OV_ASSERT_OK(ov_infer_request_start_async(infer_request));

    if (!HasFatalFailure()) {
        std::unique_lock<std::mutex> lock(ov_infer_request::m);
        ov_infer_request::condVar.wait(lock, [] {
            return ov_infer_request::ready;
        });
    }
}

TEST_P(ov_infer_request, get_profiling_info) {
    auto device_name = GetParam();
    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, in_tensor_name, input_tensor));

    OV_EXPECT_OK(ov_infer_request_infer(infer_request));

    OV_EXPECT_OK(ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor));
    EXPECT_NE(nullptr, output_tensor);

    ov_profiling_info_list_t profiling_infos;
    profiling_infos.size = 0;
    profiling_infos.profiling_infos = nullptr;
    OV_EXPECT_OK(ov_infer_request_get_profiling_info(infer_request, &profiling_infos));
    EXPECT_NE(0, profiling_infos.size);
    EXPECT_NE(nullptr, profiling_infos.profiling_infos);

    ov_profiling_info_list_free(&profiling_infos);
}