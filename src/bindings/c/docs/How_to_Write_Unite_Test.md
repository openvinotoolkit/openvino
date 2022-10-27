# How to Write Unite Test for C API

To ensure the accuracy of C API, all interfaces need to implement function level unite test at least. According to the object define, unite test cases are classified into following components:

| Unite Case Component | Location | Description |
|:---     |:---   |:---
|Core|[ov_core_test.cpp](../tests/ov_core_test.cpp)| including all core related interfaces tests
|Model|[ov_model_test.cpp](../tests/ov_model_test.cpp)| including all model related interfaces tests
|Compiled Model|[ov_compiled_model_test.cpp](../tests/ov_compiled_model_test.cpp)| including all compiled model related interfaces tests
|Infer Request|[ov_infer_request_test.cpp](../tests/ov_infer_request_test.cpp)| including all infer request related interfaces tests
|Tensor|[ov_tensor_test.cpp](../tests/ov_tensor_test.cpp)| including all tensor related interfaces tests
|Partial Shape|[ov_partial_shape_test.cpp](../tests/ov_partial_shape_test.cpp)| including all partial shape related interfaces tests
|Layout|[ov_layout_test.cpp](../tests/ov_layout_test.cpp)| including all layout related interfaces tests
|Preprocess|[ov_preprocess_test.cpp](../tests/ov_preprocess_test.cpp)| including all preprocess related interfaces tests


If developer wrap new interfaces from OpenVINO C++, you also need add the unite test case in the correct location.
Here is an example for C interface unite test case:
* C++ interface for read model,
```ruby
    /**
     * @brief Reads models from IR/ONNX/PDPD formats.
     * @param model_path Path to a model.
     * @param bin_path Path to a data file.
     * For IR format (*.bin):
     *  * if path is empty, will try to read a bin file with the same name as xml and
     *  * if the bin file with the same name is not found, will load IR without weights.
     * For ONNX format (*.onnx):
     *  * the bin_path parameter is not used.
     * For PDPD format (*.pdmodel)
     *  * the bin_path parameter is not used.
     * @return A model.
     */
    std::shared_ptr<ov::Model> read_model(const std::string& model_path, const std::string& bin_path = {}) const;
``` 

* C wrap this interface like,
```ruby
ov_status_e ov_core_read_model(const ov_core_t* core,
                               const char* model_path,
                               const char* bin_path,
                               ov_model_t** model) {
    if (!core || !model_path || !model) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        std::string bin = "";
        if (bin_path) {
            bin = bin_path;
        }
        std::unique_ptr<ov_model_t> _model(new ov_model_t);
        _model->object = core->object->read_model(model_path, bin);
        *model = _model.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}
```

* Create unite test case for this interface. At first, this interface is for core operation so the location should at [ov_core_test.cpp](../tests/ov_core_test.cpp). Also, the interface has default parameter so need to make unite test case for parameter missing. The final based function level test like:
```ruby
TEST(ov_core, ov_core_read_model) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, bin, &model));
    EXPECT_NE(nullptr, model);

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_core, ov_core_read_model_no_bin) {
    ov_core_t* core = nullptr;
    OV_EXPECT_OK(ov_core_create(&core));
    EXPECT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_EXPECT_OK(ov_core_read_model(core, xml, nullptr, &model));
    EXPECT_NE(nullptr, model);

    ov_model_free(model);
    ov_core_free(core);
}
```



