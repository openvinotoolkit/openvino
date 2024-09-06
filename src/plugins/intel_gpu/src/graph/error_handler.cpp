// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "intel_gpu/runtime/error_handler.hpp"
#include <string>
#include <vector>

namespace cldnn {

void err_details::cldnn_print_error_message(
#ifndef NDEBUG
                                            const std::string& file, int line,
#else
                                            const std::string&, int,
#endif
                                            const std::string& instance_id,
                                            std::stringstream& msg,
                                            const std::string& add_msg) {
    {
        std::stringstream source_of_error;

#ifndef NDEBUG
        source_of_error << file << " at line: " << line << std::endl;
#endif
        source_of_error << "Error has occured for: " << instance_id << std::endl;

        std::stringstream addidtional_message;
        if (!add_msg.empty()) {
            addidtional_message << add_msg << std::endl;
        }

        throw std::invalid_argument(source_of_error.str() + msg.str() + addidtional_message.str());
    }
}

void error_message(const std::string& file, int line, const std::string& instance_id, const std::string& message) {
    std::stringstream error_msg;
    error_msg << message << std::endl;
    err_details::cldnn_print_error_message(file, line, instance_id, error_msg);
}

void error_on_not_supported_fp16(const std::string& file,
                                 int line,
                                 const std::string& instance_id,
                                 uint8_t supp_fp16,
                                 bool fp16_used) {
    if (!supp_fp16 && fp16_used) {
        std::stringstream error_msg;
        error_msg << "GPU device does not support half precision floating-point formats (cl_khr_fp16 extension)"
                  << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg);
    }
}

void error_on_bool(const std::string& file,
                   int line,
                   const std::string& instance_id,
                   const std::string& condition_id,
                   bool condition,
                   const std::string& additional_message) {
    if (condition) {
        std::stringstream error_msg;
        auto condition_to_string = [](const bool& condi) -> std::string { return condi ? "true" : "false"; };
        error_msg << condition_id << "(" << condition_to_string(condition) << ") should be "
                  << condition_to_string(!condition) << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_mismatching_data_types(const std::string& file,
                                     int line,
                                     const std::string& instance_id,
                                     const std::string& data_format_1_id,
                                     data_types data_format_1,
                                     const std::string& data_format_2_id,
                                     data_types data_format_2,
                                     const std::string& additional_message,
                                     bool ignore_sign) {
    if (data_format_1 != data_format_2 && !ignore_sign &&
        ((data_format_1 == data_types::i8 && data_format_2 == data_types::u8) ||
         (data_format_1 == data_types::u8 && data_format_2 == data_types::i8))) {
        std::stringstream error_msg;
        error_msg << "Data formats are incompatible." << std::endl;
        error_msg << data_format_1_id << " format is: " << ov::element::Type(data_format_1) << ", "
                  << data_format_2_id << " is: " << ov::element::Type(data_format_2) << std::endl;
        error_msg << "Data formats should be the same!" << std::endl;
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_tensor_dims_less_than_other_tensor_dims(const std::string& file,
                                                      int line,
                                                      const std::string& instance_id,
                                                      const std::string& tensor_id,
                                                      const tensor& tens,
                                                      const std::string& tensor_to_compare_to_id,
                                                      const tensor& tens_to_compre,
                                                      const std::string& additional_message) {
    std::vector<std::string> errors;
    if (tens.batch[0] < tens_to_compre.batch[0]) {
        errors.push_back("Batch");
    }
    if (tens.feature[0] < tens_to_compre.feature[0]) {
        errors.push_back("Feature");
    }
    if (tens.spatial[0] < tens_to_compre.spatial[0]) {
        errors.push_back("Spatial x");
    }
    if (tens.spatial[1] < tens_to_compre.spatial[1]) {
        errors.push_back("Spatial y");
    }

    if (!errors.empty()) {
        std::stringstream error_msg;
        error_msg << tensor_id << " sizes: " << tens << std::endl;
        error_msg << tensor_to_compare_to_id << " sizes: " << tens_to_compre << std::endl;
        error_msg << "All " << tensor_id << " dimensions should not be less than " << tensor_to_compare_to_id
                  << " dimensions." << std::endl;
        error_msg << "Mismatching dimensions: ";
        for (size_t i = 0; i < errors.size(); i++) {
            error_msg << errors.at(i) << std::endl;
        }
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_tensor_dims_greater_than_other_tensor_dims(const std::string& file,
                                                         int line,
                                                         const std::string& instance_id,
                                                         const std::string& tensor_id,
                                                         const tensor& tens,
                                                         const std::string& tensor_to_compare_to_id,
                                                         const tensor& tens_to_compre,
                                                         const std::string& additional_message) {
    std::vector<std::string> errors;
    if (tens.batch[0] > tens_to_compre.batch[0]) {
        errors.push_back("Batch");
    }
    if (tens.feature[0] > tens_to_compre.feature[0]) {
        errors.push_back("Feature");
    }
    if (tens.spatial[0] > tens_to_compre.spatial[0]) {
        errors.push_back("Spatial x");
    }
    if (tens.spatial[1] > tens_to_compre.spatial[1]) {
        errors.push_back("Spatial y");
    }

    if (!errors.empty()) {
        std::stringstream error_msg;
        error_msg << tensor_id << " sizes: " << tens << std::endl;
        error_msg << tensor_to_compare_to_id << " sizes: " << tens_to_compre << std::endl;
        error_msg << "All " << tensor_id << " dimensions should not be greater than " << tensor_to_compare_to_id
                  << std::endl;
        error_msg << "Mismatching dimensions: ";
        for (size_t i = 0; i < errors.size(); i++) {
            error_msg << errors.at(i) << std::endl;
        }
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_tensor_dims_not_dividable_by_other_tensor_dims(const std::string& file,
                                                             int line,
                                                             const std::string& instance_id,
                                                             const std::string& tensor_id,
                                                             const tensor& tens,
                                                             const std::string& tensor_to_compare_to_id,
                                                             const tensor& tens_to_compre,
                                                             const std::string& additional_message) {
    std::vector<std::string> errors;
    if (tens.batch[0] % tens_to_compre.batch[0] != 0) {
        errors.push_back("Batch");
    }
    if (tens.feature[0] % tens_to_compre.feature[0] != 0) {
        errors.push_back("Feature");
    }
    if (tens.spatial[0] % tens_to_compre.spatial[0] != 0) {
        errors.push_back("Spatial x");
    }
    if (tens.spatial[1] % tens_to_compre.spatial[1] != 0) {
        errors.push_back("Spatial y");
    }

    if (!errors.empty()) {
        std::stringstream error_msg;
        error_msg << tensor_id << " sizes: " << tens << std::endl;
        error_msg << tensor_to_compare_to_id << " sizes: " << tens_to_compre << std::endl;
        error_msg << "All " << tensor_id << " dimensions must be dividable by corresponding dimensions from "
                  << tensor_to_compare_to_id << std::endl;
        error_msg << "Mismatching dimensions: ";
        for (size_t i = 0; i < errors.size(); i++) {
            error_msg << errors.at(i) << std::endl;
        }
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

void error_on_mismatch_layout(const std::string& file,
                              int line,
                              const std::string& instance_id,
                              const std::string& layout_1_id,
                              const layout& layout_1,
                              const std::string& layout_2_id,
                              const layout& layout_2,
                              const std::string& additional_message) {
    if (layout_1 != layout_2) {
        std::stringstream error_msg;
        error_msg << "Layouts mismatch." << std::endl;

        if (layout_1.data_padding != layout_2.data_padding) {
            error_msg << layout_1_id << " data padding mismatch: " << layout_2_id << " data padding." << std::endl;
        }
        if (layout_1.data_type != layout_2.data_type) {
            error_msg << layout_1_id << " data type mismatch: " << layout_2_id << " data type." << std::endl;
            error_msg << layout_1_id << " data type: " << ov::element::Type(layout_1.data_type) << ", "
                      << layout_2_id << " data type: " << ov::element::Type(layout_2.data_type) << std::endl;
        }
        if (layout_1.format != layout_2.format) {
            error_msg << layout_1_id << " format mismatch: " << layout_2_id << " format." << std::endl;
            error_msg << layout_1_id << " format: " << format::traits(layout_1.format).order << ", " << layout_2_id
                      << " format: " << format::traits(layout_2.format).order << std::endl;
        }
        if (layout_1.get_tensor() != layout_2.get_tensor()) {
            error_msg << layout_1_id << " size mismatch : " << layout_2_id << " size." << std::endl;
            error_msg << layout_1_id << " size: " << layout_1.get_tensor() << ", " << layout_2_id << " size: " << layout_2.get_tensor()
                      << std::endl;
        }
        err_details::cldnn_print_error_message(file, line, instance_id, error_msg, additional_message);
    }
}

}  // namespace cldnn
