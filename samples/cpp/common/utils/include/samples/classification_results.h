// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with output classification results
 * @file classification_results.h
 */
#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "openvino/openvino.hpp"

/**
 * @class ClassificationResult
 * @brief A ClassificationResult creates an output table with results
 */
class ClassificationResult {
private:
    const std::string _classidStr = "classid";
    const std::string _probabilityStr = "probability";
    const std::string _labelStr = "label";
    size_t _nTop;
    ov::Tensor _outTensor;
    const std::vector<std::string> _labels;
    const std::vector<std::string> _imageNames;
    const size_t _batchSize;
    std::vector<unsigned> _results;

    void printHeader() {
        std::cout << _classidStr << " " << _probabilityStr;
        if (!_labels.empty())
            std::cout << " " << _labelStr;
        std::string classidColumn(_classidStr.length(), '-');
        std::string probabilityColumn(_probabilityStr.length(), '-');
        std::string labelColumn(_labelStr.length(), '-');
        std::cout << std::endl << classidColumn << " " << probabilityColumn;
        if (!_labels.empty())
            std::cout << " " << labelColumn;
        std::cout << std::endl;
    }

    /**
     * @brief Gets the top n results from a tensor
     *
     * @param n Top n count
     * @param input 1D tensor that contains probabilities
     * @param output Vector of indexes for the top n places
     */
    template <class T>
    void topResults(unsigned int n, const ov::Tensor& input, std::vector<unsigned>& output) {
        ov::Shape shape = input.get_shape();
        size_t input_rank = shape.size();
        OPENVINO_ASSERT(input_rank != 0 && shape[0] != 0, "Input tensor has incorrect dimensions!");
        size_t batchSize = shape[0];
        std::vector<unsigned> indexes(input.get_size() / batchSize);

        n = static_cast<unsigned>(std::min<size_t>((size_t)n, input.get_size()));
        output.resize(n * batchSize);

        for (size_t i = 0; i < batchSize; i++) {
            const size_t offset = i * (input.get_size() / batchSize);
            const T* batchData = input.data<const T>();
            batchData += offset;

            std::iota(std::begin(indexes), std::end(indexes), 0);
            std::partial_sort(std::begin(indexes),
                              std::begin(indexes) + n,
                              std::end(indexes),
                              [&batchData](unsigned l, unsigned r) {
                                  return batchData[l] > batchData[r];
                              });
            for (unsigned j = 0; j < n; j++) {
                output.at(i * n + j) = indexes.at(j);
            }
        }
    }

    /**
     * @brief Gets the top n results from a blob
     *
     * @param n Top n count
     * @param input 1D blob that contains probabilities
     * @param output Vector of indexes for the top n places
     */
    void topResults(unsigned int n, const ov::Tensor& input, std::vector<unsigned>& output) {
#define TENSOR_TOP_RESULT(elem_type)                                                  \
    case ov::element::Type_t::elem_type: {                                            \
        using tensor_type = ov::fundamental_type_for<ov::element::Type_t::elem_type>; \
        topResults<tensor_type>(n, input, output);                                    \
        break;                                                                        \
    }

        switch (input.get_element_type()) {
            TENSOR_TOP_RESULT(f32);
            TENSOR_TOP_RESULT(f64);
            TENSOR_TOP_RESULT(f16);
            TENSOR_TOP_RESULT(i16);
            TENSOR_TOP_RESULT(u8);
            TENSOR_TOP_RESULT(i8);
            TENSOR_TOP_RESULT(u16);
            TENSOR_TOP_RESULT(i32);
            TENSOR_TOP_RESULT(u32);
            TENSOR_TOP_RESULT(i64);
            TENSOR_TOP_RESULT(u64);
        default:
            OPENVINO_ASSERT(false, "cannot locate tensor with element type: ", input.get_element_type());
        }

#undef TENSOR_TOP_RESULT
    }

public:
    explicit ClassificationResult(const ov::Tensor& output_tensor,
                                  const std::vector<std::string>& image_names = {},
                                  size_t batch_size = 1,
                                  size_t num_of_top = 10,
                                  const std::vector<std::string>& labels = {})
        : _nTop(num_of_top),
          _outTensor(output_tensor),
          _labels(labels),
          _imageNames(image_names),
          _batchSize(batch_size),
          _results() {
        OPENVINO_ASSERT(_imageNames.size() == _batchSize, "Batch size should be equal to the number of images.");

        topResults(_nTop, _outTensor, _results);
    }

    /**
     * @brief prints formatted classification results
     */
    void show() {
        /** Print the result iterating over each batch **/
        std::ios::fmtflags fmt(std::cout.flags());
        std::cout << std::endl << "Top " << _nTop << " results:" << std::endl << std::endl;
        for (size_t image_id = 0; image_id < _batchSize; ++image_id) {
            std::string out(_imageNames[image_id].begin(), _imageNames[image_id].end());
            std::cout << "Image " << out;
            std::cout.flush();
            std::cout.clear();
            std::cout << std::endl << std::endl;
            printHeader();

            for (size_t id = image_id * _nTop; id < (image_id + 1) * _nTop; ++id) {
                std::cout.precision(7);
                // Getting probability for resulting class
                const auto index = _results.at(id) + image_id * (_outTensor.get_size() / _batchSize);
                const auto result = _outTensor.data<const float>()[index];

                std::cout << std::setw(static_cast<int>(_classidStr.length())) << std::left << _results.at(id) << " ";
                std::cout << std::left << std::setw(static_cast<int>(_probabilityStr.length())) << std::fixed << result;

                if (!_labels.empty()) {
                    std::cout << " " + _labels[_results.at(id)];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout.flags(fmt);
    }

    void print() {
        /** Print the result iterating over each batch **/
        std::ios::fmtflags fmt(std::cout.flags());
        std::cout << std::endl << "Top " << _nTop << " results:" << std::endl << std::endl;
        for (size_t image_id = 0; image_id < _batchSize; ++image_id) {
            std::string out(_imageNames[image_id].begin(), _imageNames[image_id].end());
            std::cout << "Image " << out;
            std::cout.flush();
            std::cout.clear();
            std::cout << std::endl << std::endl;
            printHeader();

            for (size_t id = image_id * _nTop; id < (image_id + 1) * _nTop; ++id) {
                std::cout.precision(7);
                // Getting probability for resulting class
                const auto result = _outTensor.data<float>();
                std::cout << std::setw(static_cast<int>(_classidStr.length())) << std::left << _results.at(id) << " ";
                std::cout << std::left << std::setw(static_cast<int>(_probabilityStr.length())) << std::fixed << result;

                if (!_labels.empty()) {
                    std::cout << " " + _labels[_results.at(id)];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout.flags(fmt);
    }

    /**
     * @brief returns the classification results in a vector
     */
    std::vector<unsigned> getResults() {
        return _results;
    }
};
