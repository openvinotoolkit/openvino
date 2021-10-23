// Copyright (C) 2018-2021 Intel Corporation
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

#include "inference_engine.hpp"
#include "openvino/openvino.hpp"

/**
 * @class ClassificationResult
 * @brief A ClassificationResult creates an output table with results
 */
template <class strType = std::string>
class ClassificationResultT {
private:
    const std::string _classidStr = "classid";
    const std::string _probabilityStr = "probability";
    const std::string _labelStr = "label";
    size_t _nTop;
    ov::runtime::Tensor _outTensor;
    InferenceEngine::Blob::Ptr _outBlob;
    const std::vector<std::string> _labels;
    const std::vector<strType> _imageNames;
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
    void topResults(unsigned int n, const ov::runtime::Tensor& input, std::vector<unsigned>& output) {
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

    template <class T>
    void topResults(unsigned int n, InferenceEngine::Blob::Ptr& input, std::vector<unsigned>& output) {
        InferenceEngine::SizeVector dims = input->getTensorDesc().getDims();
        size_t input_rank = dims.size();
        if (!input_rank || !dims[0])
            IE_THROW() << "Input blob has incorrect dimensions!";
        size_t batchSize = dims[0];
        std::vector<unsigned> indexes(input->size() / batchSize);

        n = static_cast<unsigned>(std::min<size_t>((size_t)n, input->size()));

        output.resize(n * batchSize);
        InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(input);
        if (!moutput) {
            IE_THROW() << "Output blob should be inherited from MemoryBlob";
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto moutputHolder = moutput->rmap();

        for (size_t i = 0; i < batchSize; i++) {
            size_t offset = i * (input->size() / batchSize);
            T* batchData = moutputHolder.as<T*>();
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
    void topResults(unsigned int n, const ov::runtime::Tensor& input, std::vector<unsigned>& output) {
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

    void topResults(unsigned int n, InferenceEngine::Blob::Ptr& input, std::vector<unsigned>& output) {
#define TBLOB_TOP_RESULT(precision)                                                                            \
    case InferenceEngine::Precision::precision: {                                                              \
        using myBlobType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::precision>::value_type; \
        topResults<myBlobType>(n, input, output);                                                              \
        break;                                                                                                 \
    }

        switch (input->getTensorDesc().getPrecision()) {
            TBLOB_TOP_RESULT(FP32);
            TBLOB_TOP_RESULT(FP64);
            TBLOB_TOP_RESULT(FP16);
            TBLOB_TOP_RESULT(Q78);
            TBLOB_TOP_RESULT(I16);
            TBLOB_TOP_RESULT(U8);
            TBLOB_TOP_RESULT(I8);
            TBLOB_TOP_RESULT(U16);
            TBLOB_TOP_RESULT(I32);
            TBLOB_TOP_RESULT(U32);
            TBLOB_TOP_RESULT(U64);
            TBLOB_TOP_RESULT(I64);
        default:
            IE_THROW() << "cannot locate blob for precision: " << input->getTensorDesc().getPrecision();
        }

#undef TBLOB_TOP_RESULT
    }

public:
    explicit ClassificationResultT(InferenceEngine::Blob::Ptr output_blob,
                                   std::vector<strType> image_names = {},
                                   size_t batch_size = 1,
                                   size_t num_of_top = 10,
                                   std::vector<std::string> labels = {})
        : _nTop(num_of_top),
          _outBlob(std::move(output_blob)),
          _labels(std::move(labels)),
          _imageNames(std::move(image_names)),
          _batchSize(batch_size),
          _results() {
        if (_imageNames.size() != _batchSize) {
            throw std::logic_error("Batch size should be equal to the number of images.");
        }
        topResults(_nTop, _outBlob, _results);
    }

    explicit ClassificationResultT(const ov::runtime::Tensor& output_tensor,
                                   const std::vector<strType>& image_names = {},
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
        std::cout << std::endl << "Top " << _nTop << " results:" << std::endl << std::endl;
        for (unsigned int image_id = 0; image_id < _batchSize; ++image_id) {
            std::wstring out(_imageNames[image_id].begin(), _imageNames[image_id].end());
            std::wcout << L"Image " << out;
            std::wcout.flush();
            std::wcout.clear();
            std::wcout << std::endl << std::endl;
            printHeader();

            for (size_t id = image_id * _nTop, cnt = 0; id < (image_id + 1) * _nTop; ++cnt, ++id) {
                std::cout.precision(7);
                /** Getting probability for resulting class **/
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
    }

    void print() {
        /** Print the result iterating over each batch **/
        std::cout << std::endl << "Top " << _nTop << " results:" << std::endl << std::endl;
        for (unsigned int image_id = 0; image_id < _batchSize; ++image_id) {
            std::wstring out(_imageNames[image_id].begin(), _imageNames[image_id].end());
            std::wcout << L"Image " << out;
            std::wcout.flush();
            std::wcout.clear();
            std::wcout << std::endl << std::endl;
            printHeader();

            InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(_outBlob);
            auto moutputHolder = moutput->rmap();
            for (size_t id = image_id * _nTop, cnt = 0; id < (image_id + 1) * _nTop; ++cnt, ++id) {
                std::cout.precision(7);
                /** Getting probability for resulting class **/
                if (!moutput) {
                    throw std::logic_error("We expect _outBlob to be inherited from MemoryBlob in "
                                           "ClassificationResult::print, "
                                           "but by fact we were not able to cast _outBlob to MemoryBlob");
                }
                // locked memory holder should be alive all time while access to its buffer happens
                const auto result =
                    moutputHolder
                        .as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>()
                            [_results.at(id) + image_id * (_outBlob->size() / _batchSize)];

                std::cout << std::setw(static_cast<int>(_classidStr.length())) << std::left << _results.at(id) << " ";
                std::cout << std::left << std::setw(static_cast<int>(_probabilityStr.length())) << std::fixed << result;

                if (!_labels.empty()) {
                    std::cout << " " + _labels[_results.at(id)];
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    /**
     * @brief returns the classification results in a vector
     */
    std::vector<unsigned> getResults() {
        return _results;
    }
};

using ClassificationResult = ClassificationResultT<>;
using ClassificationResultW = ClassificationResultT<std::wstring>;
