// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with output classification results
 * @file classification_results.h
 */
#pragma once

#include <algorithm>
#include <inference_engine.hpp>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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
     * @brief Gets the top n results from a tblob
     *
     * @param n Top n count
     * @param input 1D tblob that contains probabilities
     * @param output Vector of indexes for the top n places
     */
    template <class T>
    void topResults(unsigned int n, InferenceEngine::TBlob<T>& input, std::vector<unsigned>& output) {
        InferenceEngine::SizeVector dims = input.getTensorDesc().getDims();
        size_t input_rank = dims.size();
        if (!input_rank || !dims[0])
            IE_THROW() << "Input blob has incorrect dimensions!";
        size_t batchSize = dims[0];
        std::vector<unsigned> indexes(input.size() / batchSize);

        n = static_cast<unsigned>(std::min<size_t>((size_t)n, input.size()));

        output.resize(n * batchSize);

        for (size_t i = 0; i < batchSize; i++) {
            size_t offset = i * (input.size() / batchSize);
            T* batchData = input.data();
            batchData += offset;

            std::iota(std::begin(indexes), std::end(indexes), 0);
            std::partial_sort(std::begin(indexes), std::begin(indexes) + n, std::end(indexes), [&batchData](unsigned l, unsigned r) {
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
    void topResults(unsigned int n, InferenceEngine::Blob& input, std::vector<unsigned>& output) {
#define TBLOB_TOP_RESULT(precision)                                                                            \
    case InferenceEngine::Precision::precision: {                                                              \
        using myBlobType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::precision>::value_type; \
        InferenceEngine::TBlob<myBlobType>& tblob = dynamic_cast<InferenceEngine::TBlob<myBlobType>&>(input);  \
        topResults(n, tblob, output);                                                                          \
        break;                                                                                                 \
    }

        switch (input.getTensorDesc().getPrecision()) {
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
            IE_THROW() << "cannot locate blob for precision: " << input.getTensorDesc().getPrecision();
        }

#undef TBLOB_TOP_RESULT
    }

public:
    explicit ClassificationResultT(InferenceEngine::Blob::Ptr output_blob, std::vector<strType> image_names = {}, size_t batch_size = 1, size_t num_of_top = 10,
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
        topResults(_nTop, *_outBlob, _results);
    }

    /**
     * @brief prints formatted classification results
     */
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

            for (size_t id = image_id * _nTop, cnt = 0; id < (image_id + 1) * _nTop; ++cnt, ++id) {
                std::cout.precision(7);
                /** Getting probability for resulting class **/
                InferenceEngine::MemoryBlob::CPtr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(_outBlob);
                if (!moutput) {
                    throw std::logic_error("We expect _outBlob to be inherited from MemoryBlob in "
                                           "ClassificationResult::print, "
                                           "but by fact we were not able to cast _outBlob to MemoryBlob");
                }
                // locked memory holder should be alive all time while access to its buffer happens
                auto moutputHolder = moutput->rmap();

                const auto result =
                    moutputHolder
                        .as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>()[_results.at(id) +
                                                                                                                    image_id * (_outBlob->size() / _batchSize)];

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
