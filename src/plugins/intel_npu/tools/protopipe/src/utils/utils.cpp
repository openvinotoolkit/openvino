//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <opencv2/gapi/own/assert.hpp>

#include <fstream>

namespace utils {

void createNDMat(cv::Mat& mat, const std::vector<int>& dims, int depth) {
    GAPI_Assert(!dims.empty());
    mat.create(dims, depth);
    if (dims.size() == 1) {
        // FIXME: Well-known 1D mat WA
        mat.dims = 1;
    }
}

void generateRandom(cv::Mat& out) {
    switch (out.depth()) {
    case CV_8U:
        cv::randu(out, 0, 255);
        break;
    case CV_32S:
        cv::randu(out, 0, 255);
        break;
    case CV_32F:
        cv::randu(out, 0.f, 255.f);
        break;
    case CV_16F: {
        std::vector<int> dims;
        for (int i = 0; i < out.size.dims(); ++i) {
            dims.push_back(out.size[i]);
        }
        cv::Mat fp32_mat;
        createNDMat(fp32_mat, dims, CV_32F);
        cv::randu(fp32_mat, 0.f, 255.f);
        fp32_mat.convertTo(out, out.type());
        break;
    }
    default:
        throw std::logic_error("Unsupported preprocessing depth");
    }
}

cv::Mat createRandom(const std::vector<int>& dims, int depth) {
    cv::Mat mat;
    createNDMat(mat, dims, depth);
    generateRandom(mat);
    return mat;
}

void readFromBinFile(const std::string& filepath, cv::Mat& mat) {
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);

    if (!ifs.is_open()) {
        throw std::logic_error("Failed to open: " + filepath);
    }

    const auto file_byte_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    const auto mat_byte_size = mat.total() * mat.elemSize();
    if (file_byte_size != mat_byte_size) {
        throw std::logic_error("Failed to read cv::Mat from binary file: " + filepath + ". Mat size: " +
                               std::to_string(mat_byte_size) + ", File size: " + std::to_string(file_byte_size));
    }

    ifs.read(mat.ptr<char>(), mat_byte_size);
}

void writeToBinFile(const std::string& filepath, const cv::Mat& mat) {
    std::ofstream fout(filepath, std::ios::out | std::ios::binary);
    if (!fout.is_open()) {
        throw std::logic_error("Failed to open/create: " + filepath);
    }
    fout.write(mat.ptr<const char>(), mat.total() * mat.elemSize());
}

}  // namespace utils
