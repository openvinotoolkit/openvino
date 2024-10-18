//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "data_providers.hpp"

#include <sstream>

#include "utils.hpp"
#include "utils/error.hpp"

UniformGenerator::UniformGenerator(double low, double high): m_low(low), m_high(high) {
    ASSERT(low <= high);
}

void UniformGenerator::generate(cv::Mat& mat) {
    cv::randu(mat, m_low, m_high);
}

std::string UniformGenerator::str() const {
    std::stringstream ss;
    ss << "{dist: uniform, range: [" << m_low << ", " << m_high << "]}";
    return ss.str();
}

RandomProvider::RandomProvider(IRandomGenerator::Ptr impl, const std::vector<int>& dims, const int depth)
        : m_impl(impl), m_dims(dims), m_depth(depth) {
}

void RandomProvider::pull(cv::Mat& mat) {
    utils::createNDMat(mat, m_dims, m_depth);
    m_impl->generate(mat);
}

cv::GMatDesc RandomProvider::desc() {
    if (m_dims.size() == 2u) {
        return cv::GMatDesc{m_depth, 1, cv::Size(m_dims[1], m_dims[0])};
    }
    return cv::GMatDesc{m_depth, m_dims};
}

CircleBuffer::CircleBuffer(const std::vector<cv::Mat>& buffer): m_buffer(buffer), m_pos(0u) {
    ASSERT(!m_buffer.empty());
}

CircleBuffer::CircleBuffer(std::vector<cv::Mat>&& buffer): m_buffer(std::move(buffer)), m_pos(0u) {
    ASSERT(!m_buffer.empty());
}

CircleBuffer::CircleBuffer(cv::Mat mat): CircleBuffer(std::vector<cv::Mat>{mat}) {
}

void CircleBuffer::pull(cv::Mat& mat) {
    m_buffer[m_pos++].copyTo(mat);
    if (m_pos == m_buffer.size()) {
        m_pos = 0;
    }
}

cv::GMatDesc CircleBuffer::desc() {
    return cv::descr_of(m_buffer[0]);
}
