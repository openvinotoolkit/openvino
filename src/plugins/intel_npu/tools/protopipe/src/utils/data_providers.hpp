//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <opencv2/gapi/gmat.hpp>

struct IDataProvider {
    using Ptr = std::shared_ptr<IDataProvider>;
    virtual void pull(cv::Mat& mat) = 0;
    virtual cv::GMatDesc desc() = 0;
    virtual void reset() = 0;
    virtual ~IDataProvider() = default;
};

class IRandomGenerator {
public:
    using Ptr = std::shared_ptr<IRandomGenerator>;
    virtual void generate(cv::Mat& mat) = 0;
    virtual ~IRandomGenerator() = default;
    virtual std::string str() const = 0;
};

class UniformGenerator : public IRandomGenerator {
public:
    using Ptr = std::shared_ptr<UniformGenerator>;
    UniformGenerator(double low, double high);
    void generate(cv::Mat& mat) override;
    virtual std::string str() const override;

private:
    double m_low, m_high;
};

class RandomProvider : public IDataProvider {
public:
    RandomProvider(IRandomGenerator::Ptr impl, const std::vector<int>& dims, const int depth);

    void pull(cv::Mat& mat) override;
    cv::GMatDesc desc() override;
    void reset() override { /* do nothing */
    }

private:
    IRandomGenerator::Ptr m_impl;
    std::vector<int> m_dims;
    int m_depth;
};

class CircleBuffer : public IDataProvider {
public:
    CircleBuffer(const std::vector<cv::Mat>& buffer);
    CircleBuffer(std::vector<cv::Mat>&& buffer);
    CircleBuffer(cv::Mat mat);

    void pull(cv::Mat& mat) override;
    cv::GMatDesc desc() override;
    void reset() override {
        m_pos = 0;
    }

private:
    std::vector<cv::Mat> m_buffer;
    uint64_t m_pos;
};
