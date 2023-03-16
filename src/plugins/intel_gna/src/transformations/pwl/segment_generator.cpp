// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "segment_generator.hpp"

#include "base_segment_precision_validator.hpp"
#include "common/numerical_utils.hpp"
#include "ie_common.h"

namespace ov {
namespace intel_gna {
using namespace common;
namespace pass {
namespace pwl {

SegmentsGeneratorImpl::SegmentsGeneratorImpl(const std::shared_ptr<ov::Node>& node,
                                             std::shared_ptr<ActivationRepresentation> representation)
    : m_node(node),
      m_representation(std::move(representation)) {}

std::vector<Segment> SegmentsGeneratorImpl::generate_segments() const {
    if (nullptr == m_representation) {
        return {};
    }

    auto precalculated_segments = m_representation->get_precalculated_segments();
    if (nullptr != precalculated_segments) {
        return precalculated_segments->get_precalculated_segments();
    }

    std::vector<Segment> total_pwls;
    auto data_for_calculation = m_representation->get_data_for_calculation();
    if (nullptr == data_for_calculation) {
        return {};
    }

    auto subintervals = data_for_calculation->get_subintervals();

    auto overall_lower_bound = std::numeric_limits<double>::infinity();
    auto overall_upper_bound = -std::numeric_limits<double>::infinity();

    for (const auto& subinterval : subintervals) {
        auto lower_bound = subinterval.get_lower_bound();
        auto upper_bound = subinterval.get_upper_bound();

        if (overall_lower_bound > lower_bound) {
            overall_lower_bound = lower_bound;
        }

        if (overall_upper_bound < upper_bound) {
            overall_upper_bound = upper_bound;
        }

        auto error = 0.0;
        std::vector<Segment> pwls;

        auto segments = kMinSegmentsNum;
        BaseSegmentPrecisionValidator validator(data_for_calculation->get_function(),
                                                data_for_calculation->get_allowed_error_percentage(),
                                                lower_bound,
                                                upper_bound);

        do {
            std::tie(error, pwls) = pivot_search(data_for_calculation->get_function(),
                                                 segments,
                                                 lower_bound,
                                                 upper_bound,
                                                 subinterval.is_negative(),
                                                 data_for_calculation->get_max_iterations());

        } while (!validator.is_valid(error) && ++segments <= kMaxSegmentsNum);

        IE_ASSERT(pwls.size() <= kMaxSegmentsNum)
            << "Failed to converge part of activation_function(" << m_representation->get_name()
            << ", lower_bound: " << lower_bound << ", upper_bound: " << upper_bound
            << ", with allowed error percentage: " << data_for_calculation->get_allowed_error_percentage() << ")!\n";

        if (subinterval.is_negative()) {
            negative_pwls(pwls);
        }

        // Remove last segment from previous subinterval in case inserting subsequent subinterval
        if (!total_pwls.empty()) {
            total_pwls.pop_back();
        }
        total_pwls.insert(total_pwls.end(), pwls.begin(), pwls.end());
    }

    if (!total_pwls.empty()) {
        auto inserter = data_for_calculation->get_surrounding_segments_inserter();
        inserter->insert_surrounding_segments(total_pwls);
    }

    IE_ASSERT(total_pwls.size() <= kMaxSegmentsOverall)
        << "Failed to converge whole activation_function(" << m_representation->get_name()
        << ", lower_bound: " << overall_lower_bound << ", upper_bound: " << overall_upper_bound
        << ", with allowed error percentage: " << data_for_calculation->get_allowed_error_percentage() << ")!\n";

    return total_pwls;
}

std::shared_ptr<ov::Node> SegmentsGeneratorImpl::get_node() const {
    return m_node;
}

std::pair<double, std::vector<Segment>> SegmentsGeneratorImpl::pivot_search(const Function& activation_function,
                                                                            uint32_t segments_num,
                                                                            double alpha_0,
                                                                            double alpha_N,
                                                                            bool negative,
                                                                            size_t max_iterations) const {
    std::vector<std::vector<double>> t(segments_num + 1);
    std::vector<std::vector<double>> alpha(segments_num + 1);
    std::vector<std::vector<double>> epsilon(segments_num + 1);
    std::vector<std::vector<double>> d(segments_num + 1);
    bool same_epsilon = false;
    double Delta;
    double epsilon_final = 0.0;
    double max_epsilon = 0.0;
    double max_epsilon_prev;
    double min_epsilon;
    double sgn = (negative) ? -1.0 : 1.0;
    int j;

    // Figure 4:  Box #1
    j = 0;
    Delta = 1.0;

    for (int i = 0; i < segments_num; i++) {
        t[i].push_back(alpha_0 +
                       (static_cast<double>((i + 1)) / static_cast<double>((segments_num + 1))) * (alpha_N - alpha_0));
    }

    while (true) {
        // Figure 4:  Box #2
        alpha[0].resize(j + 1);
        alpha[0][j] = alpha_0;
        for (int i = 1; i < segments_num; i++) {
            alpha[i].resize(j + 1);
            alpha[i][j] = (activation_function.get_value(t[i - 1][j]) - activation_function.get_value(t[i][j]) +
                           activation_function.get_first_derivative(t[i][j]) * t[i][j] -
                           activation_function.get_first_derivative(t[i - 1][j]) * t[i - 1][j]) /
                          (activation_function.get_first_derivative(t[i][j]) -
                           activation_function.get_first_derivative(t[i - 1][j]));
        }
        alpha[segments_num].resize(j + 1);
        alpha[segments_num][j] = alpha_N;

        // Figure 4:  Box #3
        for (int i = 0; i < segments_num; i++) {
            epsilon[i].resize(j + 1);
            epsilon[i][j] = sgn * (activation_function.get_first_derivative(t[i][j]) * (alpha[i][j] - t[i][j]) +
                                   activation_function.get_value(t[i][j]) - activation_function.get_value(alpha[i][j]));
            if (std::isnan(epsilon[i][j])) {
                throw std::runtime_error("The value is out of range.");
            }
        }
        epsilon[segments_num].resize(j + 1);
        epsilon[segments_num][j] = sgn * (activation_function.get_first_derivative(t[segments_num - 1][j]) *
                                              (alpha[segments_num][j] - t[segments_num - 1][j]) +
                                          activation_function.get_value(t[segments_num - 1][j]) -
                                          activation_function.get_value(alpha[segments_num][j]));
        if (std::isnan(epsilon[segments_num][j])) {
            throw std::runtime_error("The value is out of range.");
        }

        // Figure 4:  Test for completion
        max_epsilon_prev = max_epsilon;
        max_epsilon = std::fabs(epsilon[0][j]);
        min_epsilon = std::fabs(epsilon[0][j]);
        for (int i = 1; i < segments_num + 1; i++) {
            if (std::fabs(epsilon[i][j]) > max_epsilon)
                max_epsilon = std::fabs(epsilon[i][j]);
            if (std::fabs(epsilon[i][j]) < min_epsilon)
                min_epsilon = std::fabs(epsilon[i][j]);
        }
        if (j == max_iterations || max_epsilon - min_epsilon < kThreshold * min_epsilon) {
            Segment value;
            std::vector<Segment> pwls;

            epsilon_final = (max_epsilon + min_epsilon) / 4.0;  // Andrzej's modification
            for (int i = 0; i < segments_num; i++) {
                value.alpha = alpha[i][j];
                value.beta = sgn * activation_function.get_first_derivative(t[i][j]) * (value.alpha - t[i][j]) +
                             sgn * activation_function.get_value(t[i][j]) - epsilon_final;
                value.m = sgn * activation_function.get_first_derivative(t[i][j]);
                value.b = value.beta - value.m * value.alpha;
                pwls.push_back(value);
            }

            pwls.emplace_back(0,
                              0,
                              alpha[segments_num][j],
                              sgn * activation_function.get_first_derivative(t[segments_num - 1][j]) *
                                      (alpha[segments_num][j] - t[segments_num - 1][j]) +
                                  sgn * activation_function.get_value(t[segments_num - 1][j]) - epsilon_final);

            if (j == max_iterations) {
                throw std::runtime_error("Failed to converge in pivot_search!");
            }
            return {epsilon_final, std::move(pwls)};
        }

        if (j > 0) {
            if (max_epsilon > max_epsilon_prev) {
                j = j - 1;
                Delta = Delta / 2;
                same_epsilon = false;
            } else if (AreFpEq(max_epsilon, max_epsilon_prev)) {
                if (!same_epsilon) {
                    same_epsilon = true;
                } else {
                    j = j - 1;
                    Delta = Delta / 2;
                    same_epsilon = false;
                }
            }
        }

        // Figure 4:  Box #4
        for (int i = 0; i < segments_num; i++) {
            d[i].resize(j + 1);
            d[i][j] = Delta * (epsilon[i + 1][j] - epsilon[i][j]) /
                      ((epsilon[i + 1][j] / (alpha[i + 1][j] - t[i][j])) + (epsilon[i][j] / (t[i][j] - alpha[i][j])));
        }

        // Figure 4:  Box #5
        for (int i = 0; i < segments_num; i++) {
            t[i].resize(j + 2);
            t[i][j + 1] = t[i][j] + d[i][j];
        }
        t[segments_num].resize(j + 2);

        j = j + 1;
    }
}

void SegmentsGeneratorImpl::negative_pwls(std::vector<Segment>& data) const {
    for (auto& e : data) {
        e.m = -e.m;
        e.b = -e.b;
        e.beta = -e.beta;
    }
}

}  // namespace pwl
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov