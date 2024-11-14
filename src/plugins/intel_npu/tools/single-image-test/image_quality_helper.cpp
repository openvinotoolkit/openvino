//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "image_quality_helper.hpp"

#include <cmath>
#include <iostream>

#include "data_type_converters.hpp"


float utils::runPSNRMetric(std::vector<std::vector<float>>& actOutput,
                           std::vector<std::vector<float>>& refOutput,
                           const size_t imgHeight,
                           const size_t imgWidth,
                           int scaleBorder,
                           bool normalizedImage) {
    size_t colorScale;
    float imageDiff;
    float sum = 0;

    if (!normalizedImage) {
        colorScale = 255;
    } else {
        colorScale = 1;
    }

    for (size_t iout = 0; iout < actOutput.size(); ++iout) {
        for (size_t h = scaleBorder; h < imgHeight - scaleBorder; h++) {
            for (size_t w = scaleBorder; w < imgWidth - scaleBorder; w++) {
                imageDiff = ((actOutput[iout][h * imgWidth + w] - refOutput[iout][h * imgWidth + w]) /
                             npu::utils::convertValuePrecision<float>(colorScale));

                sum = sum + (imageDiff * imageDiff);
            }
        }
    }

    auto mse = sum / (imgWidth * imgHeight);
    auto psnr = -10 * log10(mse);

    std::cout << "psnr: " << psnr << " Db" << std::endl;

    return psnr;
}
