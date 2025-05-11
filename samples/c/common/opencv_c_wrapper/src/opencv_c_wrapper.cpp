// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "opencv_c_wrapper.h"

extern "C" {
#include "bmp_reader.h"
}

#ifndef USE_OPENCV

int image_read(const char* img_path, c_mat_t* img) {
    BitMap bmp;
    bmp.data = NULL;
    int retCode = readBmpImage(img_path, &bmp);

    img->mat_data = bmp.data;
    img->mat_data_size = bmp.width * bmp.height * 3;
    img->mat_width = bmp.width;
    img->mat_height = bmp.height;
    img->mat_channels = 3;
    img->mat_type = bmp.header.type;

    return retCode;
}
int image_resize(const c_mat_t* src_img, c_mat_t* dst_img, const int width, const int height) {
    return -1;
}
int image_save(const char* img_path, c_mat_t* img) {
    return -1;
}
int image_free(c_mat_t* img) {
    delete img->mat_data;
    return 0;
}
int image_add_rectangles(c_mat_t* img, rectangle_t rects[], int classes[], int num, int thickness) {
    return -1;
}

#else

#    include <algorithm>
#    include <opencv2/opencv.hpp>

int image_read(const char* img_path, c_mat_t* img) {
    if (img_path == nullptr || img == nullptr) {
        return -1;
    }

    cv::Mat mat = cv::imread(img_path);
    if (mat.data == NULL) {
        return -1;
    }

    img->mat_channels = mat.channels();
    img->mat_width = mat.size().width;
    img->mat_height = mat.size().height;
    img->mat_type = mat.type();
    img->mat_data_size = mat.elemSize() * img->mat_width * img->mat_height;
    img->mat_data = (unsigned char*)malloc(sizeof(unsigned char) * img->mat_data_size);

    if (img->mat_data == NULL) {
        return -1;
    }

    for (int i = 0; i < img->mat_data_size; ++i) {
        img->mat_data[i] = mat.data[i];
    }

    return 0;
}

int image_resize(const c_mat_t* src_img, c_mat_t* dst_img, const int width, const int height) {
    if (src_img == nullptr || dst_img == nullptr) {
        return -1;
    }

    cv::Mat mat_src(cv::Size(src_img->mat_width, src_img->mat_height), src_img->mat_type, src_img->mat_data);

    cv::Mat mat_dst;
    cv::resize(mat_src, mat_dst, cv::Size(width, height));
    if (mat_dst.data) {
        dst_img->mat_channels = mat_dst.channels();
        dst_img->mat_width = mat_dst.size().width;
        dst_img->mat_height = mat_dst.size().height;
        dst_img->mat_type = mat_dst.type();
        dst_img->mat_data_size = mat_dst.elemSize() * dst_img->mat_width * dst_img->mat_height;
        dst_img->mat_data = (unsigned char*)malloc(sizeof(unsigned char) * dst_img->mat_data_size);

        if (dst_img->mat_data == NULL) {
            return -1;
        }

        for (int i = 0; i < dst_img->mat_data_size; ++i) {
            dst_img->mat_data[i] = mat_dst.data[i];
        }
    } else {
        return -1;
    }

    return 1;
}

int image_save(const char* img_path, c_mat_t* img) {
    cv::Mat mat(cv::Size(img->mat_width, img->mat_height), img->mat_type, img->mat_data);
    return cv::imwrite(img_path, mat);
}

int image_free(c_mat_t* img) {
    if (img) {
        free(img->mat_data);
        img->mat_data = NULL;
    }
    return -1;
}

int image_add_rectangles(c_mat_t* img, rectangle_t rects[], int classes[], int num, int thickness) {
    int colors_num = 21;
    color_t colors[21] = {// colors to be used for bounding boxes
                          {128, 64, 128},  {232, 35, 244}, {70, 70, 70},  {156, 102, 102}, {153, 153, 190},
                          {153, 153, 153}, {30, 170, 250}, {0, 220, 220}, {35, 142, 107},  {152, 251, 152},
                          {180, 130, 70},  {60, 20, 220},  {0, 0, 255},   {142, 0, 0},     {70, 0, 0},
                          {100, 60, 0},    {90, 0, 0},     {230, 0, 0},   {32, 11, 119},   {0, 74, 111},
                          {81, 0, 81}};

    for (int i = 0; i < num; i++) {
        int x = rects[i].x_min;
        int y = rects[i].y_min;
        int w = rects[i].rect_width;
        int h = rects[i].rect_height;

        int cls = classes[i] % colors_num;  // color of a bounding box line

        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (w < 0)
            w = 0;
        if (h < 0)
            h = 0;

        if (x >= img->mat_width) {
            x = img->mat_width - 1;
            w = 0;
            thickness = 1;
        }
        if (y >= img->mat_height) {
            y = img->mat_height - 1;
            h = 0;
            thickness = 1;
        }

        if ((x + w) >= img->mat_width) {
            w = img->mat_width - x - 1;
        }
        if ((y + h) >= img->mat_height) {
            h = img->mat_height - y - 1;
        }

        thickness = std::min(std::min(thickness, w / 2 + 1), h / 2 + 1);

        size_t shift_first;
        size_t shift_second;
        for (int t = 0; t < thickness; t++) {
            shift_first = (y + t) * img->mat_width * 3;
            shift_second = (y + h - t) * img->mat_width * 3;
            for (int ii = x; ii < x + w + 1; ii++) {
                img->mat_data[shift_first + ii * 3] = colors[cls].r;
                img->mat_data[shift_first + ii * 3 + 1] = colors[cls].g;
                img->mat_data[shift_first + ii * 3 + 2] = colors[cls].b;
                img->mat_data[shift_second + ii * 3] = colors[cls].r;
                img->mat_data[shift_second + ii * 3 + 1] = colors[cls].g;
                img->mat_data[shift_second + ii * 3 + 2] = colors[cls].b;
            }
        }

        for (int t = 0; t < thickness; t++) {
            shift_first = (x + t) * 3;
            shift_second = (x + w - t) * 3;
            for (int ii = y; ii < y + h + 1; ii++) {
                img->mat_data[shift_first + ii * img->mat_width * 3] = colors[cls].r;
                img->mat_data[shift_first + ii * img->mat_width * 3 + 1] = colors[cls].g;
                img->mat_data[shift_first + ii * img->mat_width * 3 + 2] = colors[cls].b;
                img->mat_data[shift_second + ii * img->mat_width * 3] = colors[cls].r;
                img->mat_data[shift_second + ii * img->mat_width * 3 + 1] = colors[cls].g;
                img->mat_data[shift_second + ii * img->mat_width * 3 + 2] = colors[cls].b;
            }
        }
    }
    return 0;
}

#endif  // USE_OPENCV
