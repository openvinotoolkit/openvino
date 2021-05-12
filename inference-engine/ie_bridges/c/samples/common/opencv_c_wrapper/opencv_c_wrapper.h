// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
    #define OPENCV_C_EXTERN extern "C"
#else
    #define OPENCV_C_EXTERN
#endif

#if defined(__GNUC__) && (__GNUC__ < 4)
    #define OPENCV_C_WRAPPER(...) OPENCV_C_EXTERN __VA_ARGS__
#else
    #if defined(_WIN32)
        #ifdef opencv_c_wrapper_EXPORTS
            #define OPENCV_C_WRAPPER(...) OPENCV_C_EXTERN __declspec(dllexport) __VA_ARGS__ __cdecl
        #else
            #define OPENCV_C_WRAPPER(...) OPENCV_C_EXTERN __declspec(dllimport) __VA_ARGS__ __cdecl
        #endif
    #else
        #define OPENCV_C_WRAPPER(...) OPENCV_C_EXTERN __attribute__((visibility("default"))) __VA_ARGS__
    #endif
#endif

/**
 * @struct c_mat
 * @brief OpenCV Mat Wrapper
 */
typedef struct c_mat {
    unsigned char* mat_data;
    int mat_data_size;
    int mat_width;
    int mat_height;
    int mat_channels;
    int mat_type;
} c_mat_t;

/**
 * @struct rectangle
 * @brief This structure describes rectangle data.
 */
typedef struct rectangle {
    int x_min;
    int y_min;
    int rect_width;
    int rect_height;
} rectangle_t;

/**
 * @struct color
 * @brief  Stores channels of a given color
 */
typedef struct color {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} color_t;

/**
 * @brief Load an image from a file. If the image cannot be read, the function return -1.
 * @param img_path Path of file to be loaded.
 * @param img A pointer to the newly created c_mat_t.
 * @return Status of the operation: 0 for success, -1 for fail.
 */
OPENCV_C_WRAPPER(int) image_read(const char* img_path, c_mat_t* img);

/**
 * @brief  Resizes an image.
 * @param src_img A pointer to the input image.
 * @param dst_img A pointer to the output image.
 * @param width The width of dst_img.
 * @param height The height of dst_img.
 * @return Status of the operation: 0 for success, -1 for fail.
 */
OPENCV_C_WRAPPER(int)
image_resize(const c_mat_t* src_img, c_mat_t* dst_img, const int width, const int height);

/**
 * @brief Saves an image to a specified file.The image format is chosen based on the filename
 * extension.
 * @param img_path Path of the file to be saved.
 * @param img Image to be saved.
 * @return Status of the operation: 0 for success, -1 for fail.
 */
OPENCV_C_WRAPPER(int) image_save(const char* img_path, c_mat_t* img);

/**
 * @brief Releases memory occupied by a c_mat_t instance.
 * @param img A pointer to the c_mat_t instance to free memory.
 * @return Status of the operation: 0 for success, -1 for fail.
 */
OPENCV_C_WRAPPER(int) image_free(c_mat_t* img);

/**
 * @brief Adds colored rectangles to the image
 * @param img - image where rectangles are put
 * @param rects - array for the rectangle
 * @param classes - array for classes
 * @param num - number of the rects and classes
 * @param thickness - thickness of a line (in pixels) to be used for bounding boxes
 * @return Status of the operation: 0 for success, -1 for fail.
 */
OPENCV_C_WRAPPER(int)
image_add_rectangles(c_mat_t* img, rectangle_t rects[], int classes[], int num, int thickness);
