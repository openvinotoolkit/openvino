/*
* Copyright 2017-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

#ifndef MVSTRINGUTILS_H__
#define MVSTRINGUTILS_H__

#include <stdlib.h>

/*
 * Maximum sring length, 4KB.
 */
#define RSIZE_MAX_STR ( 4UL << 10 )

/**
 * @brief If the string utils functions have worked without errors, EOK will be
 *    returned. On error, one of error codes, which is not equal to zero, will
 *    be returned.
 */
typedef enum {
    EOK     = 0,  // Successful operation.
    ESNULLP = 1,  // NULL pointer.
    ESZEROL = 2,  // Zero length.
    ESLEMAX = 3,  // Length exceeds max limit.
    ESOVRLP = 4,  // Strings overlap.
    ESNOSPC = 5   // Not enough space to copy src
} mvStringUtilsError;

/**
 * @brief The mv_strcpy function copies the string pointed to by src
 *    (including the terminating null character) into the array
 *    pointed to by dest.
 * @param dest
 *    pointer to string that will be replaced by src.
 * @param destsz
 *    restricted maximum length of dest.
 * @param src
 *    pointer to the string that will be copied to dest
 * @return zero on success and non-zero value on error.
 */
 int mv_strcpy(char *dest, size_t destsz, const char *src);

/**
 * @brief The mv_strncpy function copies at most count characters from the
 *    string pointed to by src, including the terminating null byte ('\0'), to
 *    the array pointed to by dest. Exactly count characters are written at
 *    dest. If the length strlen_s(src) is greater than or equal to count, the
 *    string pointed to by dest will contain count characters from src plus a
 *    null characters (dest will be null-terminated). Therefore, destsz must
 *    be at least count+1 in order to contain the terminator.
 * @param dest
 *    pointer to string that will be replaced by src.
 * @param destsz
 *    restricted maximum length of dest (must be at least count+1).
 * @param src
 *    pointer to the string that will be copied to dest.
 * @param count
 *    the maximum number of characters from src to copy into dest.
 * @return zero on success and non-zero value on error.
 */
int mv_strncpy(char *dest, size_t destsz, const char *src, size_t count);

#endif  // MVSTRINGUTILS_H__
