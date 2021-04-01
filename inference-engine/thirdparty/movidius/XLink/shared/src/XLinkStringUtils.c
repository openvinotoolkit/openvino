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

#include "XLinkStringUtils.h"

int mv_strcpy(char *dest, size_t destsz, const char *src) {
   const char *overlap_bumper;

   if (dest == NULL) {
       return ESNULLP;
   }
   if (destsz == 0) {
       return ESZEROL;
   }
   if (destsz > RSIZE_MAX_STR) {
       return ESLEMAX;
   }
   if (src == NULL) {
       while (destsz) {
           *dest = '\0';
           destsz--;
           dest++;
       }
       return ESNULLP;
   }
   // It’s ok if src and dest are in the same place. No action is needed.
   if (dest == src) {
       return EOK;
   }

   if (dest < src) {
       overlap_bumper = src;

       while (destsz > 0) {
           if (dest == overlap_bumper) {
               return ESOVRLP;
           }

           *dest = *src;
           if (*dest == '\0') {
               while (destsz) {
                   *dest = '\0';
                   destsz--;
                   dest++;
               }
               return EOK;
           }
           destsz--;
           dest++;
           src++;
       }
   } else {
       overlap_bumper = dest;

       while (destsz > 0) {
           if (src == overlap_bumper) {
               return ESOVRLP;
           }

           *dest = *src;
           if (*dest == '\0') {
               while (destsz) {
                   *dest = '\0';
                   destsz--;
                   dest++;
               }
               return EOK;
           }
           destsz--;
           dest++;
           src++;
       }
   }

   // Ran out of space in dest, and did not find the null terminator in src.
   return ESNOSPC;
}

int mv_strncpy(char *dest, size_t destsz, const char *src, size_t count) {
    if (dest == NULL) {
        return ESNULLP;
    }
    if (src == NULL) {
        while (destsz) {
            *dest = '\0';
            destsz--;
            dest++;
        }
        return ESNULLP;
    }
    if (destsz == 0) {
        return ESZEROL;
    }
    if (destsz > RSIZE_MAX_STR || count > RSIZE_MAX_STR) {
        return ESLEMAX;
    }
    if (destsz < (count + 1)) {
        dest[0] = '\0';
        return ESNOSPC;
    }
    if (((src  < dest) && ((src + destsz) >= dest)) ||
            ((dest < src)  && ((dest + destsz) >= src))) {
        dest[0] = '\0';
        return ESOVRLP;
    }

    if (dest == src) {
        while (destsz > 0) {
            if (*dest == '\0') {
                // Add nulls to complete dest.
                while (destsz) {
                    *dest = '\0';
                    destsz--;
                    dest++;
                }
                return EOK;
            }
            destsz--;
            dest++;
            count--;
            if (count == 0) {
                // We have copied count characters, add null terminator.
                *dest = '\0';
            }
        }
        return ESNOSPC;
    }

    // Normal copying.
    while (destsz > 0) {
        // Copy data from src to dest.
        *dest = *src;

        if (count == 0) {
            // We have copied count characters, add null terminator.
            *dest = '\0';
        }

        // Check for end of copying.
        if (*dest == '\0') {
            // Add nulls to complete dest.
            while (destsz) {
                *dest = '\0';
                destsz--;
                dest++;
            }
            return EOK;
        }
        destsz--;
        count--;
        dest++;
        src++;
    }

    // Ran out of space in dest, and did not find the null terminator in src.
    return ESNOSPC;
}
