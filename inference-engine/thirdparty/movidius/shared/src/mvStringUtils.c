// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvStringUtils.h"

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
