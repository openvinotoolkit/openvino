#include "bmp_reader.h"

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#define CLEANUP_AND_RETURN(x)                           \
    if (0 != x && NULL != image && NULL != image->data) \
        free(image->data);                              \
    if (input != NULL)                                  \
        fclose(input);                                  \
    return x;

int readBmpImage(const char* fileName, BitMap* image) {
    size_t cnt;
    FILE* input = 0;

    if (NULL == fileName || NULL == image) {
        printf("[BMP] bad arguments\n");
        CLEANUP_AND_RETURN(-1);
    }

    memset(image, 0, sizeof(BitMap));

    input = fopen(fileName, "rb");
    if (input == NULL) {
        printf("[BMP] file %s is not opened\n", fileName);
        CLEANUP_AND_RETURN(-1);
    }

    cnt = fread(&image->header.type, sizeof(image->header.type), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.type)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(-2);
    }

    if (image->header.type != 'M' * 256 + 'B') {
        printf("[BMP] file is not bmp type\n");
        CLEANUP_AND_RETURN(2);
    }

    cnt = fread(&image->header.size, sizeof(image->header.size), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.size)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(2);
    }

    cnt = fread(&image->header.reserved, sizeof(image->header.reserved), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.reserved)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(2);
    }

    cnt = fread(&image->header.offset, sizeof(image->header.offset), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.offset)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(2);
    }

    cnt = fread(&image->infoHeader, sizeof(BmpInfoHeader), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.offset)) {
        printf("[BMP] file read error\n");
        CLEANUP_AND_RETURN(2);
    }

    image->width = image->infoHeader.width;
    image->height = abs(image->infoHeader.height);

    if (image->infoHeader.bits != 24) {
        printf("[BMP] 24bpp only supported. But input has: %d\n", image->infoHeader.bits);
        CLEANUP_AND_RETURN(3);
    }

    if (image->infoHeader.compression != 0) {
        printf("[BMP] compression not supported\n");
        CLEANUP_AND_RETURN(4);
    }

    int padSize = image->width & 3;
    size_t row_size = (size_t)image->width * 3;
    char pad[3];
    size_t size = row_size * image->height;

    image->data = malloc(sizeof(char) * size);
    if (NULL == image->data) {
        printf("[BMP] memory allocation failed\n");
        CLEANUP_AND_RETURN(5);
    }

    if (0 != fseek(input, image->header.offset, SEEK_SET)) {
        printf("[BMP] file seek error\n");
        CLEANUP_AND_RETURN(2);
    }

    // reading by rows in invert vertically
    int i;
    int image_height = image->height;
    for (i = 0; i < image_height; i++) {
        unsigned int storeAt = image->infoHeader.height < 0 ? i : (unsigned int)image_height - 1 - i;
        cnt = fread(image->data + row_size * storeAt, row_size, sizeof(unsigned char), input);
        if (cnt != row_size) {
            printf("[BMP] file read error\n");
            CLEANUP_AND_RETURN(2);
        }

        cnt = fread(pad, padSize, sizeof(unsigned char), input);
        if (cnt != padSize) {
            printf("[BMP] file read error\n");
            CLEANUP_AND_RETURN(2);
        }
    }

    return 0;
}
