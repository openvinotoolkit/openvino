#include "bmp_reader.h"

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

int readBmpImage(const char* fileName, BitMap* image) {
    size_t cnt;
    int status = 0;
    FILE* input = 0;

    if (NULL == fileName || NULL == image) {
        printf("[BMP] bad arguments\n");
        status = -1;
        goto Exit;
    }

    memset(image, 0, sizeof(BitMap));

    input = fopen(fileName, "rb");
    if (input == NULL) {
        printf("[BMP] file %s is not opened\n", fileName);
        status = 1;
        goto Exit;
    }

    cnt = fread(&image->header.type, sizeof(image->header.type), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.type)) {
        printf("[BMP] file read error\n");
        status = 2;
        goto Exit;
    }

    if (image->header.type != 'M' * 256 + 'B') {
        printf("[BMP] file is not bmp type\n");
        status = 2;
        goto Exit;
    }

    cnt = fread(&image->header.size, sizeof(image->header.size), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.size)) {
        printf("[BMP] file read error\n");
        status = 2;
        goto Exit;
    }

    cnt = fread(&image->header.reserved, sizeof(image->header.reserved), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.reserved)) {
        printf("[BMP] file read error\n");
        status = 2;
        goto Exit;
    }

    cnt = fread(&image->header.offset, sizeof(image->header.offset), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.offset)) {
        printf("[BMP] file read error\n");
        status = 2;
        goto Exit;
    }

    cnt = fread(&image->infoHeader, sizeof(BmpInfoHeader), sizeof(unsigned char), input);
    if (cnt != sizeof(image->header.offset)) {
        printf("[BMP] file read error\n");
        status = 2;
        goto Exit;
    }

    image->width = image->infoHeader.width;
    image->height = abs(image->infoHeader.height);

    if (image->infoHeader.bits != 24) {
        printf("[BMP] 24bpp only supported. But input has: %d\n", image->infoHeader.bits);
        return 3;
    }

    if (image->infoHeader.compression != 0) {
        printf("[BMP] compression not supported\n");
        return 4;
    }

    int padSize = image->width & 3;
    size_t row_size = (size_t)image->width * 3;
    char pad[3];
    size_t size = row_size * image->height;

    image->data = malloc(sizeof(char) * size);
    if (NULL == image->data) {
        printf("[BMP] memory allocation failed\n");
        return 5;
    }

    if (0 != fseek(input, image->header.offset, SEEK_SET)) {
        printf("[BMP] file seek error\n");
        status = 2;
        goto Exit;
    }

    // reading by rows in invert vertically
    int i;
    for (i = 0; i < image->height; i++) {
        unsigned int storeAt = image->infoHeader.height < 0 ? i : (unsigned int)image->height - 1 - i;
        cnt = fread(image->data + row_size * storeAt, row_size, sizeof(unsigned char), input);
        if (cnt != row_size) {
            printf("[BMP] file read error\n");
            status = 2;
            goto Exit;
        }

        cnt = fread(pad, padSize, sizeof(unsigned char), input);
        if (cnt != padSize) {
            printf("[BMP] file read error\n");
            status = 2;
            goto Exit;
        }
    }

Exit:
    if (0 != status && NULL != image && NULL != image->data) {
        free(image->data);
    }

    if (NULL != input) {
        fclose(input);
    }

    return status;
}
