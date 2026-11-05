#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "components.h"

#define RGB_COMPONENT_COLOR 255

void sobel(unsigned char* in, unsigned char* out, int HEIGHT, int WIDTH, int a0, int a1, int a2, int a3, int a4) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            if ((i < 1) || (i > HEIGHT - 2) || (j < 1) || (j > WIDTH - 2)) {
                out[i * WIDTH + j] = 0;
            } else {
                uint12_t_hw p00 = in[(i - 1) * WIDTH + (j - 1)];
                uint12_t_hw p01 = in[(i - 1) * WIDTH + (j    )];
                uint12_t_hw p02 = in[(i - 1) * WIDTH + (j + 1)];
                uint12_t_hw p10 = in[(i    ) * WIDTH + (j - 1)];
                uint12_t_hw p12 = in[(i    ) * WIDTH + (j + 1)];
                uint12_t_hw p20 = in[(i + 1) * WIDTH + (j - 1)];
                uint12_t_hw p21 = in[(i + 1) * WIDTH + (j    )];
                uint12_t_hw p22 = in[(i + 1) * WIDTH + (j + 1)];

                // Convolução Horizontal (Gx)
                uint12_t_hw gx_x1 = add[a0](p00, p20);
                uint12_t_hw gx_x2 = p10 << 1;
                uint12_t_hw gx_x3 = add[a1](gx_x1, gx_x2);
                uint12_t_hw gx_x4 = add[a2](p02, p22);
                uint12_t_hw gx_x5 = p12 << 1;
                uint12_t_hw gx_x6 = add[a3](gx_x4, gx_x5);
                int gx = abs((int)gx_x3 - (int)gx_x6);

                // Convolução Vertical (Gy)
                uint12_t_hw gy_x1 = add[a0](p00, p02);
                uint12_t_hw gy_x2 = p01 << 1;
                uint12_t_hw gy_x3 = add[a1](gy_x1, gy_x2);
                uint12_t_hw gy_x4 = add[a2](p20, p22);
                uint12_t_hw gy_x5 = p21 << 1;
                uint12_t_hw gy_x6 = add[a3](gy_x4, gy_x5);
                int gy = abs((int)gy_x3 - (int)gy_x6);

                // Magnitude final usando o 5º somador aproximado sintetizado (a4)
                uint12_t_hw sum = add[a4]((uint12_t_hw)gx, (uint12_t_hw)gy);
                int sum_val = (int)sum;
                sum_val = (sum_val > 255) ? 255 : sum_val;
                out[i * WIDTH + j] = (unsigned char)sum_val;
            }
        }
    }
}

typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "PPMImage: Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format '%s' (must be 'P6')\n", filename);
        exit(1);
    }

    img = (PPMImage *)malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img->data) {
        fprintf(stderr, "Unable to allocate memory for image data\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != (size_t)img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(char* outname, unsigned char* img, const char *filename, const char * kind, int height, int width) {
    char base[512];
    strncpy(base, filename, sizeof(base) - 1);
    base[sizeof(base) - 1] = '\0';

    char *dot = strrchr(base, '.');
    if (dot != NULL) {
        *dot = '\0';
    }

    sprintf(outname, "%s.%s.sobel.ppm", base, kind);

    FILE *output = fopen(outname, "wb");
    if (!output) {
        fprintf(stderr, "Error: could not open output file '%s'!\n", outname);
        return;
    }

    // Grava no formato P5 (Grayscale binário - ultrarrápido)
    fprintf(output, "P5\n%d %d\n255\n", width, height);
    fwrite(img, 1, width * height, output);
    fclose(output);
}

void init(unsigned char* buf, PPMImage * image) {
    for (int i = 0; i < image->y * image->x; i++) {
        buf[i] = image->data[i].red;
    }
}

void print(char **r, unsigned size) {
    for (unsigned i = 0; i < size; i++) {
        printf(" %s", r[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc < 8) {
        fprintf(stderr, "Uso: %s <dataset.in> <kind> <a0> <a1> <a2> <a3> <a4>\n", argv[0]);
        return 1;
    }

    char *dataset_file = argv[1];
    char *kind = argv[2];
    int a0 = atoi(argv[3]);
    int a1 = atoi(argv[4]);
    int a2 = atoi(argv[5]);
    int a3 = atoi(argv[6]);
    int a4 = atoi(argv[7]);

    FILE *input = fopen(dataset_file, "r");
    if (!input) {
        fprintf(stderr, "Error: could not open file '%s'\n", dataset_file);
        return 1;
    }

    unsigned size = 0;
    if (fscanf(input, "%u", &size) != 1) {
        fprintf(stderr, "Error: invalid dataset header\n");
        fclose(input);
        return 1;
    }

    char filename[256];
    char **inputList = (char**) malloc(size * sizeof(char*));
    char **filesList = (char**) malloc(size * sizeof(char*));

    for (unsigned i = 0; i < size; i++) {
        if (fscanf(input, "%255s\n", filename) != 1) break;
        inputList[i] = (char*) malloc(strlen(filename) + 1);
        filesList[i] = (char*) malloc(strlen(filename) + 128);
        strcpy(inputList[i], filename);
    }
    fclose(input);

#pragma omp parallel for shared(inputList, filesList)
    for (unsigned i = 0; i < size; i++) {
        PPMImage *image = readPPM(inputList[i]);
        unsigned char *inputImage = (unsigned char *) malloc(sizeof(unsigned char) * image->y * image->x);
        unsigned char *outImage = (unsigned char *) malloc(sizeof(unsigned char) * image->y * image->x);

        init(inputImage, image);

        sobel(inputImage, outImage, image->y, image->x, a0, a1, a2, a3, a4);

        writePPM(filesList[i], outImage, inputList[i], kind, image->y, image->x);

        free(image->data);
        free(image);
        free(inputImage);
        free(outImage);
    }

    print(filesList, size);

    for (unsigned i = 0; i < size; i++) {
        free(inputList[i]);
        free(filesList[i]);
    }
    free(inputList);
    free(filesList);

    return EXIT_SUCCESS;
}
