#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <"C:\Include\mpi.h">
#include <omp.h>

#include "C:\Users\PRIYANKA\Desktop\SEM VI\HPC\hpc project\mpi_omp_functions.h"

/* Read command line arguments from user and store them to a struct */
void ParseInput(int argc, char **argv, inputArguments *input) {
    int i;

    if (argc == 11) {
        for (i = 1 ; i < argc ; i++) {
            if (!strcmp(argv[i], "-i")) { // If flag is -i ..
                strcpy(input->imageNameInput, argv[++i]);

				sprintf(input->imageNameOutput, "Filtered_%s", argv[i]);
                continue;
            }
            if (!strcmp(argv[i], "-h")) { // If flag is -h ..          
                input->height = atoi(argv[++i]);
                continue;
            }
            if (!strcmp(argv[i], "-w")) { // If flag is -w ..
                input->width = atoi(argv[++i]);
                continue;
            }
            if (!strcmp(argv[i], "-l")) { // If flag is -l ..          
                input->iterations = atoi(argv[++i]);
                continue;
            }
            if (!strcmp(argv[i], "-t")) { // If flag is -t ..          
                if (!strcmp(argv[++i], "GREY"))
                    input->imageType = GREY;
                else if (!strcmp(argv[i], "RGB"))
                    input->imageType = RGB;
                else {
                    fprintf(stderr, "Input error: wrong image type\n");
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    exit(EXIT_FAILURE);
                }
                continue;
            }
        }
    }
    else {
        fprintf(stderr, "Input error: wrong number of command line arguments\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
}


/* Find a proper image split to balance load calculations on each process */
void splitImageToBlocks(int height, int width, int numProcesses, int *splitRows, int *splitCols) {
    int rowNumber, colNumber, blockSize, blockSizeMin = INT_MAX;

    for (rowNumber = 1 ; rowNumber <= numProcesses ; rowNumber++) {
        if ((height%rowNumber == 0) && (numProcesses%rowNumber == 0))
            colNumber = numProcesses / rowNumber;
        else
            continue;

        if (width%colNumber == 0)
            blockSize = height/rowNumber + width/colNumber;
        else
            continue;

        if (blockSize < blockSizeMin) {
            blockSizeMin = blockSize;
            *splitRows = rowNumber;
            *splitCols = colNumber;
        }
    }
}

void convoluteGREY(uint8_t *source, uint8_t *destination, float **filter, int width, int rowStart, int colStart, int rowEnd, int colEnd) {
    int i, j;
    float sum;

#pragma omp parallel for shared(source, destination) schedule(static) collapse(2)
    for (i = rowStart ; i <= rowEnd ; i++) {
        for (j = colStart ; j <= colEnd ; j++) {
            innerConvoluteGREY(source, destination, filter, width, i, j);
        }
    }
}

void innerConvoluteGREY(uint8_t *source, uint8_t *destination, float **filter, int width, int i, int j) {
    int x, y, sum = 0;

    for (x = (i-1) ; x <= (i+1) ; x++) {
        for (y = (j-1) ; y <= (j+1) ; y++) {
            sum += source[x*width + y] * filter[x-(i-1)][y-(j-1)];
        }
    }
    destination[i*width + j] = sum;
}


/* Apply filter through convolution on RGB images */

void convoluteRGB(uint8_t *source, uint8_t *destination, float **filter, int width, int rowStart, int colStart, int rowEnd, int colEnd) {
    int i, j, x, y;
    float sumR, sumG, sumB;

#pragma omp parallel for shared(source, destination) schedule(static) collapse(2)
    for (i = rowStart ; i <= rowEnd ; i++) {
        for (j = colStart ; j <= colEnd ; j++) {
            innerConvoluteRGB(source, destination, filter, width, i, j);
        }
    }
}


void innerConvoluteRGB(uint8_t *source, uint8_t *destination, float **filter, int width, int i, int j) {
    int x, y, sumR = 0, sumG = 0, sumB = 0;

    for (x = (i-1) ; x <= (i+1) ; x++) {
        for (y = (3*(j-1)) ; y <= (3*(j+1)) ; y+=3) {
            sumR += source[x*width + y] * filter[x-(i-1)][(y-(3*(j-1)))/3];
            sumG += source[x*width + y+1] * filter[x-(i-1)][(y-(3*(j-1)))/3];
            sumB += source[x*width + y+2] * filter[x-(i-1)][(y-(3*(j-1)))/3];
        }
    }
    destination[i*width + 3*j] = sumR;
    destination[i*width + 3*j+1] = sumG;
    destination[i*width + 3*j+2] = sumB;
}
