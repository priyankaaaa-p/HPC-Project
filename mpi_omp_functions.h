#ifndef _MPI_OMP_FUNCTIONS_
#define _MPI_OMP_FUNCTIONS_

#include <stdint.h>

#define GREY 0
#define RGB 1

typedef struct _inputArguments_ {
    char imageNameInput[50];
    char imageNameOutput[50];
    int height;
    int width;
    int iterations;
    int imageType;
}inputArguments;


void ParseInput(int, char **, inputArguments *);

void splitImageToBlocks(int, int, int, int *, int *);

void convoluteGREY(uint8_t *, uint8_t *, float **, int, int, int, int, int);

void innerConvoluteGREY(uint8_t *, uint8_t *, float **, int , int, int);

void convoluteRGB(uint8_t *, uint8_t *, float **, int, int, int, int, int);

void innerConvoluteRGB(uint8_t *, uint8_t *, float **, int , int, int);

#endif
