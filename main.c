#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <"C:\hpc project\mpi_omp_functions.h">

#include "mpi_omp_functions.h"

int main (int argc, char *argv[]) {
    int i, j, myRank, numProcesses, splitRows, splitCols, rowsPerProcess, colsPerProcess, processRow, processCol, neighbourUp, neighbourDown, neighbourRight, neighbourLeft;
    float filterBlur[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};    /* Filter */
    float **normalizedFilterBlur;
    double timeStart, timeEnd, timeFinal;
	bool neighbourUpExists = false, neighbourDownExists = false, neighbourRightExists = false, neighbourLeftExists = false;
    uint8_t *source, *destination, *tmp;
    inputArguments input;

    /* Normalize filter */
    normalizedFilterBlur = (float**) malloc(3 * sizeof(float*));
    if (normalizedFilterBlur == NULL) {
        fprintf(stderr, "malloc: failed to allocate memory in main\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    for (i = 0 ; i < 3 ; i++) {
        normalizedFilterBlur[i] = (float*) malloc(3 * sizeof(float));
        if (normalizedFilterBlur[i] == NULL) {
            fprintf(stderr, "malloc: failed to allocate memory in main\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
        }

        for (j = 0 ; j < 3 ; j++)
            normalizedFilterBlur[i][j] = (float) filterBlur[i][j]/16.0;
    }


    /* MPI Start */
    MPI_Init(&argc, &argv);                                     /* MPI initialization */
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);                     /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);               /* get number of processes */

    /* MPI Variables */
    MPI_Status status;
    MPI_Datatype greyRow, greyCol, rgbRow, rgbCol;
    MPI_File inputFile, outputFile;
    MPI_Request sendUp, sendDown, sendRight, sendLeft, receiveUp, receiveDown, receiveRight, receiveLeft;

    /* First process parses the command line input from user and computes the image split */
    if (myRank == 0) {
        /* Parse command line arguments */
        ParseInput(argc, argv, &input);

        /* Find a good split into blocks for image */
        splitImageToBlocks(input.height, input.width, numProcesses, &splitRows, &splitCols);

        /* Number of pixels per block processing by each process */
        rowsPerProcess = input.height / splitRows;
        colsPerProcess = input.width / splitCols;
    }

    /* First process broadcasts variables to other processes */
    MPI_Bcast(&input.imageNameInput, 50, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&input.imageNameOutput, 50, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&input.height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&input.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&input.iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&input.imageType, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&splitCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsPerProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&colsPerProcess, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* For each process, find the start pixel of its block */
    processRow = (myRank / splitCols) * rowsPerProcess;
    processCol = (myRank % splitCols) * colsPerProcess;

    /* Find neighbour processes existance and id */
    if (processRow != 0) {
        neighbourUpExists = true;
        neighbourUp = myRank - splitCols;
    }
    if (processCol != 0) {
		neighbourLeftExists = true;
		neighbourLeft = myRank - 1;
	}
    if (processRow+rowsPerProcess != input.height) {
		neighbourDownExists = true;
		neighbourDown = myRank + splitCols;
	}
    if (processCol+colsPerProcess != input.width) {
		neighbourRightExists = true;
		neighbourRight = myRank + 1;
	}


    if (input.imageType == GREY) {
		/* GREY images datatypes */
		MPI_Type_contiguous(colsPerProcess, MPI_BYTE, &greyRow);
        MPI_Type_commit(&greyRow);

        MPI_Type_vector(rowsPerProcess, 1, colsPerProcess+2, MPI_BYTE, &greyCol);
        MPI_Type_commit(&greyCol);

        /* Allocate memory for GREY source image */
        source = (uint8_t*) calloc((rowsPerProcess+2) * (colsPerProcess+2), sizeof(uint8_t));
        if (source == NULL) {
            fprintf(stderr, "calloc: failed to allocate memory in main\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
    	}

        /* Allocate memory for GREY destination image */
		destination = (uint8_t*) calloc((rowsPerProcess+2) * (colsPerProcess+2), sizeof(uint8_t));
        if (destination == NULL) {
            fprintf(stderr, "calloc: failed to allocate memory in main\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
    	}

        /* Open GREY image file to read */
        MPI_File_open(MPI_COMM_WORLD, input.imageNameInput, MPI_MODE_RDONLY, MPI_INFO_NULL, &inputFile);

        /* Read only block of pixels corresponding to each process */
        for (i = 0 ; i < rowsPerProcess ; i++) {
			MPI_File_seek(inputFile, (processRow + i) * input.width + processCol, MPI_SEEK_SET);
			MPI_File_read(inputFile, &source[(i+1)*(colsPerProcess+2) + 1], colsPerProcess, MPI_BYTE, &status);
		}

        /* Close GREY image file */
        MPI_File_close(&inputFile);
    }
    else {
		/* RGB images datatypes */
		MPI_Type_contiguous(3*colsPerProcess, MPI_BYTE, &rgbRow);
        MPI_Type_commit(&rgbRow);

        MPI_Type_vector(rowsPerProcess, 3, 3*(colsPerProcess+2), MPI_BYTE, &rgbCol);
        MPI_Type_commit(&rgbCol);

        /* Allocate memory for RGB source image */
        source = (uint8_t*) calloc((rowsPerProcess+2) * (3*(colsPerProcess+2)), sizeof(uint8_t));
        if (source == NULL) {
            fprintf(stderr, "calloc: failed to allocate memory in main\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
    	}

        /* Allocate memory for RGB destination image */
		destination = (uint8_t*) calloc((rowsPerProcess+2) * (3*(colsPerProcess+2)), sizeof(uint8_t));
        if (destination == NULL) {
            fprintf(stderr, "calloc: failed to allocate memory in main\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return EXIT_FAILURE;
    	}

        /* Open RGB image file to read*/
        MPI_File_open(MPI_COMM_WORLD, input.imageNameInput, MPI_MODE_RDONLY, MPI_INFO_NULL, &inputFile);

        /* Read only block of pixels corresponding to each process */
        for (i = 0 ; i < rowsPerProcess ; i++) {
			MPI_File_seek(inputFile, 3*(processRow+i)*input.width + 3*processCol, MPI_SEEK_SET);
			MPI_File_read(inputFile, &source[(i+1)*(3*(colsPerProcess+2)) + 3], 3*colsPerProcess, MPI_BYTE, &status);
		}

        /* Close RGB image file */
        MPI_File_close(&inputFile);
    }

    /* Wait for all processes to reach this command to start the convolution all together */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Start counting time */
    timeStart = MPI_Wtime();

    /* Apply normalized filter n times */
    for (i = 0 ; i < input.iterations ; i++) {
        if (input.imageType == GREY) {
            if (neighbourUpExists) {
				MPI_Isend(&source[colsPerProcess+3], 1, greyRow, neighbourUp, 0, MPI_COMM_WORLD, &sendUp);
                MPI_Irecv(&source[1], 1, greyRow, neighbourUp, 0, MPI_COMM_WORLD, &receiveUp);
            }
            if (neighbourDownExists) {
				MPI_Isend(&source[rowsPerProcess*(colsPerProcess+2) + 1], 1, greyRow, neighbourDown, 0, MPI_COMM_WORLD, &sendDown);
                MPI_Irecv(&source[(rowsPerProcess+1)*(colsPerProcess+2) + 1], 1, greyRow, neighbourDown, 0, MPI_COMM_WORLD, &receiveDown);
            }
            if (neighbourRightExists) {
				MPI_Isend(&source[colsPerProcess+2 + colsPerProcess], 1, greyCol, neighbourRight, 0, MPI_COMM_WORLD, &sendRight);
                MPI_Irecv(&source[(colsPerProcess+2)+colsPerProcess+1], 1, greyCol, neighbourRight, 0, MPI_COMM_WORLD, &receiveRight);
            }
            if (neighbourLeftExists) {
				MPI_Isend(&source[colsPerProcess+3], 1, greyCol, neighbourLeft, 0, MPI_COMM_WORLD, &sendLeft);
                MPI_Irecv(&source[colsPerProcess+2], 1, greyCol, neighbourLeft, 0, MPI_COMM_WORLD, &receiveLeft);
            }

            convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, 1, 1, rowsPerProcess, colsPerProcess);

            if (neighbourUpExists) {
                MPI_Wait(&receiveUp, &status);
                convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, 1, 2, 1, colsPerProcess-1);
            }
            if (neighbourDownExists) {
                MPI_Wait(&receiveDown, &status);
                convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, rowsPerProcess, 2, rowsPerProcess, colsPerProcess-1);
            }
            if (neighbourRightExists) {
                MPI_Wait(&receiveRight, &status);
                convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, 2, colsPerProcess, rowsPerProcess-1, colsPerProcess);
            }
            if (neighbourLeftExists) {
                MPI_Wait(&receiveLeft, &status);
                convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, 2, 1, rowsPerProcess-1, 1);
            }

            /* Calculate corner data new values */
            if (neighbourUpExists && neighbourLeftExists)
                convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, 1, 1, 1, 1);
            if (neighbourDownExists && neighbourRightExists)
                convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, rowsPerProcess, colsPerProcess, rowsPerProcess, colsPerProcess);
            if (neighbourRightExists && neighbourUpExists)
                convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, 1, colsPerProcess, 1, colsPerProcess);
            if (neighbourLeftExists && neighbourDownExists)
                convoluteGREY(source, destination, normalizedFilterBlur, colsPerProcess+2, rowsPerProcess, 1, rowsPerProcess, 1);
        }
        else {
            if (neighbourUpExists) {
				MPI_Isend(&source[3*(colsPerProcess+2) + 3], 1, rgbRow, neighbourUp, 0, MPI_COMM_WORLD, &sendUp);
                MPI_Irecv(&source[3], 1, rgbRow, neighbourUp, 0, MPI_COMM_WORLD, &receiveUp);
            }
            if (neighbourDownExists) {
				MPI_Isend(&source[rowsPerProcess*3*(colsPerProcess+2) + 3], 1, rgbRow, neighbourDown, 0, MPI_COMM_WORLD, &sendDown);
                MPI_Irecv(&source[(rowsPerProcess+1)*3*(colsPerProcess+2) + 3], 1, rgbRow, neighbourDown, 0, MPI_COMM_WORLD, &receiveDown);
            }
            if (neighbourRightExists) {
				MPI_Isend(&source[3*(colsPerProcess+2) + 3*colsPerProcess], 1, rgbCol, neighbourRight, 0, MPI_COMM_WORLD, &sendRight);
                MPI_Irecv(&source[3*(colsPerProcess+2) + 3*(colsPerProcess+1)], 1, rgbCol, neighbourRight, 0, MPI_COMM_WORLD, &receiveRight);
            }
            if (neighbourLeftExists) {
				MPI_Isend(&source[3*(colsPerProcess+2) + 3], 1, rgbCol, neighbourLeft, 0, MPI_COMM_WORLD, &sendLeft);
                MPI_Irecv(&source[3*(colsPerProcess+2)], 1, rgbCol, neighbourLeft, 0, MPI_COMM_WORLD, &receiveLeft);
            }

            convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), 1, 1, rowsPerProcess, colsPerProcess);

            if (neighbourUpExists) {
                MPI_Wait(&receiveUp, &status);
                convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), 1, 2, 1, colsPerProcess-1);
            }
            if (neighbourDownExists) {
                MPI_Wait(&receiveDown, &status);
                convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), rowsPerProcess, 2, rowsPerProcess, colsPerProcess-1);
            }
            if (neighbourRightExists) {
                MPI_Wait(&receiveRight, &status);
                convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), 2, colsPerProcess, rowsPerProcess-1, colsPerProcess);
            }
            if (neighbourLeftExists) {
                MPI_Wait(&receiveLeft, &status);
                convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), 2, 1, rowsPerProcess-1, 1);
            }

            /* Calculate corner data new values */
            if (neighbourUpExists && neighbourLeftExists)
                convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), 1, 1, 1, 1);
            if (neighbourDownExists && neighbourRightExists)
                convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), rowsPerProcess, colsPerProcess, rowsPerProcess, colsPerProcess);
            if (neighbourRightExists && neighbourUpExists)
                convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), 1, colsPerProcess, 1, colsPerProcess);
            if (neighbourLeftExists && neighbourDownExists)
                convoluteRGB(source, destination, normalizedFilterBlur, 3*(colsPerProcess+2), rowsPerProcess, 1, rowsPerProcess, 1);
        }

        /* Wait until all above data sent and received from processes */
        if (neighbourUpExists)
            MPI_Wait(&sendUp, &status);
        if (neighbourDownExists)
            MPI_Wait(&sendDown, &status);
        if (neighbourRightExists)
            MPI_Wait(&sendRight, &status);
        if (neighbourLeftExists)
            MPI_Wait(&sendLeft, &status);

        /* Swap images to re-apply the filter as many times needed */
        tmp = source;
        source = destination;
        destination = tmp;
    }

    timeEnd = MPI_Wtime();
    timeFinal = timeEnd - timeStart;

    /* Write to output image */
    if (input.imageType == GREY) {
        /* Create and open output file to write */
        MPI_File_open(MPI_COMM_WORLD, input.imageNameOutput, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outputFile);

        /* Write only block of pixels corresponding to each process */
        for (i = 0 ; i < rowsPerProcess ; i++) {
			MPI_File_seek(outputFile, (processRow + i) * input.width + processCol, MPI_SEEK_SET);
			MPI_File_write(outputFile, &source[(i+1)*(colsPerProcess+2) + 1], colsPerProcess, MPI_BYTE, &status);
		}

        /* Close image file */
        MPI_File_close(&outputFile);
    }
    else {
        /* Create and open output file to write */
        MPI_File_open(MPI_COMM_WORLD, input.imageNameOutput, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outputFile);

        /* Write only block of pixels corresponding to each process */
        for (i = 0 ; i < rowsPerProcess ; i++) {
			MPI_File_seek(outputFile, 3*(processRow+i)*input.width + 3*processCol, MPI_SEEK_SET);
			MPI_File_write(outputFile, &source[(i+1)*(3*(colsPerProcess+2)) + 3], 3*colsPerProcess, MPI_BYTE, &status);
		}

        /* Close image file */
        MPI_File_close(&outputFile);
    }

    fprintf(stdout, "My rank is %d and my total time was %f\n", myRank, timeFinal);

    free(source);
    free(destination);
    free(normalizedFilterBlur);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
