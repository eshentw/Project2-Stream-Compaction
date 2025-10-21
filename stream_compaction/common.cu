#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}


namespace StreamCompaction {
    namespace Common {

        /**
         * Maps an array to an array of 0s and 1s for stream compaction. Elements
         * which map to 0 will be removed, and elements which map to 1 will be kept.
         */
        __global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
            // TODO
        	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        	if (index >= n) return;
            bools[index] = (idata[index] != 0) ? 1 : 0;
            return;

        }

        /**
         * Performs scatter on an array. That is, for each element in idata,
         * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
         */
        __global__ void kernScatter(int n, int *odata,
                const int *idata, const int *bools, const int *indices) {
            // TODO
        	size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        	if (index >= n) return;
        	if (bools[index])
        		odata[indices[index]] = idata[index];
        	return;
        }

        __global__ void kernAddOffset(size_t n, int *data, const int *blockSum, int blockSize) {
            size_t blockStart = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockSize);
            int offset = blockSum[blockIdx.x];

            size_t indexA = blockStart + threadIdx.x;
            size_t indexB = blockStart + threadIdx.x + blockDim.x;

            if (indexA < n) {
                data[indexA] += offset;
            }

            if (indexB < n) {
                data[indexB] += offset;
            }
        }

    }
}
