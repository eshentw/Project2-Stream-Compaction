#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCK_SIZE 512

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScan(int n, int logoffset, int *todata, const int *frdata) {
            // Since GPU threads are not guaranteed to be in order, we need to compute
            // the scan result using 2 device arrays the data from the previous step frdata
            // and writing to the todata array.
            size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) {
                return;
            }
            size_t offset = powf(2, logoffset - 1);
            if (index >= offset) {
                todata[index] = frdata[index - offset] + frdata[index];
            }
            else {
                todata[index] = frdata[index];
            }
            
            return;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Allocate device memory for ping-pong buffers
            const int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            int *dev_bufA, *dev_bufB;
            cudaMalloc((void**)&dev_bufA, n * sizeof(int));
            cudaMalloc((void**)&dev_bufB, n * sizeof(int));

            // Shift right by 1 for exclusive scan (set first element to 0)
            // initialize dev_bufB to all 0s also handle when it's not power of 2
            cudaMemset(dev_bufB, 0, sizeof(int));
            cudaMemcpy(dev_bufB + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_bufA, dev_bufB, n * sizeof(int), cudaMemcpyDeviceToDevice);

            timer().startGpuTimer();
            // TODO
            for (int level = 1; level <= ilog2ceil(n); level++) {
                kernScan<<<numBlocks, BLOCK_SIZE>>>(n, level, dev_bufB, dev_bufA);
                checkCUDAError("kernScan failed!");
                // Swap buffers
                std::swap(dev_bufA, dev_bufB);
            }
            timer().endGpuTimer();
            // Copy result back to host and free device memory
            cudaMemcpy(odata, dev_bufA, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_bufA);
            cudaFree(dev_bufB);

        }
    }
}
