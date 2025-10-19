#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 512
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void kernUpsweep(size_t n, int level, int *data) {
            // need to use size_t for index calculation to avoid overflow
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            size_t offset = 1 << level;
            size_t idx = index * offset;
            if (idx + offset - 1 > n) {
                return;
            }
            data[idx + offset - 1] += data[idx + (offset >> 1) - 1];
        }

        __global__ void kernDownsweep(size_t n, int level, int *data) {
            // need to use size_t for index calculation to avoid overflow
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            size_t offset = 1 << level;
            size_t idx = index * offset;
            if (idx + offset - 1 > n) {
                return;
            }
            int t = data[idx + (offset >> 1) - 1];
            data[idx + (offset >> 1) - 1] = data[idx + offset - 1];
            data[idx + offset - 1] += t;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void exclusiveScanImpl(int n, int *odata, const int *idata) {
            int *dev_data;

            // pad the input array to the next power of two
            int log2ceil = ilog2ceil(n);
            size_t fullSize = 1 << log2ceil;

            cudaMalloc(&dev_data, fullSize * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
             if (fullSize > n) {
                 cudaMemset(dev_data + n, 0, (fullSize - n) * sizeof(int));
                 checkCUDAError("padding memset failed");
             }
            // const int gridSize = (fullSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int depth = ilog2ceil(fullSize);
            for (int level = 1; level <= depth; ++level) {
                size_t active = static_cast<size_t>(fullSize) >> level;
                if (active == 0) {
                    active = 1;
                }
                int gridSize = static_cast<int>((active + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpsweep<<<gridSize, BLOCK_SIZE>>>(fullSize, level, dev_data);
                checkCUDAError("kernUpsweep failed!");
            }
            cudaMemset(dev_data + fullSize - 1, 0, sizeof(int));
            checkCUDAError("memset failed!");
            for (int level = depth; level >= 1; --level) {
                size_t active = static_cast<size_t>(fullSize) >> level;
                if (active == 0) {
                    active = 1;
                }
                int gridSize = static_cast<int>((active + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownsweep<<<gridSize, BLOCK_SIZE>>>(fullSize, level, dev_data);
                checkCUDAError("kernDownsweep failed!");
            }
            // only copy the n elements back
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }

        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            exclusiveScanImpl(n, odata, idata);
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int *dev_idata, *dev_odata;
			int *dev_indices, *indices;
			int *dev_mask, *mask;
			// pad the input array to the next power of two
			int log2ceil = ilog2ceil(n);
			size_t fullSize = 1 << log2ceil;

			cudaMalloc(&dev_mask, n * sizeof(int));
            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));
            cudaMalloc(&dev_indices, n * sizeof(int));
			indices = new int[n];
			mask = new int[n];

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemset(dev_mask, 0, (n) * sizeof(int));
            checkCUDAError("mask memset failed");
            memset(indices, 0, n * sizeof(int));
            memset(mask, 0, n * sizeof(int));

			int gridSize = static_cast<int>((fullSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
			timer().startGpuTimer();
			// TODO
			// parallel populate the boolean mask
			StreamCompaction::Common::kernMapToBoolean<<<gridSize, BLOCK_SIZE>>>(
														n, dev_mask, dev_idata);
			checkCUDAError("kernMapToBoolean failed!");
            // copy back to host
            cudaMemcpy(mask, dev_mask, n * sizeof(int), cudaMemcpyDeviceToHost);
			//create scan indices from mask
			exclusiveScanImpl(n, indices, mask);
            // scatter the non-zero elements to output array
            cudaMemcpy(dev_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);
			StreamCompaction::Common::kernScatter<<<gridSize, BLOCK_SIZE>>>(
														n, dev_odata, dev_idata, dev_mask, dev_indices);
			checkCUDAError("kernScatter failed!");
			timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            int compactedSize = 0;
            if (n > 0){
                compactedSize = indices[n - 1];
                if (mask[n - 1]) compactedSize += 1;
            }
            delete[] indices;
            delete[] mask;
            cudaFree(dev_mask);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_indices);
            return compactedSize;

        }
    }
}
