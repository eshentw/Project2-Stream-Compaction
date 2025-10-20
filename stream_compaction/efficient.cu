#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 512
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

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
                active = active == 0 ? 1 : active; // ensure at least one block
                int gridSize = static_cast<int>((active + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpsweep<<<gridSize, BLOCK_SIZE>>>(fullSize, level, dev_data);
                checkCUDAError("kernUpsweep failed!");
            }
            cudaMemset(dev_data + fullSize - 1, 0, sizeof(int));
            checkCUDAError("memset failed!");
            for (int level = depth; level >= 1; --level) {
                size_t active = static_cast<size_t>(fullSize) >> level;
                active = active == 0 ? 1 : active;
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

        __global__ void kernScanOpt(size_t n, const int *idata, int *odata, int *blockSum) {
            extern __shared__ int smem[];
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int tid = threadIdx.x;
            // pad the shared memory to avoid bank conflicts
            // since share memory is allocated per block, so we need to add offset based on tid only
            int paddedIdx = tid + CONFLICT_FREE_OFFSET(tid);
            int value = 0;
            if (index < n) {
                value = idata[index];
            }
            smem[paddedIdx] = value;
            int left, right;
            // up-sweep
            for (int offset = 1; offset <= ilog2ceil(blockDim.x); offset += 1) {
                __syncthreads();
                size_t stride = 1 << offset;
                size_t idx = threadIdx.x * stride;
                if (idx + stride - 1 < blockDim.x) {
                    left = idx + (stride >> 1) - 1;
                    left += CONFLICT_FREE_OFFSET(left);
                    right = idx + stride - 1;
                    right += CONFLICT_FREE_OFFSET(right);
                    smem[right] += smem[left];
                }
            }
            // write the sum of each block to blockSum array
            if (threadIdx.x == 0) {
                int last = blockDim.x - 1;
                int paddedLast = last + CONFLICT_FREE_OFFSET(last);
                blockSum[blockIdx.x] = smem[paddedLast];
                smem[paddedLast] = 0;
            }
            // down-sweep
            for (int offset = ilog2ceil(blockDim.x); offset >= 1; --offset) {
                __syncthreads();
                size_t stride = 1 << offset;
                size_t idx = threadIdx.x * stride;
                if (idx + stride - 1 < blockDim.x) {
                    left = idx + (stride >> 1) - 1;
                    left += CONFLICT_FREE_OFFSET(left);
                    right = idx + stride - 1;
                    right += CONFLICT_FREE_OFFSET(right);
                    int t = smem[left];
                    smem[left] = smem[right];
                    smem[right] += t;
                }
            }
            __syncthreads();
            if (index < n) {
                odata[index] = smem[paddedIdx];
            }
        }

        void scanOpt(int n, int *odata, const int *idata) {
            int *dev_idata, *dev_odata;
            int *dev_blockSum, *blockSum;

            // pad the input array to the next power of two
            int log2ceil = ilog2ceil(n);
            size_t fullSize = 1 << log2ceil;

            cudaMalloc(&dev_idata, fullSize * sizeof(int));
            cudaMalloc(&dev_odata, fullSize * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
             if (fullSize > n) {
                 cudaMemset(dev_idata + n, 0, (fullSize - n) * sizeof(int));
                 checkCUDAError("padding memset failed");
             }

            int gridSize = static_cast<int>((fullSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
            cudaMalloc(&dev_blockSum, gridSize * sizeof(int));
            cudaMemset(dev_blockSum, 0, gridSize * sizeof(int));
            blockSum = new int[gridSize];

            timer().startGpuTimer();
            kernScanOpt<<<gridSize, BLOCK_SIZE,
                (BLOCK_SIZE + CONFLICT_FREE_OFFSET(BLOCK_SIZE)) * sizeof(int)>>>(
                    fullSize, dev_idata, dev_odata, dev_blockSum);
            checkCUDAError("kernScanOpt failed!");
            cudaMemcpy(blockSum, dev_blockSum, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
            // scan the blockSum array across blocks
//            exclusiveScanImpl(gridSize, blockSum, blockSum);
            // simple CPU exclusive scan since gridSize is small
            int prefix = 0;
            for (int i = 0; i < gridSize; ++i) {
                int current = blockSum[i];
                blockSum[i] = prefix;
                prefix += current;
            }
            
            cudaMemcpy(dev_blockSum, blockSum, gridSize * sizeof(int), cudaMemcpyHostToDevice);
            // add the scanned blockSum to each block
            StreamCompaction::Common::kernAddOffset<<<(fullSize + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
                fullSize, dev_odata, dev_blockSum, BLOCK_SIZE);
            checkCUDAError("kernAddOffset failed!");

            timer().endGpuTimer();
            // copy back the n elements
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_blockSum);
            delete[] blockSum;
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
