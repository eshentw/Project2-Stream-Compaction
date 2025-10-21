#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define BLOCK_SIZE 256
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

        void exclusiveScanImplOpt(int n, int *odata, const int *idata){
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

            int gridSize = static_cast<int>((fullSize + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE));
            cudaMalloc(&dev_blockSum, gridSize * sizeof(int));
            cudaMemset(dev_blockSum, 0, gridSize * sizeof(int));
            blockSum = new int[gridSize];

            kernScanOpt<<<gridSize, BLOCK_SIZE,
                ((2 * BLOCK_SIZE) + CONFLICT_FREE_OFFSET(2 * BLOCK_SIZE)) * sizeof(int)>>>(
                    fullSize, dev_idata, dev_odata, dev_blockSum);
            checkCUDAError("kernScanOpt failed!");
            cudaMemcpy(blockSum, dev_blockSum, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
            // simple CPU exclusive scan since gridSize is small
            int prefix = 0;
            for (int i = 0; i < gridSize; ++i) {
                int current = blockSum[i];
                blockSum[i] = prefix;
                prefix += current;
            }
            
            cudaMemcpy(dev_blockSum, blockSum, gridSize * sizeof(int), cudaMemcpyHostToDevice);
            // add the scanned blockSum to each block
            StreamCompaction::Common::kernAddOffset<<<gridSize, BLOCK_SIZE>>>(
                fullSize, dev_odata, dev_blockSum, (2 * BLOCK_SIZE));
            checkCUDAError("kernAddOffset failed!");
            // copy back the n elements
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_blockSum);
            delete[] blockSum;
        }

        void scan(int n, int *odata, const int *idata) {
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
            timer().startGpuTimer();
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
            timer().endGpuTimer();
            // only copy the n elements back
            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_data);
        }
        /*
            * Work-efficient parallel scan (prefix sum) with shared memory optimization.
            Upsweep and downsweep in shared memory within each block.
            smem = [a0, a1, a2, a3, a4, a5, a6, a7]
            Upsweep:
                step1: offset = 1, d = 4, threads tid = 0..3
                    left = offset * ((tid<<1)+1)-1 = (1*((0,2,4,6)+1)-1) = (0,2,4,6)
                    right = offset * ((tid<<1)+2)-1 = (1*((0,2,4,6)+2)-1) = (1,3,5,7)
                    smem = [a0, a0+a1, a2, a2+a3, a4, a4+a5, a6, a6+a7]
                step2: offset = 2, d = 2, threads tid = 0..1\
                    left = offset * ((tid<<1)+1)-1 = (2*((0,2)+1)-1) = (1,5)
                    right = offset * ((tid<<1)+2)-1 = (2*((0,2)+2)-1) = (3,7)
                    smem = [a0, a0+a1, a2, a0+a1+a2+a3, a4, a4+a5, a6, a4+a5+a6+a7]
                step3: offset = 4, d = 1, threads tid = 0
                    left = offset * ((tid<<1)+1)-1 = (4*(0+1)-1) = 3
                    right = offset * ((tid<<1)+2)-1 = (4*(0+2)-1) = 7
                    smem = [a0, a0+a1, a2, a0+a1+a2+a3, a4, a4+a5, a6, a0+a1+a2+a3+a4+a5+a6+a7]

            smem = [a0, a0+a1, a2, a0+a1+a2+a3, a4, a4+a5, a6, 0]
            Downsweep:
                step1: offset = 4, d = 1, threads tid = 0
                    left = offset * ((tid<<1)+1)-1 = (4*(0+1)-1) = 3
                    right = offset * ((tid<<1)+2)-1 = (4*(0+2)-1) = 7
                    smem = [a0, a0+a1, a2, 0, a4, a4+a5, a6, a0+a1+a2+a3]
                step2: offset = 2, d = 2, threads tid = 0..1
                    left = offset * ((tid<<1)+1)-1 = (2*((0,2)+1)-1) = (1,5)
                    right = offset * ((tid<<1)+2)-1 = (2*((0,2)+2)-1) = (3,7)
                    smem = [a0, 0, a2, a0+a1, a4, a0+a1+a2+a3, a6, a0+a1+a2+a3+a4+a5]
                step3: offset = 1, d = 4, threads tid = 0..3
                    left = offset * ((tid<<1)+1)-1 = (0*((0,2,4,6)+1)-1) = (0,2,4,6)
                    right = offset * ((tid<<1)+2)-1 = (0*((0,2,4,6)+2)-1) = (1,3,5,7)
                    smem = [0, a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, a0+a1+a2+a3+a4, a0+a1+a2+a3+a4+a5, a0+a1+a2+a3+a4+a5+a6]
        */

        __global__ void kernScanOpt(size_t n, const int *idata, int *odata, int *blockSum) {
            extern __shared__ int smem[];

            int blockElements = blockDim.x << 1; // 2 * blocksize
            size_t blockStart = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockElements);

            int ai = threadIdx.x;
            int bi = threadIdx.x + blockDim.x;

            size_t indexA = blockStart + static_cast<size_t>(ai);
            size_t indexB = blockStart + static_cast<size_t>(bi);

            int paddedA = ai + CONFLICT_FREE_OFFSET(ai);
            int paddedB = bi + CONFLICT_FREE_OFFSET(bi);

            smem[paddedA] = (indexA < n) ? idata[indexA] : 0;
            smem[paddedB] = (indexB < n) ? idata[indexB] : 0;

            int offset = 1;
            int totalThreads = blockDim.x;

            for (int d = totalThreads; d > 0; d >>= 1) {
                __syncthreads();
                if (threadIdx.x < d) {
                    int left = offset * ((threadIdx.x << 1) + 1) - 1;
                    int right = offset * ((threadIdx.x << 1) + 2) - 1;
                    left += CONFLICT_FREE_OFFSET(left);
                    right += CONFLICT_FREE_OFFSET(right);
                    smem[right] += smem[left];
                }
                offset <<= 1;
            }

            if (threadIdx.x == 0) {
                int last = blockElements - 1;
                int paddedLast = last + CONFLICT_FREE_OFFSET(last);
                blockSum[blockIdx.x] = smem[paddedLast];
                smem[paddedLast] = 0;
            }

            for (int d = 1; d < blockElements; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (threadIdx.x < d) {
                    int left = offset * ((threadIdx.x << 1) + 1) - 1;
                    int right = offset * ((threadIdx.x << 1) + 2) - 1;
                    left += CONFLICT_FREE_OFFSET(left);
                    right += CONFLICT_FREE_OFFSET(right);
                    int t = smem[left];
                    smem[left] = smem[right];
                    smem[right] += t;
                }
            }

            __syncthreads();

            if (indexA < n) {
                odata[indexA] = smem[paddedA];
            }
            if (indexB < n) {
                odata[indexB] = smem[paddedB];
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

            int gridSize = static_cast<int>((fullSize + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE));
            cudaMalloc(&dev_blockSum, gridSize * sizeof(int));
            cudaMemset(dev_blockSum, 0, gridSize * sizeof(int));
            blockSum = new int[gridSize];
            timer().startGpuTimer();

            kernScanOpt<<<gridSize, BLOCK_SIZE,
                ((2 * BLOCK_SIZE) + CONFLICT_FREE_OFFSET(2 * BLOCK_SIZE)) * sizeof(int)>>>(
                    fullSize, dev_idata, dev_odata, dev_blockSum);
            checkCUDAError("kernScanOpt failed!");
            cudaMemcpy(blockSum, dev_blockSum, gridSize * sizeof(int), cudaMemcpyDeviceToHost);
            // simple CPU exclusive scan since gridSize is small
            int prefix = 0;
            for (int i = 0; i < gridSize; ++i) {
                int current = blockSum[i];
                blockSum[i] = prefix;
                prefix += current;
            }
            
            cudaMemcpy(dev_blockSum, blockSum, gridSize * sizeof(int), cudaMemcpyHostToDevice);
            // add the scanned blockSum to each block
            StreamCompaction::Common::kernAddOffset<<<gridSize, BLOCK_SIZE>>>(
                fullSize, dev_odata, dev_blockSum, (2 * BLOCK_SIZE));
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

        int compactOpt(int n, int *odata, const int *idata) {
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
			exclusiveScanImplOpt(n, indices, mask);
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
