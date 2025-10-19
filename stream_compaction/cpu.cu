#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            // static persist between calls throughout program execution
            static PerformanceTimer timer;
            return timer;
        }

        namespace {
            void exclusiveScanImpl(int n, int *odata, const int *idata) {
                if (n <= 0) {
                    return;
                }
                odata[0] = 0;
                for (int i = 1; i < n; ++i) {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            exclusiveScanImpl(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int j = 0;
            for (int i = 0; i < n; i++){
                if (idata[i] != 0){
                    odata[j] = idata[i];
                    j++;
                }
            }
            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int *mask = new int[n];
            for (int i = 0; i < n; i++){
                mask[i] = (idata[i] != 0) ? 1 : 0;
            }
            int *scanResult = new int[n];
            exclusiveScanImpl(n, scanResult, mask);
            for (int i = 0; i < n; i++){
                if (mask[i] == 1){
                    odata[scanResult[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            int compactedSize = 0;
            if (n > 0){
                compactedSize = scanResult[n - 1] + mask[n - 1];
            }
            delete[] mask;
            delete[] scanResult;
            return compactedSize;
        }
    }
}
