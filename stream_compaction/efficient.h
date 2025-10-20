#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        __global__ void kernUpsweep(size_t n, int level, int *data);

        __global__ void kernDownsweep(size_t n, int level, int *data);

        __global__ void kernScanOpt(size_t n, const int *idata, int *odata, int *blockSum);

        void exclusiveScanImpl(int n, int *odata, const int *idata);

        void scan(int n, int *odata, const int *idata);

        void scanOpt(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
