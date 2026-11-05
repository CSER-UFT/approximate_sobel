#include "ap_int.h"
typedef ap_uint<12> uint12_t_hw;

uint12_t_hw add_trunc0_1(uint12_t_hw a, uint12_t_hw b) {
    #pragma HLS INLINE OFF
    ap_uint<11> sum_trunc = a(11, 1) + b(11, 1);
    uint12_t_hw result = (sum_trunc, ap_uint<1>(0));

    return result;
}
