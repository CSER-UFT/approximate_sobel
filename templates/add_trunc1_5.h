
#include "ap_int.h"
typedef ap_uint<12> uint12_t_hw;

uint12_t_hw add_trunc1_5(uint12_t_hw a, uint12_t_hw b) {
    #pragma HLS INLINE OFF
    ap_uint<7> sum_trunc = a(11, 5) + b(11, 5);
    uint12_t_hw result = (sum_trunc, ap_uint<5>(1));
    return result;
}
