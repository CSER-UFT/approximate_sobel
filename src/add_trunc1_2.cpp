#include "ap_int.h"
typedef ap_uint<12> uint12_t_hw;

uint12_t_hw add_trunc1_2(uint12_t_hw a, uint12_t_hw b) {
    #pragma HLS inline off
    ap_uint<10> sum_trunc = a(11, 2) + b(11, 2);
    uint12_t_hw result = (sum_trunc, ap_uint<2>(1));

    return result;
}
