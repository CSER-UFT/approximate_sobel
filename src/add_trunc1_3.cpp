#include "ap_int.h"

typedef ap_uint<12> uint12_t_hw;

uint12_t_hw add_trunc1_3(uint12_t_hw a, uint12_t_hw b) {
    #pragma HLS inline off
    ap_uint<9> sum_trunc = a(11, 3) + b(11, 3);
    uint12_t_hw result = (sum_trunc, ap_uint<3>(1));
    return result;
}
