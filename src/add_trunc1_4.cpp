#include "ap_int.h"
typedef ap_uint<12> uint12_t_hw;

uint12_t_hw add_trunc1_4(uint12_t_hw a, uint12_t_hw b) {
    #pragma HLS inline off
    ap_uint<8> sum_trunc = a(11, 4) + b(11, 4);
    uint12_t_hw result = (sum_trunc, ap_uint<4>(1));
    return result;
}
