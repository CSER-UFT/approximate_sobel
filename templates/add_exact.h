#include "ap_int.h"
typedef ap_uint<12> uint12_t_hw;

uint12_t_hw add_exact(uint12_t_hw a, uint12_t_hw b) {
    #pragma HLS INLINE OFF
    return a + b;
}
