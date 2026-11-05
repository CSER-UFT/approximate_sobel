#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
/*
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "hls_stream.h"
#include "ap_int.h"

#include "templates/add_exact.h"
#include "templates/add_trunc0_1.h"
#include "templates/add_trunc0_2.h"
#include "templates/add_trunc0_3.h"
#include "templates/add_trunc0_4.h"
#include "templates/add_trunc0_5.h"
#include "templates/add_trunc1_1.h"
#include "templates/add_trunc1_2.h"
#include "templates/add_trunc1_3.h"
#include "templates/add_trunc1_4.h"
#include "templates/add_trunc1_5.h"

#define MAX_IMAGE_WIDTH     512
#define MAX_IMAGE_HEIGHT    512

#define FILTER_V_SIZE		3
#define FILTER_H_SIZE		3

#ifndef MIN
#define MIN(a,b) ((a<b)?a:b)
#endif

typedef unsigned char      		U8;
typedef unsigned short     		U16;
typedef unsigned int       		U32;

typedef signed char        		I8;
typedef signed short       		I16;
typedef signed int         		I32;

void ReadFromMem(
        unsigned short       width,
        unsigned short       height,
        unsigned short       stride,
        const unsigned char *src,
        hls::stream<U8>     &pixel_stream )
{
    stride = (stride/64)*64; // Makes compiler see that stride is a multiple of 64, enables auto-widening
    unsigned offset = 0;
    unsigned x = 0;
    read_image: for (int n = 0; n < height*stride; n++) {
        U8 pix = src[n];
        if (x<width) pixel_stream.write( pix );
        if (x==(stride-1)) x=0; else x++;
     }
}


void WriteToMem(
        unsigned short       width,
        unsigned short       height,
        unsigned short       stride,
        hls::stream<U8>     &pixel_stream,
        unsigned char       *dst)
{
    assert(stride <= MAX_IMAGE_WIDTH);
    assert(height <= MAX_IMAGE_HEIGHT);
    assert(stride%64 == 0);

    stride = (stride/64)*64; // Makes compiler see that stride is a multiple of 64, enables auto-widening
    unsigned offset = 0;
    unsigned x = 0;
    write_image: for (int n = 0; n < height*stride; n++) {
        U8 pix = (x<width) ? pixel_stream.read() : 0;
        dst[n] = pix;
        if (x==(stride-1)) x=0; else x++;
    }
}


struct window {
    U8 pix[FILTER_V_SIZE][FILTER_H_SIZE];
};


void Window2D(
        unsigned short        width,
        unsigned short        height,
        hls::stream<U8>      &pixel_stream,
        hls::stream<window>  &window_stream)
{
    // Line buffers - used to store [FILTER_V_SIZE-1] entire lines of pixels
    U8 LineBuffer[FILTER_V_SIZE-1][MAX_IMAGE_WIDTH];
#pragma HLS ARRAY_PARTITION variable=LineBuffer dim=1 complete
#pragma HLS DEPENDENCE variable=LineBuffer inter false
#pragma HLS DEPENDENCE variable=LineBuffer intra false

    // Sliding window of [FILTER_V_SIZE][FILTER_H_SIZE] pixels
    window Window;

    unsigned col_ptr = 0;
    unsigned ramp_up = width*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2;
    unsigned num_pixels = width*height;
    unsigned num_iterations = num_pixels + ramp_up;

    const unsigned max_iterations = MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT + MAX_IMAGE_WIDTH*((FILTER_V_SIZE-1)/2)+(FILTER_H_SIZE-1)/2;

    // Iterate until all pixels have been processed
    update_window: for (int n=0; n<num_iterations; n++)
    {
#pragma HLS LOOP_TRIPCOUNT max=max_iterations
#pragma HLS PIPELINE II=1

        // Read a new pixel from the input stream
        U8 new_pixel = (n<num_pixels) ? pixel_stream.read() : 0;

        // Shift the window and add a column of new pixels from the line buffer
        for(int i = 0; i < FILTER_V_SIZE; i++) {
            for(int j = 0; j < FILTER_H_SIZE-1; j++) {
                Window.pix[i][j] = Window.pix[i][j+1];
            }
            Window.pix[i][FILTER_H_SIZE-1] = (i<FILTER_V_SIZE-1) ? LineBuffer[i][col_ptr] : new_pixel;
        }

        // Shift pixels in the column of pixels in the line buffer, add the newest pixel
        for(int i = 0; i < FILTER_V_SIZE-2; i++) {
            LineBuffer[i][col_ptr] = LineBuffer[i+1][col_ptr];
        }
        LineBuffer[FILTER_V_SIZE-2][col_ptr] = new_pixel;

        // Update the line buffer column pointer
        if (col_ptr==(width-1)) {
            col_ptr = 0;
        } else {
            col_ptr++;
        }

        // Write output only when enough pixels have been read the buffers and ramped-up
        if (n>=ramp_up) {
            window_stream.write(Window);
        }

    }
}

ap_int<16> filterGx(const window w, ap_int<16> m) {
#pragma HLS pipeline II=1
	ap_int<16> x1, x2, x3, x4, x5, x6, out;
	x1 = "add1"(w.pix[0][0], w.pix[2][0]);
	x2 = w.pix[1][0] * m;
	x3 = "add2"(x1, x2);
	x4 = "add3"(w.pix[0][2], w.pix[2][2]);
	x5 = w.pix[1][2] * m;
	x6 = "add4"(x4, x5);
	out = x3 - x6;

	return (out < 0) ? (ap_int<16>)-out: out;
}

ap_int<16> filterGy(const window w, ap_int<16> m) {
#pragma HLS pipeline II=1
	ap_int<16> x1, x2, x3, x4, x5, x6, out;
	x1 = "add1"(w.pix[0][0], w.pix[0][2]);
	x2 = (w.pix[0][1] * m);
	x3 = "add2"(x1, x2);
	x4 = "add3"(w.pix[2][0], w.pix[2][2]);
	x5 = (w.pix[2][1] * m);
	x6 = "add4"(x4, x5);
	out = x3 - x6;

	return (out < 0) ? (ap_int<16>)-out: out;
}

ap_int<16> addall(const ap_int<16> gx, const ap_int<16> gy) {
	return "add5"(gx, gy);
}

void Filter2D(
        ap_int<16> m,
        unsigned short       width,
        unsigned short       height,
        hls::stream<window> &window_stream,
		hls::stream<U8>     &pixel_stream )
{

    // Process the incoming stream of pixel windows
    apply_filter: for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
#pragma HLS UNROLL factor=2
            // Read a 2D window of pixels
            window w = window_stream.read();

            // Apply filter to the 2D window
            int sum = 0;

            // Set output to 0 if the 3x3 receptive field is out of bound.
			if ((i > 0) & (i < height - 1) & (j > 0) & (j < width - 1)) {
#pragma HLS DATAFLOW
				short gx_sum, gy_sum;
				// Apply the sobel filter at the current "receptive field".
				gx_sum = filterGx(w, m);
				gy_sum = filterGy(w, m);

				sum = addall(gx_sum, gy_sum);
			}
            // Normalize result
            unsigned char outpix = MIN(sum, 255);

            // Write the output pixel
            pixel_stream.write(outpix);
        }
    }
}


extern "C" {

void sobel(
        const unsigned char  src[MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT],
        unsigned char        dst[MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT], 
        unsigned short height, 
        unsigned short width,
        ap_int<16> m)
  {


#pragma HLS DATAFLOW
	// Stream of pixels from kernel input to filter, and from filter to output
    hls::stream<U8>      pixel_stream;
    hls::stream<window>  window_stream; // Set FIFO depth to 0 to minimize resources
    hls::stream<U8>      output_stream;

	// Read image data from global memory over AXI4 MM, and stream pixels out
    ReadFromMem(width, height, width, src, pixel_stream);

    // Read incoming pixels and form valid HxV windows
    Window2D(width, height, pixel_stream, window_stream);

	// Process incoming stream of pixels, and stream pixels out
	Filter2D(m, width, height, window_stream, output_stream);

	// Write incoming stream of pixels and write them to global memory over AXI4 MM
	WriteToMem(width, height, width, output_stream, dst);
  }

}
