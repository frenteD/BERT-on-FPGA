#include <hls_vector.h>
#include <hls_stream.h>
#include <ap_int.h>
#include <iostream>
#include <cmath>
#include "ap_axi_sdata.h"
#include "ap_fixed.h"
#include "qop.h"

//在qop.h中：
/*
    #define VDATA_SIZE 16
    Tsize=1024

    typedef ap_int<8> Dt; 
    typedef struct v_inWrrtype{ //v_arr：一个长为16的8位整型数组
        Dt data[VDATA_SIZE]; 
    } v_arr;

    typedef ap_int<32> It;
    typedef struct v_wbtype{	//wb_arr：一个长为16的32位整型数组
        It data[VDATA_SIZE]; 
    } wb_arr;
*/


typedef struct v_in128 {    //一个长为128的8位整型数组
	Dt data[Tsize/8];	//ap_int<8> data[128]; 
} v_arr128;

enum {
	Bsize = Tsize/8,	//Bsize=Tsize/8=1024/8=128
	BsizeP2 = Tp2-3, 	//BsizeP2=Tp2-3=7
};


//读取输入
void hbmV(
	ap_uint<4> nmat,
	ap_uint<8> veclen,
    const v_arr *inV,            
	hls::stream<v_arr128 > &oV_s0,
	hls::stream<v_arr128 > &oV_s1,
	hls::stream<v_arr128 > &oV_s2,
	hls::stream<v_arr128 > &oV_s3,
	hls::stream<v_arr128 > &oV_s4,
	hls::stream<v_arr128 > &oV_s5,
	hls::stream<v_arr128 > &oV_s6,
	hls::stream<v_arr128 > &oV_s7
	)
{
	v_arr128 v[Veclen][8];
	#pragma HLS array_partition variable=v dim=2
	#pragma HLS array_partition variable=v dim=3

	for(int q = 0; q < veclen; q++) {
		#pragma HLS loop_tripcount min=8 max=1024
		for(int i=0; i < 64; i++) {
			#pragma HLS pipeline II=1
			v_arr t;
			t = inV[i+q*8*8];
			for(int z=0; z < VDATA_SIZE; z++)
				v[q][i>>3].data[(i%8)*16+z] = t.data[z];
		}
	}

	for(int iter=0; iter < Tsize*nmat; iter++) {
		#pragma HLS loop_tripcount min=1024 max=8192
		for(int q = 0; q < veclen; q++) {
			#pragma HLS loop_tripcount min=14 max=128
			#pragma HLS pipeline II=1
			oV_s0.write(v[q][0]);
			oV_s1.write(v[q][1]);
			oV_s2.write(v[q][2]);
			oV_s3.write(v[q][3]);
			oV_s4.write(v[q][4]);
			oV_s5.write(v[q][5]);
			oV_s6.write(v[q][6]);
			oV_s7.write(v[q][7]);
		}
	}
}

//读取权重
void consume(
	ap_uint<4> nmat,
    const v_arr *inW,            // Read-Only Weights
	hls::stream<v_arr128 > &oW_s
	)
{
	v_arr v[8];
	v_arr128 oW;
	#pragma HLS array_partition variable=v dim=0
	#pragma HLS array_partition variable=oW dim=0

	for(int iter=0; iter < Tsize*nmat*Tsize/VDATA_SIZE/Mempaths; iter++) {
		#pragma HLS loop_tripcount min=8192 max=65536
		v[iter%8] = inW[iter];
		if((iter%8) == 7) {
			makeone(v, oW);
			oW_s.write(oW);
		}
	}
}


//将输入的向量复制多次，并输出到指定流
void replicate(
	ap_uint<4> nmat,
	ap_uint<8> veclen,
	hls::stream<v_arr128 > &i_s,
	hls::stream<v_arr128 > &o_s
	)
{
	v_arr128 v;
	for(int iter=0; iter < Tsize*nmat; iter++) {
		#pragma HLS loop_tripcount min=1024 max=8192 
		#pragma HLS pipeline II=14
		i_s.read(v);
		for(int i=0; i < veclen; i++) {
			#pragma HLS loop_tripcount min=8 max=1024
			o_s.write(v);
		}
	}
}

// 累加求和
void finalsum(
	ap_uint<4> nmat,
	ap_uint<8> veclen,
	hls::stream<It > &i_s0,
	hls::stream<It > &i_s1,
	hls::stream<It > &i_s2,
	hls::stream<It > &i_s3,
	hls::stream<It > &i_s4,
	hls::stream<It > &i_s5,
	hls::stream<It > &i_s6,
	hls::stream<It > &i_s7,
	hls::stream<It > &o_s
	)
{
	It s[8], o;
	#pragma HLS array_partition variable=s dim=0
	for(int iter=0; iter < Tsize*veclen*nmat; iter++) {
		#pragma HLS loop_tripcount min=1024 max=1048576
		#pragma HLS pipeline II=1
		i_s0.read(s[0]);
		i_s1.read(s[1]);
		i_s2.read(s[2]);
		i_s3.read(s[3]);
		i_s4.read(s[4]);
		i_s5.read(s[5]);
		i_s6.read(s[6]);
		i_s7.read(s[7]);
		o = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
		o_s.write(o);
	}
}


//self-attention
void onemath(
	It inV[Bsize], 
	It W[Bsize], 
	hls::stream<It > &o_s)	//Bsize=128
{
	It imm[BsizeP2+1][Bsize];   //It：32位整型数，imm[8][128]
	#pragma HLS pipeline II=1
	#pragma HLS array_partition variable=imm dim=0	//指定分区，便于并行计算
	l1: for(int b = 0; b < Bsize; b++) {    //inV和W相乘，存储在imm的第一行中
		imm[0][b] = inV[b] * W[b];
	}
	//Mask操作
	for(int s = 0; s < BsizeP2; s++) {	//BsizeP2=7
		for(int k = 0; k < (Bsize>>(s+1)); k++) {   //Bsize=128
			imm[s+1][k] = imm[s][k] + imm[s][k+(Bsize>>(s+1))];
		}
	}
	o_s.write(imm[BsizeP2][0]);
}

//
void batchmath(
	ap_uint<4> nmat,
	ap_uint<8> veclen,
	hls::stream<v_arr128 > &inW_s0,
	hls::stream<v_arr128 > &inV_s0,
	hls::stream<It > &o_s0
	)
{
	v_arr128 V,W;
	It bV[Bsize], bW[Bsize];
	#pragma HLS array_partition variable=bV dim=0
	#pragma HLS array_partition variable=bW dim=0
	#pragma HLS array_partition variable=W dim=1
	#pragma HLS array_partition variable=V dim=1
	for(int i=0; i < veclen*Tsize*nmat; i++) {
		#pragma HLS loop_tripcount min=1024 max=1048576
		#pragma HLS pipeline
		inV_s0.read(V);
		inW_s0.read(W);
		for(int i = 0; i < Bsize; i++) {	//Bsize = 128
			bV[i] = V.data[i];
			bW[i] = W.data[i];
		}
		onemath(bV, bW, o_s0);
	}
}

//处理批计算结果，存储到指定数组（指定位置）
void wb(
	ap_uint<4> nmat,
	ap_uint<8> veclen,
    int shift,
	hls::stream<It > &i_s0,
    wb_arr *o_tensor	//输出张量
    )
{
	It e;
	wb_arr V;
	#pragma HLS array_partition variable=V dim=0
	l_a: for(int i=0; i < (nmat*Tsize*veclen)/VDATA_SIZE; i++) {
		#pragma HLS loop_tripcount min=64 max=65536
		#pragma HLS pipeline II=16
		for(int q=0; q < VDATA_SIZE; q++) {
			i_s0.read(e);			//从输入流中读取数据
			V.data[q] = e >> shift;	//执行位移操作，将结果存储到输出张量中
		}
		o_tensor[i] = V;
	}
}

extern "C" {
    void feeder(
        const v_arr *inV,             // input data
        const v_arr *inW0,            // Read-Only Weights
        const v_arr *inW1,            // Read-Only Weights
        const v_arr *inW2,            // Read-Only Weights
        const v_arr *inW3,            // Read-Only Weights
        const v_arr *inW4,            // Read-Only Weights
        const v_arr *inW5,            // Read-Only Weights
        const v_arr *inW6,            // Read-Only Weights
        const v_arr *inW7,            // Read-Only Weights
        wb_arr *o_tensor,             // weights(copy)
        ap_uint<4> nmat,              // 传递过来的定量，char 3
        ap_uint<8> veclen,            // 传递过来的定量，char 14
        int shift                     // 传递过来的定量，0
    )
    {
        #pragma HLS INTERFACE m_axi port = inV offset = slave bundle = gmem16 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = inW0 offset = slave bundle = gmem0 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = inW1 offset = slave bundle = gmem1 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = inW2 offset = slave bundle = gmem2 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = inW3 offset = slave bundle = gmem3 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = inW4 offset = slave bundle = gmem4 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = inW5 offset = slave bundle = gmem5 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = inW6 offset = slave bundle = gmem6 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = inW7 offset = slave bundle = gmem7 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE m_axi port = o_tensor offset=slave bundle = gmem17 max_read_burst_length=16 max_write_burst_length=16
        #pragma HLS INTERFACE s_axilite bundle = control port = inV
        #pragma HLS INTERFACE s_axilite bundle = control port = inW0
        #pragma HLS INTERFACE s_axilite bundle = control port = inW1
        #pragma HLS INTERFACE s_axilite bundle = control port = inW2
        #pragma HLS INTERFACE s_axilite bundle = control port = inW3
        #pragma HLS INTERFACE s_axilite bundle = control port = inW4
        #pragma HLS INTERFACE s_axilite bundle = control port = inW5
        #pragma HLS INTERFACE s_axilite bundle = control port = inW6
        #pragma HLS INTERFACE s_axilite bundle = control port = inW7
        #pragma HLS INTERFACE s_axilite bundle = control port = o_tensor
        #pragma HLS INTERFACE s_axilite bundle = control port = shift
        #pragma HLS INTERFACE s_axilite bundle = control port = nmat
        #pragma HLS INTERFACE s_axilite bundle = control port = veclen
        #pragma HLS INTERFACE s_axilite bundle = control port = return

        #pragma HLS dataflow

        static hls::stream<v_arr128 > oV_s0;
        static hls::stream<v_arr128 > oV_s1;
        static hls::stream<v_arr128 > oV_s2;
        static hls::stream<v_arr128 > oV_s3;
        static hls::stream<v_arr128 > oV_s4;
        static hls::stream<v_arr128 > oV_s5;
        static hls::stream<v_arr128 > oV_s6;
        static hls::stream<v_arr128 > oV_s7;
        static hls::stream<v_arr128 > oW_s0;
        static hls::stream<v_arr128 > oW_s1;
        static hls::stream<v_arr128 > oW_s2;
        static hls::stream<v_arr128 > oW_s3;
        static hls::stream<v_arr128 > oW_s4;
        static hls::stream<v_arr128 > oW_s5;
        static hls::stream<v_arr128 > oW_s6;
        static hls::stream<v_arr128 > oW_s7;
        static hls::stream<v_arr128 > oWa_s0;
        static hls::stream<v_arr128 > oWa_s1;
        static hls::stream<v_arr128 > oWa_s2;
        static hls::stream<v_arr128 > oWa_s3;
        static hls::stream<v_arr128 > oWa_s4;
        static hls::stream<v_arr128 > oWa_s5;
        static hls::stream<v_arr128 > oWa_s6;
        static hls::stream<v_arr128 > oWa_s7;
        static hls::stream<It > o_i0;
        static hls::stream<It > o_i1;
        static hls::stream<It > o_i2;
        static hls::stream<It > o_i3;
        static hls::stream<It > o_i4;
        static hls::stream<It > o_i5;
        static hls::stream<It > o_i6;
        static hls::stream<It > o_i7;
        static hls::stream<It > o_wb;

        #pragma HLS STREAM variable=oV_s0 depth=3	
        #pragma HLS STREAM variable=oV_s1 depth=3	
        #pragma HLS STREAM variable=oV_s2 depth=3	
        #pragma HLS STREAM variable=oV_s3 depth=3	
        #pragma HLS STREAM variable=oV_s4 depth=3	
        #pragma HLS STREAM variable=oV_s5 depth=3	
        #pragma HLS STREAM variable=oV_s6 depth=3	
        #pragma HLS STREAM variable=oV_s7 depth=3	
        #pragma HLS STREAM variable=oW_s0 depth=2	
        #pragma HLS STREAM variable=oW_s1 depth=2	
        #pragma HLS STREAM variable=oW_s2 depth=2	
        #pragma HLS STREAM variable=oW_s3 depth=2	
        #pragma HLS STREAM variable=oW_s4 depth=2	
        #pragma HLS STREAM variable=oW_s5 depth=2	
        #pragma HLS STREAM variable=oW_s6 depth=2	
        #pragma HLS STREAM variable=oW_s7 depth=2	
        #pragma HLS STREAM variable=oWa_s0 depth=2	
        #pragma HLS STREAM variable=oWa_s1 depth=2	
        #pragma HLS STREAM variable=oWa_s2 depth=2	
        #pragma HLS STREAM variable=oWa_s3 depth=2	
        #pragma HLS STREAM variable=oWa_s4 depth=2	
        #pragma HLS STREAM variable=oWa_s5 depth=2	
        #pragma HLS STREAM variable=oWa_s6 depth=2	
        #pragma HLS STREAM variable=oWa_s7 depth=2	
        #pragma HLS STREAM variable=o_i0 depth=2	
        #pragma HLS STREAM variable=o_i1 depth=2	
        #pragma HLS STREAM variable=o_i2 depth=2	
        #pragma HLS STREAM variable=o_i3 depth=2	
        #pragma HLS STREAM variable=o_i4 depth=2	
        #pragma HLS STREAM variable=o_i5 depth=2	
        #pragma HLS STREAM variable=o_i6 depth=2	
        #pragma HLS STREAM variable=o_i7 depth=2	
        #pragma HLS STREAM variable=o_wb depth=2	

        //读取输入数据
        hbmV(nmat, veclen, inV, oV_s0, oV_s1, oV_s2, oV_s3, oV_s4, oV_s5, oV_s6, oV_s7);
        //读取权重矩阵
        consume(nmat, inW0, oW_s0);	//inW0 -> oW_s0
        consume(nmat, inW1, oW_s1);
        consume(nmat, inW2, oW_s2);
        consume(nmat, inW3, oW_s3);
        consume(nmat, inW4, oW_s4);
        consume(nmat, inW5, oW_s5);
        consume(nmat, inW6, oW_s6);
        consume(nmat, inW7, oW_s7);
        //复制权重矩阵
        replicate(nmat, veclen, oW_s0, oWa_s0);	//oW_s0 -> oWa_s0
        replicate(nmat, veclen, oW_s1, oWa_s1);
        replicate(nmat, veclen, oW_s2, oWa_s2);
        replicate(nmat, veclen, oW_s3, oWa_s3);
        replicate(nmat, veclen, oW_s4, oWa_s4);
        replicate(nmat, veclen, oW_s5, oWa_s5);
        replicate(nmat, veclen, oW_s6, oWa_s6);
        replicate(nmat, veclen, oW_s7, oWa_s7);
        //批量计算：输入&权重
        batchmath(nmat, veclen, oWa_s0, oV_s0, o_i0);	//权重oWa_s0，输入向量oV_s0，结果输出流o_i0
        batchmath(nmat, veclen, oWa_s1, oV_s1, o_i1);
        batchmath(nmat, veclen, oWa_s2, oV_s2, o_i2);
        batchmath(nmat, veclen, oWa_s3, oV_s3, o_i3);
        batchmath(nmat, veclen, oWa_s4, oV_s4, o_i4);
        batchmath(nmat, veclen, oWa_s5, oV_s5, o_i5);
        batchmath(nmat, veclen, oWa_s6, oV_s6, o_i6);
        batchmath(nmat, veclen, oWa_s7, oV_s7, o_i7);
        //将多个流中的数据累加求和，输出到linear layer
        finalsum(nmat, veclen, o_i0, o_i1, o_i2, o_i3, o_i4, o_i5, o_i6, o_i7, o_wb);
        //存储批计算结果
        wb(nmat, veclen, shift, o_wb, o_tensor);
    }

} // extern C
