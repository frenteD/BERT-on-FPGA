#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "qop.h"

static ap_axiu<256,0,0,0> convWo(v_dt W) {
	#pragma HLS inline
	ap_axiu<256,0,0,0> Wo;
	for(int i=0; i < VDATA_SIZE; i++) {
		Wo.data(W.data[0].width*(i+1)-1, W.data[0].width*i) = W.data[i](W.data[0].width-1, 0);
	}
	return Wo;
}

//从in_tensor中读取数据，转换为ap_axiu<256,0,0,0>类型，写入输出流
static void controlsubv(
	const v_dt *in_tensor,           // Read-only tensor (1024, 14)
	hls::stream<ap_axiu<256,0,0,0>> &ov00_s
	)
{
	v_dt W;
	for(int i=0; i < (Tsize*Veclen)/VDATA_SIZE; i++) {
		W = in_tensor[i];
		ov00_s.write(convWo(W));
	}
}

extern "C" {
void control(
	    const v_dt *in_tensor,           			// Read-only tensor (1024, 14)
		hls::stream<ap_axiu<256,0,0,0> > &ov00_s	// 对输入数据处理（转换类型）后写入流
	    )
{
	//ap_axiu<256,0,0,0>类型数据用于与其他模块通信
	#pragma HLS INTERFACE m_axi port = in_tensor offset = slave bundle = gmem16 
	#pragma HLS INTERFACE s_axilite bundle = cfg port = in_tensor
	#pragma HLS INTERFACE s_axilite bundle = cfg port = return	//将函数control的返回值打包在cfg逻辑组中
	#pragma HLS INTERFACE axis register_mode=both register port=ov00_s
	controlsubv(in_tensor, ov00_s);
}
}

