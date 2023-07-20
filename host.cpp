#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <cmath>
#include "xcl2.hpp"

// HBM Banks requirements
#define MAX_HBM_BANKCOUNT 32
#define BANK_NAME(n) n | XCL_MEM_TOPOLOGY
const int bank[MAX_HBM_BANKCOUNT] = {
    BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
    BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
    BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
    BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
    BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
    BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
    BANK_NAME(30), BANK_NAME(31)};

const int map[8] = {
  BANK_NAME(0),
  BANK_NAME(4),
  BANK_NAME(8),
  BANK_NAME(12),
  BANK_NAME(16),
  BANK_NAME(20),
  BANK_NAME(24),
  BANK_NAME(26)
};

enum {
  Npk = 8,
  Wr = 3*1024,
  Wc = 1024,
  Vr = 1024,
  Vc = 14,
  Niter= 1,
};

int swres[Wr][Vc];

typedef char Dt;


// Function for verifying results
bool verify(std::vector<int, aligned_allocator<int>>& source_hw_wb_results) 
{
    bool check = true;
    for (int i = 0; i < Wr*Vc; i++) {
      if (source_hw_wb_results[i] != 0) {
        std::cout << "i = " << i << "; v = " << source_hw_wb_results[i] << std::endl;
      }
    }
    return check;
}

double run_krnl(cl::Context& context,
                cl::CommandQueue& q,
                cl::Kernel& kernel,
                std::vector<Dt, aligned_allocator<Dt>>& source_w,
                std::vector<Dt, aligned_allocator<Dt>>& source_v,
                std::vector<int, aligned_allocator<int>>& source_hw_wb_results) 
{
    cl_int err;

    // For Allocating Buffer to specific Global Memory PC, user has to use
    // cl_mem_ext_ptr_t
    // and provide the PCs
    //cl_mem_ext_ptr_t inBufExt1, inBufExt2, outBufExt1, outBufExt2;

    std::vector<cl_mem_ext_ptr_t> inBufExtw(Npk);
    std::vector<cl_mem_ext_ptr_t> inBufExtv(1);
    std::vector<cl_mem_ext_ptr_t> outBufExtwb(1);

    std::vector<cl::Buffer> buffer_inputw(Npk);
    std::vector<cl::Buffer> buffer_inputv(1);
    std::vector<cl::Buffer> buffer_output_wb(1);
    
    for(int i = 0; i < Npk; i++) {
      inBufExtw[i].obj = source_w[i].data();
      inBufExtw[i].param = 0;
      inBufExtw[i].flags = map[i];
      // if(is_sw_emulation)
      //   inBufExtw[i].flags = bank[0];
    }

    inBufExtv[0].obj = source_v.data();
    inBufExtv[0].param = 0;
    inBufExtv[0].flags = bank[14];
    // if(is_sw_emulation) {
    //   inBufExtv[0].flags = bank[0];
    // }

    outBufExtwb[0].obj = source_hw_wb_results.data();
    outBufExtwb[0].param = 0;
    outBufExtwb[0].flags = bank[14];
    // if(is_sw_emulation) {
    //   outBufExtwb[0].flags = bank[0];
    // }

    // These commands will allocate memory on the FPGA. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    
    // Creating Buffers 
    for(int i = 0; i < Npk; i++) {
      OCL_CHECK(err, buffer_inputw[i] = cl::Buffer(
                        context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                        sizeof(Dt) * 3 * 1024 * 1024/Npk, &inBufExtw[i], &err));
      // Copy input data to Device Global Memory
      OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_inputw[i]}, 0 /* 0 means from host*/));
    }
    OCL_CHECK(err, buffer_inputv[0] = cl::Buffer(
                        context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                        sizeof(Dt) * 1024 * 14, &inBufExtv[0], &err));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_inputv[0]}, 0 /* 0 means from host*/));

    OCL_CHECK(err, buffer_output_wb[0] = cl::Buffer(
                        context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                        sizeof(int) * 3 * 1024 * 14, &outBufExtwb[0], &err));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_wb[0]}, 0 /* 0 means from host*/));
    
    // Setting the feeder Arguments
    OCL_CHECK(err, err = krnl_feeder.setArg(0, buffer_inputv[0]));
    for(int j = 0; j < Npk; j++) {
      OCL_CHECK(err, err = krnl_feeder.setArg(1+j, buffer_inputw[j]));
    }
    int outshiftscale = 0;
    OCL_CHECK(err, err = krnl_feeder.setArg(9, buffer_output_wb[0]));
    OCL_CHECK(err, err = krnl_feeder.setArg(10, (char)3));
    OCL_CHECK(err, err = krnl_feeder.setArg(11, (char)14));
    OCL_CHECK(err, err = krnl_feeder.setArg(12, outshiftscale));

    q.finish();


    std::chrono::duration<double> kernel_time(0);

    auto kernel_start = std::chrono::high_resolution_clock::now();
    for(int iter=0; iter<Niter; iter++) {
      // Invoking the kernel
      OCL_CHECK(err, err = q.enqueueTask(krnl_feeder));
      q.finish();
    }
    std::cout << "q finished" << std::endl;
    q.finish();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_wb[0]}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    std::cout << "readback finished" << std::endl;
    
    return kernel_time.count();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <XCLBIN> \n", argv[0]);
        return -1;
    }
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel kernel_vadd;
    std::string binaryFile = argv[1];

    // The get_xil_devices will return vector of Xilinx Devices
    auto devices = xcl::get_xil_devices();

    // read_binary_file() command will find the OpenCL binary file created using
    // the V++ compiler load into OpenCL Binary and return pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);

    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, kernel_vadd = cl::Kernel(program, "krnl_vadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    /******************init source data******************/

    //Dt --> char
    std::vector<Dt, aligned_allocator<Dt>> source_w[Npk];
    std::vector<Dt, aligned_allocator<Dt>> source_v(sizeof(Dt)*14*1024);
    std::vector<int, aligned_allocator<int>> source_hw_wb_results(sizeof(int)*3*14*1024);
    
    for(int i = 0; i < Npk; i++) {  //source_w包含Npk个子矩阵
      source_w[i].resize(sizeof(Dt)*3*1024*1024/Npk);
    }
    for(int i = 0; i < Npk; i++) {  //为W生成测试数据
      std::fill(source_w[i].begin(), source_w[i].end(), 0);
      //std::generate(source_w[i].begin(), source_w[i].end(), std::rand);
    }

    std::fill(source_v.begin(), source_v.end(), 8); //为V生成测试数据
    //std::generate(source_v.begin(), source_v.end(), std::rand);

    source_w[0][10] = 5;
    source_w[0][1024+11] = 7;

    //source_v[10] = ;

    // Initializing output vectors to zero
    std::fill(source_hw_wb_results.begin(), source_hw_wb_results.end(), 0);


    /******************init test data******************/

    std::chrono::duration<double> host_time(0);
    auto host_start = std::chrono::high_resolution_clock::now();

    // transfer input vector to hashcode
    for(int i = 0; i < V_SIZE; i++) {
        for(int j = 0; j < HASH_SIZE; j++) {
            float tmp = 0.0;
            for(int n = 0; n < NUM_WORDS; n++) {
                float in_tmp = source_input[i][n];
                float ran_tmp = source_random[j][n];
                tmp += ran_tmp * in_tmp;
            }
            if (tmp > 0) input_hashcode[i][j] = 1;
            else input_hashcode[i][j] = 0;
        }
    }

    // store in hashtable
    for (int i = 0; i < BUCKET_NUM; i++) {  //init hashtable 1 & 2
        for (int j = 0; j < V_SIZE; j++) {
            source_sw_hashtable1[i][j] = 0;
            source_sw_hashtable2[i][j] = 0;
            source_hw_hashtable1[i][j] = 0;
            source_hw_hashtable2[i][j] = 0;
        }
    }
    for (int i = 0; i < V_SIZE; i++) {
        h_dt tmp_hashcode = input_hashcode[i];
        int h1 = 0;
        int h2 = 0;
        int flag = 1;
        // compute hashcode1 & hashcode2
        for (int j = 0; j < 4; j++) {
            // hashcode_a += (tmp_hashcode[j] << j);
            if(tmp_hashcode[j] == 1)
                h1 += flag;
            flag = flag * 2;
        }
        flag = 1;
        for (int j = 4; j < HASH_SIZE; j++) {
            if(tmp_hashcode[j] == 1)
                h2 += flag;
            flag = flag * 2;
        }
        source_sw_hashtable1[h1][i] = 1;
        source_sw_hashtable2[h2][i] = 1;
        // std::cout << "source_sw_hashtable1[" << h1 << "][" << i << "]\n";
        // std::cout << "source_sw_hashtable2[" << h2 << "][" << i << "]\n";
    }

    auto host_end = std::chrono::high_resolution_clock::now();
    host_time = std::chrono::duration<double>(host_end - host_start);
    std::cout << "----------------host time in second: " << host_time.count() << "----------------\n";

    /*********************run kernel********************/

    double kernel_time_in_sec = 0, result = 0;
    bool match = true;
    //const int numBuf = 32; // Since 32 buffers are being used
    //int pc_assign[numBuf];
    
    // for (int j = 0; j < numBuf; j++) {
    //     pc_assign[j] = pc[j + 1];
    // }

    kernel_time_in_sec = run_krnl(context, q, krnl_feeder, source_w, source_v, source_hw_wb_results);
    match = verify(source_hw_wb_results);
    
    std::cout << "******************kernel time in second: " << kernel_time_in_sec << "******************\n";

    std::cout << (match ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
