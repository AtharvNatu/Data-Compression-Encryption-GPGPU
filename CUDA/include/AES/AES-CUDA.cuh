#pragma once

#if defined (_WIN32) || defined (_WIN64) || defined (WIN32) || defined (WIN64)
    #include <windows.h>
#endif

#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "../Common/Helper.hpp"

#ifndef _HELPER_TIMER_H
    #define _HELPER_TIMER_H
    #include "../Common/helper_timer.h"
#endif

using namespace std;

// Function Prototypes

// CUDA Helper Functions
void cuda_mem_alloc(void **dev_ptr, size_t size);
void cuda_mem_copy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
void cuda_mem_free(void **dev_ptr);

// Key Related Functions
void aes_cuda_expand_key(byte_t *round_key, char *key);
__device__ void aes_cuda_add_round_key(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE], byte_t *aes_round_key, byte_t round);

// AES Operations
// -> Byte Substitution
// -> Shifting Rows
// -> Mixing Columns
// -> Adding Round Key
__device__ void aes_cuda_byte_sub(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE]);
__device__ void aes_cuda_byte_sub_inverse(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE]);

__device__ void aes_cuda_shift_rows(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE]);
__device__ void aes_cuda_shift_rows_inverse(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE]);

__device__ void aes_cuda_mix_columns(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE]);
__device__ void aes_cuda_mix_columns_inverse(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE]);

__device__ byte_t xtime(byte_t x);

// AES Encryption and Decryption Algorithm Implementations
__device__ void cipher(byte_t input[AES_LENGTH], byte_t output[AES_LENGTH], byte_t* round_key);
__device__ void inverse_cipher(byte_t input[AES_LENGTH], byte_t output[AES_LENGTH], byte_t* round_key);

__device__ void transform1D(byte_t input[AES_LENGTH], byte_t state[AES_STATE_SIDE][AES_STATE_SIDE]);
__device__ void transform2D(byte_t output[AES_LENGTH], byte_t state[AES_STATE_SIDE][AES_STATE_SIDE]);

// CUDA Kernels
__global__ void aes_cuda_ecb_encrypt(byte_t* d_plaintext, byte_t* d_ciphertext, byte_t* d_round_key, int data_size);
__global__ void aes_cuda_ecb_decrypt(byte_t* d_ciphertext, byte_t* d_plaintext, byte_t* d_round_key, int data_size);

// Library Export Wrappers
double aes_cuda_encrypt(const char *input_path, const char *output_path, const char *password);
double aes_cuda_decrypt(const char *input_path, const char *output_path, const char *password);
