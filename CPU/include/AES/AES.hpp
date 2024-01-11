#pragma once

#if defined (_WIN32) || defined (_WIN64) || defined (WIN32) || defined (WIN64)
    #include <windows.h>
#endif

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "../Common/Helper.hpp"

#ifndef _HELPER_TIMER_H
    #define _HELPER_TIMER_H
    #include "../Common/helper_timer.h"
#endif

// Function Prototypes

// Key Related Functions
void aes_cpu_expand_key(byte_t *round_key, char *key);
void aes_cpu_add_round_key(byte_t round);

// AES Operations
// -> Byte Substitution
// -> Shifting Rows
// -> Mixing Columns
// -> Adding Round Key
void aes_cpu_byte_sub(void);
void aes_cpu_byte_sub_inverse(void);

void aes_cpu_shift_rows(void);
void aes_cpu_shift_rows_inverse(void);

void aes_cpu_mix_columns(void);
void aes_cpu_mix_columns_inverse(void);

byte_t xtime(byte_t x);
byte_t multiply(byte_t x, byte_t y);

// AES Encryption and Decryption Algorithm Implementations
void cipher(void);
void decipher(void);

void aes_cpu_encrypt(byte_t* input, const byte_t* round_key, byte_t* output);
void aes_cpu_decrypt(byte_t* input, const byte_t* round_key, byte_t* output);

// Library Export Wrappers
double aes_ecb_encrypt(const char *input_path, const char *output_path, const char* password);
double aes_ecb_decrypt(const char *input_path, const char *output_path, const char* password);

