#pragma once

#if defined (_WIN32) || defined (_WIN64) || defined (WIN32) || defined (WIN64)
    #include <windows.h>
#endif

#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "../Common/Helper.hpp"

using namespace std;

// Function Prototypes

// Key Function
void aes_ocl_expand_key(byte_t *round_key, char *key);

// Library Export Wrappers
double aes_ocl_encrypt(const char *input_path, const char *output_path, const char *password, const char* kernel_path);
double aes_ocl_decrypt(const char *input_path, const char *output_path, const char *password, const char* kernel_path);
