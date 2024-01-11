#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <bitset>
#include <random>
#include <filesystem>
#include "Macros.hpp"
#include "hmac.hpp"

using namespace std;

// I/O Releated Functions 
size_t read_file(const char *input_file, byte_t *output_buffer, uintmax_t input_data_size);
size_t write_file(const char *output_file, byte_t *data, uintmax_t output_data_size);

string get_hash(const char* input);

void writeCryptFile(
	string file_path,
	string password,
	unsigned char* ciphertext, 
	unsigned int ciphertext_length
);                                                            

bool readCryptFile(
	string file_path, 
	string password, 
	unsigned char **ciphertext, 
	int *ciphertext_length,
	int *error
);
