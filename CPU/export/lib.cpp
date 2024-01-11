#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)

    #include <windows.h>
	#include "../include/AES/AES.hpp"
	#include "../include/Huffman/compress.hpp"
	#include "../include/Huffman/decompress.hpp"
	#include "../include/Huffman/utils.hpp"

	BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) 
    {
        switch (ul_reason_for_call) 
        {
            case DLL_PROCESS_ATTACH:
            case DLL_THREAD_ATTACH:
            case DLL_THREAD_DETACH:
            case DLL_PROCESS_DETACH:
                break;
        }
        return TRUE;
    }

	extern "C" __declspec(dllexport) double aes_cpu_encrypt_ffi(const char* input_path, const char* output_path, const char* password)
	{
		return aes_ecb_encrypt(input_path, output_path, password);
	}
	
	extern "C" __declspec(dllexport) double aes_cpu_decrypt_ffi(const char* input_path, const char* output_path, const char* password)
	{
		return aes_ecb_decrypt(input_path, output_path, password);
	}

	extern "C" __declspec(dllexport) double aes_cpu_encrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
	{
		string compressed_value[256] = {""};
		string input_file_name = input_path;
		ullint input_file_size, compressed_size_wo_header, compressed_file_size;
		unsigned char *compressed_file_buffer;
		input_file_size = getFileSizeBytes(input_path);
		map<char, ullint> char_freqs = coutCharFrequency(input_path, input_file_size);
		HuffNode *const root = generateHuffmanTree(char_freqs);
		string buff = "";
		compressed_size_wo_header = storeHuffmanValue(root, buff, compressed_value);
		

		double compression_time = compress(
			input_file_name,
			&compressed_file_buffer,
			compressed_size_wo_header,
			input_file_size,
			&compressed_file_size,
			compressed_value
		);
		
		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		string user_key;
		string input_file, output_file = output_path, output_file_name;
		byte_t *ciphertext = NULL;
		byte_t ciphertext_block[AES_BLOCK_SIZE];
		byte_t round_key[176];
		char key[17];

		// Reading input and output file paths
		input_file = input_path;
		filesystem::path output_file_path = filesystem::path(input_file).filename();
		output_file_name = output_file_path.string();

		#if (OS == 1)
			output_file = output_path + ("\\" + output_file_name) + ".enc";
		#elif (OS == 2)
			output_file = output_path + ("/" + output_file_name) + ".enc";
		#endif

		if (!filesystem::exists(input_file))
		{
			cerr << endl << "Error : Invalid Input File ... Exiting !!!" << endl;
			exit(AES_FAILURE);
		}

		int file_length = (int) compressed_file_size;
		uintmax_t plaintext_blocks = (file_length + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
		ciphertext = (byte_t*)malloc(plaintext_blocks * AES_BLOCK_SIZE);

		if (ciphertext == NULL)
		{
			cerr << endl << "Error : Failed To Allocate Memory To Output File Buffer ... Exiting !!!" << endl;
			exit(AES_FAILURE);
		}

		string key_str = get_hash(password);
		aes_cpu_expand_key(round_key, strcpy(key, key_str.c_str()));

		StopWatchInterface *timer = NULL;
		sdkCreateTimer(&timer);
		sdkStartTimer(&timer);
		{
			for (uintmax_t i = 0; i < plaintext_blocks; i++)
			{
				aes_cpu_encrypt(compressed_file_buffer + i * AES_BLOCK_SIZE, round_key, ciphertext_block);
				memcpy(ciphertext + i * AES_BLOCK_SIZE, ciphertext_block, sizeof(byte_t) * AES_BLOCK_SIZE);
			}
		}
		sdkStopTimer(&timer);
		double encryption_time = (double)sdkGetTimerValue(&timer);

		// Write Encrypted Data
		// write_file(output_file.c_str(), ciphertext, AES_BLOCK_SIZE * plaintext_blocks);
		writeCryptFile(output_file, key_str, ciphertext, AES_BLOCK_SIZE * plaintext_blocks);

		sdkDeleteTimer(&timer);
		timer = NULL;

		free(ciphertext);
		ciphertext = NULL;

		free(compressed_file_buffer);
		compressed_file_buffer = NULL;

		return encryption_time + compression_time;
	}
	
	extern "C" __declspec(dllexport) double aes_cpu_decrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
	{
		// Variable Declarations
		string user_key;
		string input_file, output_file, output_file_name;
		int output_file_name_index = 0, input_file_length=0;
		byte_t *plaintext = NULL;
		byte_t *ciphertext = NULL;
		byte_t plaintext_block[AES_BLOCK_SIZE];
		byte_t round_key[176];
		char key[17];
		
		// Code

		// Reading input and output file paths
		input_file = input_path;
		output_file_name_index = input_file.find("enc") - 1;
		output_file = input_file.substr(0, output_file_name_index);

		#if (OS == 1)
			output_file = output_path + ("\\" + filesystem::path(input_file.substr(0, output_file_name_index)).filename().string());
		#elif (OS == 2)
			output_file = output_path + ("/" + filesystem::path(input_file.substr(0, output_file_name_index)).filename().string());
		#endif

		if (!filesystem::exists(input_file))
		{
			cerr << endl << "Error : Invalid Input File ... Exiting !!!" << endl;
			exit(AES_FAILURE);
		}

		// Allocate memory to input buffer
		// int file_length = filesystem::file_size(input_file);
		// ciphertext = (byte_t *)malloc(sizeof(byte_t) * file_length);
		// if (ciphertext == NULL)
		// {
		// 	cerr << endl << "Error : Failed To Allocate Memory To Input File Buffer ... Exiting !!!" << endl;
		// 	exit(AES_FAILURE);
		// }

		// Read Encrypted Input File
		// uintmax_t bytes_read = read_file(input_file.c_str(), ciphertext, file_length);
		// if (bytes_read <= 0)
		// {
		// 	cerr << endl << "Error : Empty File ... Please Select A Valid File ... Exiting !!!" << endl;
		// 	exit(AES_FAILURE);
		// }
		
		string key_str = get_hash(password);
		int error_type = 0;
		if(!readCryptFile(input_file, key_str, &ciphertext, &input_file_length, &error_type)){
			return error_type;
		}
		
		uintmax_t ciphertext_blocks = (input_file_length + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
		plaintext = (byte_t*)malloc(ciphertext_blocks * AES_BLOCK_SIZE);
		if (plaintext == NULL)
		{
			cerr << endl << "Error : Failed To Allocate Memory To Output File Buffer ... Exiting !!!" << endl;
			exit(AES_FAILURE);
		}

		aes_cpu_expand_key(round_key, strcpy(key, key_str.c_str()));

		StopWatchInterface *timer = NULL;
		sdkCreateTimer(&timer);
		sdkStartTimer(&timer);
		{
			for (uintmax_t i = 0; i < ciphertext_blocks; i++)
			{
				aes_cpu_decrypt(ciphertext + i * AES_BLOCK_SIZE, round_key, plaintext_block);
				memcpy(plaintext + i * AES_BLOCK_SIZE, plaintext_block, sizeof(byte_t) * AES_BLOCK_SIZE);
			}
		}
		sdkStopTimer(&timer);
		double decryption_time = (double)sdkGetTimerValue(&timer);

		double decompression_time = decompress(plaintext, output_file, input_file_length);

		sdkDeleteTimer(&timer);
		timer = NULL;
		
		free(plaintext);
		plaintext = NULL;

		free(ciphertext);
		ciphertext = NULL;

		return decryption_time + decompression_time;
	}
	

#elif defined(__linux__)

	#include "../include/AES/AES.hpp"
	#include "../include/Huffman/compress.hpp"
	#include "../include/Huffman/decompress.hpp"
	#include "../include/Huffman/utils.hpp"

	// ? For non textual files
	extern "C" double aes_cpu_encrypt_ffi(const char* input_path, const char* output_path, const char* password)
	{
		return aes_ecb_encrypt(input_path, output_path, password);
	}
	
	extern "C" double aes_cpu_decrypt_ffi(const char* input_path, const char* output_path, const char* password)
	{
		return aes_ecb_decrypt(input_path, output_path, password);
	}

	extern "C" double aes_cpu_encrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
	{
		string compressed_value[256] = {""};
		string input_file_name = input_path;
		ullint input_file_size, compressed_size_wo_header, compressed_file_size;
		unsigned char *compressed_file_buffer;
		input_file_size = getFileSizeBytes(input_path);
		map<char, ullint> char_freqs = coutCharFrequency(input_path, input_file_size);
		HuffNode *const root = generateHuffmanTree(char_freqs);
		string buff = "";
		compressed_size_wo_header = storeHuffmanValue(root, buff, compressed_value);
		

		double compression_time = compress(
			input_file_name,
			&compressed_file_buffer,
			compressed_size_wo_header,
			input_file_size,
			&compressed_file_size,
			compressed_value
		);
		
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		string user_key;
		string input_file, output_file = output_path, output_file_name;
		byte_t *ciphertext = NULL;
		byte_t ciphertext_block[AES_BLOCK_SIZE];
		byte_t round_key[176];
		char key[17];

		// Reading input and output file paths
		input_file = input_path;
		filesystem::path output_file_path = filesystem::path(input_file).filename();
		output_file_name = output_file_path.string();

		#if (OS == 1)
			output_file = output_path + ("\\" + output_file_name) + ".enc";
		#elif (OS == 2)
			output_file = output_path + ("/" + output_file_name) + ".enc";
		#endif

		if (!filesystem::exists(input_file))
		{
			cerr << endl << "Error : Invalid Input File ... Exiting !!!" << endl;
			exit(AES_FAILURE);
		}

		int file_length = (int) compressed_file_size;
		uintmax_t plaintext_blocks = (file_length + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
		ciphertext = (byte_t*)malloc(plaintext_blocks * AES_BLOCK_SIZE);

		if (ciphertext == NULL)
		{
			cerr << endl << "Error : Failed To Allocate Memory To Output File Buffer ... Exiting !!!" << endl;
			exit(AES_FAILURE);
		}

		string key_str = get_hash(password);
		aes_cpu_expand_key(round_key, strcpy(key, key_str.c_str()));

		StopWatchInterface *timer = NULL;
		sdkCreateTimer(&timer);
		sdkStartTimer(&timer);
		{
			for (uintmax_t i = 0; i < plaintext_blocks; i++)
			{
				aes_cpu_encrypt(compressed_file_buffer + i * AES_BLOCK_SIZE, round_key, ciphertext_block);
				memcpy(ciphertext + i * AES_BLOCK_SIZE, ciphertext_block, sizeof(byte_t) * AES_BLOCK_SIZE);
			}
		}
		sdkStopTimer(&timer);
		double encryption_time = (double)sdkGetTimerValue(&timer);

		// Write Encrypted Data
		// write_file(output_file.c_str(), ciphertext, AES_BLOCK_SIZE * plaintext_blocks);
		writeCryptFile(output_file, key_str, ciphertext, AES_BLOCK_SIZE * plaintext_blocks);

		sdkDeleteTimer(&timer);
		timer = NULL;

		free(ciphertext);
		ciphertext = NULL;

		free(compressed_file_buffer);
		compressed_file_buffer = NULL;

		return encryption_time + compression_time;
	}
	
	extern "C" double aes_cpu_decrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
	{
		// Variable Declarations
		string user_key;
		string input_file, output_file, output_file_name;
		int output_file_name_index = 0, input_file_length=0;
		byte_t *plaintext = NULL;
		byte_t *ciphertext = NULL;
		byte_t plaintext_block[AES_BLOCK_SIZE];
		byte_t round_key[176];
		char key[17];
		
		// Code

		// Reading input and output file paths
		input_file = input_path;
		output_file_name_index = input_file.find("enc") - 1;
		output_file = input_file.substr(0, output_file_name_index);

		#if (OS == 1)
			output_file = output_path + ("\\" + filesystem::path(input_file.substr(0, output_file_name_index)).filename().string());
		#elif (OS == 2)
			output_file = output_path + ("/" + filesystem::path(input_file.substr(0, output_file_name_index)).filename().string());
		#endif

		if (!filesystem::exists(input_file))
		{
			cerr << endl << "Error : Invalid Input File ... Exiting !!!" << endl;
			exit(AES_FAILURE);
		}

		// Allocate memory to input buffer
		// int file_length = filesystem::file_size(input_file);
		// ciphertext = (byte_t *)malloc(sizeof(byte_t) * file_length);
		// if (ciphertext == NULL)
		// {
		// 	cerr << endl << "Error : Failed To Allocate Memory To Input File Buffer ... Exiting !!!" << endl;
		// 	exit(AES_FAILURE);
		// }

		// Read Encrypted Input File
		// uintmax_t bytes_read = read_file(input_file.c_str(), ciphertext, file_length);
		// if (bytes_read <= 0)
		// {
		// 	cerr << endl << "Error : Empty File ... Please Select A Valid File ... Exiting !!!" << endl;
		// 	exit(AES_FAILURE);
		// }
		
		string key_str = get_hash(password);
		int error_type = 0;
		if(!readCryptFile(input_file, key_str, &ciphertext, &input_file_length, &error_type)){
			return error_type;
		}
		
		uintmax_t ciphertext_blocks = (input_file_length + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
		plaintext = (byte_t*)malloc(ciphertext_blocks * AES_BLOCK_SIZE);
		if (plaintext == NULL)
		{
			cerr << endl << "Error : Failed To Allocate Memory To Output File Buffer ... Exiting !!!" << endl;
			exit(AES_FAILURE);
		}

		aes_cpu_expand_key(round_key, strcpy(key, key_str.c_str()));

		StopWatchInterface *timer = NULL;
		sdkCreateTimer(&timer);
		sdkStartTimer(&timer);
		{
			for (uintmax_t i = 0; i < ciphertext_blocks; i++)
			{
				aes_cpu_decrypt(ciphertext + i * AES_BLOCK_SIZE, round_key, plaintext_block);
				memcpy(plaintext + i * AES_BLOCK_SIZE, plaintext_block, sizeof(byte_t) * AES_BLOCK_SIZE);
			}
		}
		sdkStopTimer(&timer);
		double decryption_time = (double)sdkGetTimerValue(&timer);

		double decompression_time = decompress(plaintext, output_file, input_file_length);

		sdkDeleteTimer(&timer);
		timer = NULL;
		
		free(plaintext);
		plaintext = NULL;

		free(ciphertext);
		ciphertext = NULL;

		return decryption_time + decompression_time;
	}
	
#endif