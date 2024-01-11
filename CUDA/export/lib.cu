#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)

    #include <windows.h>
	#include "../include/AES/AES-CUDA.cuh"
	#include "../include/Huffman/encoder/compress.hpp"
	#include "../include/Huffman/decoder/decompress.hpp"

	extern "C" __declspec(dllexport) double aes_cuda_encrypt_ffi(const char* input_path, const char* output_path, const char* password)
	{
		return aes_cuda_encrypt(input_path, output_path, password);
	}
	
	extern "C" __declspec(dllexport) double aes_cuda_decrypt_ffi(const char* input_path, const char* output_path, const char* password)
	{
		return aes_cuda_decrypt(input_path, output_path, password);
	}

	extern "C" __declspec(dllexport) double aes_cuda_encrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
	{
		string input_file_path = input_path;
		string compressed_huff_file;
		
		#if (OS == 1)
			compressed_huff_file = output_path + (input_file_path.substr(input_file_path.rfind('\\')) + ".huff");
		#else
			compressed_huff_file = output_path + (input_file_path.substr(input_file_path.rfind('/')) + ".huff");
		#endif

		double compression_time = compress(input_path, compressed_huff_file);
		
		double encryption_time = aes_cuda_encrypt(compressed_huff_file.c_str(), output_path, password);

		return compression_time + encryption_time;
	}
		
	extern "C" __declspec(dllexport) double aes_cuda_decrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
	{
		double decryption_time = aes_cuda_decrypt(input_path, output_path, password);

		if (decryption_time < 0)
			return decryption_time;

		string input_file_path = input_path;
		string compressed_output_path = input_file_path.substr(0, input_file_path.rfind("enc")-1);
		string decompressed_output_path = compressed_output_path.substr(0, compressed_output_path.rfind("huff")-1);

		double decompression_time = decompress(compressed_output_path, decompressed_output_path);

		return decryption_time + decompression_time;
	}
	

#elif defined(__linux__)
	#include "../include/AES/AES-CUDA.cuh"
	#include "../include/Huffman/encoder/compress.hpp"
	#include "../include/Huffman/decoder/decompress.hpp"

	// ? For non textual files
	extern "C" double aes_cuda_encrypt_ffi(const char* input_path, const char* output_path, const char* password)
	{
		return aes_cuda_encrypt(input_path, output_path, password);
	}
	
	extern "C" double aes_cuda_decrypt_ffi(const char* input_path, const char* output_path, const char* password)
	{
		return aes_cuda_decrypt(input_path, output_path, password);
	}

	extern "C" double aes_cuda_encrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
	{
		string input_file_path = input_path;
		string compressed_huff_file;
		
		#if (OS == 1)
			compressed_huff_file = output_path + (input_file_path.substr(input_file_path.rfind('\\')) + ".huff");
		#else
			compressed_huff_file = output_path + (input_file_path.substr(input_file_path.rfind('/')) + ".huff");
		#endif

		double compression_time = compress(input_path, compressed_huff_file);
		
		double encryption_time = aes_cuda_encrypt(compressed_huff_file.c_str(), output_path, password);

		return compression_time + encryption_time;
	}
		
	extern "C" double aes_cuda_decrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
	{
		double decryption_time = aes_cuda_decrypt(input_path, output_path, password);

		if (decryption_time < 0)
			return decryption_time;

		string input_file_path = input_path;
		string compressed_output_path = input_file_path.substr(0, input_file_path.rfind("enc")-1);
		string decompressed_output_path = compressed_output_path.substr(0, compressed_output_path.rfind("huff")-1);

		double decompression_time = decompress(compressed_output_path, decompressed_output_path);

		return decryption_time + decompression_time;
	}
	
#endif