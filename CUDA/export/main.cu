#include "../include/AES/AES-CUDA.cuh"
#include "../include/Huffman/encoder/compress.hpp"
#include "../include/Huffman/decoder/decompress.hpp"

extern "C" double aes_cuda_encrypt_ffi(const char* input_path, const char* output_path, const char* password)
{
	double excryption_tim = aes_cuda_encrypt(input_path, output_path, password);
	cout << "excryption_tim = " << excryption_tim << endl;
	return excryption_tim;
}
	
extern "C" double aes_cuda_decrypt_ffi(const char* input_path, const char* output_path, const char* password)
{
	double decryption_time = aes_cuda_decrypt(input_path, output_path, password);
	cout << "decryption_time = " << decryption_time << endl;

	return decryption_time;
}

extern "C" double aes_cuda_encrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
{
	string input_file_path = input_path;
	string compressed_huff_file;
	
	if(OS == 1){
		compressed_huff_file = output_path + (input_file_path.substr(input_file_path.rfind('\\')) + ".huff");
	}
	else{
		compressed_huff_file = output_path + (input_file_path.substr(input_file_path.rfind('/')) + ".huff");
	}

	double compression_time = compress(input_path, compressed_huff_file);
	cout << "compression_time = " << compression_time << endl;
	
	double encryption_time = aes_cuda_encrypt(compressed_huff_file.c_str(), output_path, password);
	cout << "encryption_time = " << encryption_time << endl;

	return compression_time + encryption_time;
}
	
extern "C" double aes_cuda_decrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
{
	double decryption_time = aes_cuda_decrypt(input_path, output_path, password);
	cout << "decryption_time = " << decryption_time << endl;

	// if(decryption_time < 0){
	// 	return decryption_time;
	// }

	string input_file_path = input_path;
	string compressed_output_path = input_file_path.substr(0, input_file_path.rfind("enc")-1);
	string decompressed_output_path = compressed_output_path.substr(0, compressed_output_path.rfind("huff")-1);

	double decompression_time = decompress(compressed_output_path, decompressed_output_path);
	cout << "decompression_time = " << decompression_time << endl;

	return decryption_time + decompression_time;
}

int main(int argc, char *argv[])
{
	// const char* enc_input_file_path = "/home/uttkarsh/VIT/4th_Year/Major_Project/Code/Integration/CUDA/3/data/sample.txt";
	// const char* enc_output_path = "/home/uttkarsh/VIT/4th_Year/Major_Project/Code/Integration/CUDA/3/data/output";

    // const char* dec_input_file_path = "/home/uttkarsh/VIT/4th_Year/Major_Project/Code/Integration/CUDA/3/data/output/sample.txt.huff.enc";
	// const char* dec_output_path = "/home/uttkarsh/VIT/4th_Year/Major_Project/Code/Integration/CUDA/3/data/output";

	const char* enc_input_file_path = "F:\\Integration\\CUDA\\data\\test50.txt";
	const char* enc_output_path = "F:\\Integration\\CUDA\\data\\output";

	const char* dec_input_file_path = "C:\\Users\\Atharv\\Desktop\\Test\\Sample.txt.huff.enc";
	const char* dec_output_path = "C:\\Users\\Atharv\\Desktop\\Test";

	const char* password = "1234";
	double time = 0;

	// time = aes_cuda_encrypt(enc_input_file_path, enc_output_path, password);
	// time = aes_cuda_encrypt_huffman_ffi(enc_input_file_path, enc_output_path, password);
	// cout << "Encryption & Compression = " << time << " ms" << endl << endl;

	// time = aes_cuda_decrypt(dec_input_file_path, dec_output_path, password);
	time = aes_cuda_decrypt_huffman_ffi(dec_input_file_path, dec_output_path, password);
	cout << "Decryption & Decompression = " << time  << " ms" << endl;

	return 0;
}
