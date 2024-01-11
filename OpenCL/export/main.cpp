#include "../include/AES/AES.hpp"
#include "../include/CLFW/CLFW.hpp"
#include "../include/Huffman/compress.hpp"
#include "../include/Huffman/decompress.hpp"
#include "../include/Huffman/utils.hpp"

extern "C" double aes_ocl_encrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
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
	compression_time /= 5;
	
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// Variable Declarations
    string user_key;
    string input_file, output_file, output_file_name;
    byte_t *h_plaintext = NULL, *h_ciphertext = NULL;
    cl_mem d_plaintext = NULL, d_ciphertext = NULL, d_round_key = NULL;
    byte_t h_round_key[176];
    char key[17];
    CLFW *clfw = nullptr;
    
    // Code

    // Initialize CLFW
    clfw = new CLFW();
    clfw->initialize();

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

    // Data Configuration
    int file_length = (int) compressed_file_size;
    int padding = AES_LENGTH - (file_length % AES_LENGTH) + AES_LENGTH;
    int data_size = (file_length + padding) / AES_LENGTH;

    // Allocate memory to host input buffer and output buffer
    clfw->host_alloc_mem((void**)&h_plaintext, "uchar", (file_length + padding));
    clfw->host_alloc_mem((void**)&h_ciphertext, "uchar", (file_length + padding));

	memcpy(h_plaintext, compressed_file_buffer, file_length);

    // Fill Padding with 0s
    for (int i = 0; i < padding - 1; i++)
        h_plaintext[file_length + i] = 0;
    h_plaintext[file_length + padding - 1] = padding;

    // // Read Input File
    // size_t bytes_read = read_file(input_file.c_str(), h_plaintext, file_length);
    // if (bytes_read <= 0)
    // {
    //     cerr << endl << "Error : Empty File ... Please Select A Valid File ... Exiting !!!" << endl;
    //     exit(AES_FAILURE);
    // }

    // Get Password
    string key_str = get_hash(password);
	aes_ocl_expand_key(h_round_key, strcpy(key, key_str.c_str()));

    // Create OpenCL Buffers
    d_plaintext = clfw->ocl_create_buffer(CL_MEM_READ_ONLY, sizeof(byte_t) * (file_length + padding));
    d_ciphertext = clfw->ocl_create_buffer(CL_MEM_WRITE_ONLY, sizeof(byte_t) * (file_length + padding));
    d_round_key = clfw->ocl_create_buffer(CL_MEM_READ_ONLY, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1));

    // Create OpenCL Program From External Source File
    clfw->ocl_create_program("./src/OCLKernels/OCLEncrypt.cl");

    // Create OpenCL Kernel and set arguments
    clfw->ocl_create_kernel("aes_ocl_ecb_encrypt", "bbbi", d_plaintext, d_ciphertext, d_round_key, data_size);

    // Copy host data in device buffers
    clfw->ocl_write_buffer(d_round_key, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1), h_round_key);
    clfw->ocl_write_buffer(d_plaintext, sizeof(byte_t) * (file_length + padding), h_plaintext);

    // OpenCL Schedule Kernel
    float encryption_time = clfw->ocl_execute_kernel(clfw->get_global_work_size(LOCAL_WORK_SIZE, data_size), LOCAL_WORK_SIZE);

    // Copy Ciphertext from Device to Host
    clfw->ocl_read_buffer(d_ciphertext, sizeof(byte_t) * (file_length + padding), h_ciphertext);

    // Write Encrypted Data
    // write_file(output_file.c_str(), h_ciphertext, (file_length + padding));
    writeCryptFile(output_file, (const char*)key, h_ciphertext, (file_length + padding));

	free(compressed_file_buffer);
	compressed_file_buffer = NULL;

    clfw->ocl_release_buffer(d_round_key);
    clfw->ocl_release_buffer(d_ciphertext);
    clfw->ocl_release_buffer(d_plaintext);

    clfw->host_release_mem(&h_ciphertext);
    clfw->host_release_mem(&h_plaintext);

    delete clfw;
    clfw = nullptr;

	return encryption_time + compression_time;
}

double aes_ocl_decrypt_huffman_ffi(const char* input_path, const char* output_path, const char* password)
{
	// Variable Declarations
    string user_key;
    string input_file, output_file;
    int output_file_name_index;
    byte_t *h_plaintext = NULL, *h_ciphertext = NULL;
    cl_mem d_plaintext = NULL, d_ciphertext = NULL, d_round_key = NULL;
    byte_t h_round_key[176];
    char key[17];
    int file_length = 0;
    CLFW *clfw = nullptr;
    
    // Code

    // Initialize CLFW
    clfw = new CLFW();
    clfw->initialize();

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

    // Get Password
    string key_str = get_hash(password);
	aes_ocl_expand_key(h_round_key, strcpy(key, key_str.c_str()));

    // Read Encrypted File
    int error_type = 0;
    if (!readCryptFile(input_file, (const char *)key, &h_ciphertext, &file_length, &error_type))
        return error_type;

    // Data Configuration
    int data_size = (file_length) / AES_LENGTH;

    // Allocate memory to host output buffer
    clfw->host_alloc_mem((void**)&h_plaintext, "uchar", (file_length));
    
    // Create OpenCL Buffers
    d_ciphertext = clfw->ocl_create_buffer(CL_MEM_READ_ONLY, sizeof(byte_t) * (file_length));
    d_plaintext = clfw->ocl_create_buffer(CL_MEM_WRITE_ONLY, sizeof(byte_t) * (file_length));
    d_round_key = clfw->ocl_create_buffer(CL_MEM_READ_ONLY, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1));

    // Create OpenCL Program From External Source File
    clfw->ocl_create_program("./src/OCLKernels/OCLDecrypt.cl");

    // Create OpenCL Kernel and set arguments
    clfw->ocl_create_kernel("aes_ocl_ecb_decrypt", "bbbi", d_ciphertext, d_plaintext, d_round_key, data_size);

    // Copy host data in device buffers
    clfw->ocl_write_buffer(d_round_key, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1), h_round_key);
    clfw->ocl_write_buffer(d_ciphertext, sizeof(byte_t) * (file_length), h_ciphertext);
    
    // OpenCL Schedule Kernel
    float decryption_time = clfw->ocl_execute_kernel(clfw->get_global_work_size(LOCAL_WORK_SIZE, data_size), LOCAL_WORK_SIZE);
    
    // Copy Ciphertext from Device to Host
    clfw->ocl_read_buffer(d_plaintext, sizeof(byte_t) * (file_length), h_plaintext);

    // Write Encrypted Data
    int padding = (int)h_plaintext[file_length - 1];

	double decompression_time = decompress(h_plaintext, output_file, (file_length - padding));
	decompression_time /= 5;

	clfw->ocl_release_buffer(d_round_key);
    clfw->ocl_release_buffer(d_plaintext);
    clfw->ocl_release_buffer(d_ciphertext);

    clfw->host_release_mem(&h_plaintext);
    clfw->host_release_mem(&h_ciphertext);

    delete clfw;
    clfw = nullptr;

	return decryption_time + decompression_time;
}

int main(int argc, char *argv[])
{
	const char* password = "1234";

	const char* enc_input_file_path = "F:\\Integration\\OpenCL\\data\\test50.txt";
	const char* enc_output_path = "F:\\Integration\\OpenCL\\data\\output";

	cout << endl << "AES + Huffman OpenCL Encrypt = " << aes_ocl_encrypt_huffman_ffi(enc_input_file_path, enc_output_path, password, 2) << " ms" << endl;
	// cout << endl << "AES OpenCL Encrypt = " << aes_ocl_encrypt(enc_input_file_path, enc_output_path, password, 2) << " ms" << endl;

	const char* dec_input_file_path = "F:\\Integration\\OpenCL\\data\\output\\test50.txt.enc";
	const char* dec_output_path = "F:\\Integration\\OpenCL\\data\\output";

	cout << endl << "AES + Huffman OpenCL Decrypt = " << aes_ocl_decrypt_huffman_ffi(dec_input_file_path, dec_output_path, password, 2) << " ms" << endl;
	// cout << endl << "AES OpenCL Decrypt = " << aes_ocl_decrypt(dec_input_file_path, dec_output_path, password, 2) << " ms" << endl;

	return 0;
}
