#include "../../include/AES/AES.hpp"
#include "../../include/CLFW/CLFW.hpp"
#include "../../include/Common/Tables.hpp"

// Function Definitions
void aes_ocl_expand_key(byte_t *host_round_key, char *key)
{
    // Variable Declrations
    unsigned char aux[4], k;
    size_t i;

    // Code

    // 1st round is the key itself
    for (i = 0; i < AES_STATE_SIDE; i++)
    {
        host_round_key[(i * 4) + 0] = (byte_t) key[(i * 4) + 0];
        host_round_key[(i * 4) + 1] = (byte_t) key[(i * 4) + 1];
        host_round_key[(i * 4) + 2] = (byte_t) key[(i * 4) + 2];
        host_round_key[(i * 4) + 3] = (byte_t) key[(i * 4) + 3];
    }

    // All other round keys are derived from previous round keys
    while (i < (AES_STATE_SIDE * (AES_ROUNDS + 1)))
    {
        for (size_t j = 0; j < 4; j++)
            aux[j] = host_round_key[(i - 1) * AES_STATE_SIDE + j];
        
        if (i % AES_STATE_SIDE == 0)
        {
            // Rotate Word
            k = aux[0];
            aux[0] = aux[1];
            aux[1] = aux[2];
            aux[2] = aux[3];
            aux[3] = k;

            // Substitute
            aux[0] = sbox[aux[0]];
            aux[1] = sbox[aux[1]];
            aux[2] = sbox[aux[2]];
            aux[3] = sbox[aux[3]];

            aux[0] = aux[0] ^ round_constants[i / AES_STATE_SIDE];
        }

        host_round_key[i * 4 + 0] = host_round_key[(i - AES_STATE_SIDE) * 4 + 0] ^ aux[0];
        host_round_key[i * 4 + 1] = host_round_key[(i - AES_STATE_SIDE) * 4 + 1] ^ aux[1];
        host_round_key[i * 4 + 2] = host_round_key[(i - AES_STATE_SIDE) * 4 + 2] ^ aux[2];
        host_round_key[i * 4 + 3] = host_round_key[(i - AES_STATE_SIDE) * 4 + 3] ^ aux[3];

        i++;
    }
}

double aes_ocl_encrypt(const char *input_path, const char *output_path, const char *password, const char* kernel_path)
{
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
    int file_length = filesystem::file_size(input_file);
    int padding = AES_LENGTH - (file_length % AES_LENGTH) + AES_LENGTH;
    int data_size = (file_length + padding) / AES_LENGTH;

    // Allocate memory to host input buffer and output buffer
    clfw->host_alloc_mem((void**)&h_plaintext, "uchar", (file_length + padding));
    clfw->host_alloc_mem((void**)&h_ciphertext, "uchar", (file_length + padding));

    // Fill Padding with 0s
    for (int i = 0; i < padding - 1; i++)
        h_plaintext[file_length + i] = 0;
    h_plaintext[file_length + padding - 1] = padding;

    // Read Input File
    size_t bytes_read = read_file(input_file.c_str(), h_plaintext, file_length);
    if (bytes_read <= 0)
    {
        cerr << endl << "Error : Empty File ... Please Select A Valid File ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    // Get Password
    string key_str = get_hash(password);
	aes_ocl_expand_key(h_round_key, strcpy(key, key_str.c_str()));

    // Create OpenCL Buffers
    d_plaintext = clfw->ocl_create_buffer(CL_MEM_READ_ONLY, sizeof(byte_t) * (file_length + padding));
    d_ciphertext = clfw->ocl_create_buffer(CL_MEM_WRITE_ONLY, sizeof(byte_t) * (file_length + padding));
    d_round_key = clfw->ocl_create_buffer(CL_MEM_READ_ONLY, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1));

    // Create OpenCL Program From External Source File
    clfw->ocl_create_program(kernel_path);

    // Create OpenCL Kernel and set arguments
    clfw->ocl_create_kernel("aes_ocl_ecb_encrypt", "bbbi", d_plaintext, d_ciphertext, d_round_key, data_size);

    // Copy host data in device buffers
    clfw->ocl_write_buffer(d_round_key, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1), h_round_key);
    clfw->ocl_write_buffer(d_plaintext, sizeof(byte_t) * (file_length + padding), h_plaintext);

    // OpenCL Schedule Kernel
    float gpu_time = clfw->ocl_execute_kernel(clfw->get_global_work_size(LOCAL_WORK_SIZE, data_size), LOCAL_WORK_SIZE);

    // Copy Ciphertext from Device to Host
    clfw->ocl_read_buffer(d_ciphertext, sizeof(byte_t) * (file_length + padding), h_ciphertext);

    // Write Encrypted Data
    // write_file(output_file.c_str(), h_ciphertext, (file_length + padding));
    writeCryptFile(output_file, (const char*)key, h_ciphertext, (file_length + padding));

    clfw->ocl_release_buffer(d_round_key);
    clfw->ocl_release_buffer(d_ciphertext);
    clfw->ocl_release_buffer(d_plaintext);

    clfw->host_release_mem(&h_ciphertext);
    clfw->host_release_mem(&h_plaintext);

    clfw->uninitialize();

    delete clfw;
    clfw = nullptr;

    return gpu_time;
}

double aes_ocl_decrypt(const char *input_path, const char *output_path, const char *password, const char* kernel_path)
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
    clfw->ocl_create_program(kernel_path);

    // Create OpenCL Kernel and set arguments
    clfw->ocl_create_kernel("aes_ocl_ecb_decrypt", "bbbi", d_ciphertext, d_plaintext, d_round_key, data_size);

    // Copy host data in device buffers
    clfw->ocl_write_buffer(d_round_key, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1), h_round_key);
    clfw->ocl_write_buffer(d_ciphertext, sizeof(byte_t) * (file_length), h_ciphertext);
    
    // OpenCL Schedule Kernel
    float gpu_time = clfw->ocl_execute_kernel(clfw->get_global_work_size(LOCAL_WORK_SIZE, data_size), LOCAL_WORK_SIZE);
    
    // Copy Ciphertext from Device to Host
    clfw->ocl_read_buffer(d_plaintext, sizeof(byte_t) * (file_length), h_plaintext);

    // Write Encrypted Data
    int padding = (int)h_plaintext[file_length - 1];
    write_file(output_file.c_str(), h_plaintext, (file_length - padding));

    clfw->ocl_release_buffer(d_round_key);
    clfw->ocl_release_buffer(d_plaintext);
    clfw->ocl_release_buffer(d_ciphertext);

    clfw->host_release_mem(&h_plaintext);
    clfw->host_release_mem(&h_ciphertext);

    clfw->uninitialize();

    delete clfw;
    clfw = nullptr;

    return gpu_time;
}

