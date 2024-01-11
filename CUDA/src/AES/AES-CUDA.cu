#include "../../include/AES/AES-CUDA.cuh"
#include "../../include/Common/Tables.hpp"

// Global Variables
static __constant__ byte_t d_sbox[256] = 
{
    //0     1    2      3     4    5     6     7      8     9      A     B     C     D     E     F
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,	//0
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,	//1
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,	//2
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,	//3
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,	//4
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,	//5
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,	//6
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,	//7
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,	//8
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,	//9
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,	//A
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,	//B
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,	//C
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,	//D
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,	//E
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16  //F
};

static __constant__ byte_t d_sbox_inverse[256] = 
{
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

// CUDA Encryption Kernel
__global__ void aes_cuda_ecb_encrypt(byte_t* d_plaintext, byte_t* d_ciphertext, byte_t* d_round_key, int data_size)
{
    // Code
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < data_size)
    {
        cipher(d_plaintext + id * 16, d_ciphertext + id * 16, d_round_key);
    }  
}

__global__ void aes_cuda_ecb_decrypt(byte_t* d_ciphertext, byte_t* d_plaintext, byte_t* d_round_key, int data_size)
{
    // Code
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < data_size)
    {
        inverse_cipher(d_ciphertext + id * 16, d_plaintext + id * 16, d_round_key);
    }  
}

// Function Definitions
void cuda_mem_alloc(void** dev_ptr, size_t size)
{
    // Code
    cudaError_t result = cudaMalloc(dev_ptr, size);
    if (result != cudaSuccess)
    {
        cerr << endl << "Failed to allocate memory to " << dev_ptr << " : " << cudaGetErrorString(result) << " ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }
}

void cuda_mem_copy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    // Code
    cudaError_t result = cudaMemcpy(dst, src, count, kind);
    if (result != cudaSuccess)
    {
        cerr << endl << "Failed to copy memory from " << src << " to " << dst << " : " << cudaGetErrorString(result) << " ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }
}

void cuda_mem_free(void** dev_ptr)
{
    // Code
    if (*dev_ptr)
    {
        cudaFree(*dev_ptr);
        *dev_ptr = NULL;
    }
}

__device__ byte_t xtime(byte_t x)
{
	return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

void aes_cuda_expand_key(byte_t *host_round_key, char *key)
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

__device__ void aes_cuda_add_round_key(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE], byte_t *aes_round_key, byte_t round)
{
	// Code
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
            state[i][j] ^= aes_round_key[round * AES_STATE_SIDE * 4 + i * AES_STATE_SIDE + j];
    }
}

__device__ void aes_cuda_byte_sub(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
{
    // Code
    #pragma unroll
    for (byte_t i = 0; i < 4; i++)
    {
        #pragma unroll
        for (byte_t j = 0; j < 4; j++)
            state[j][i] = d_sbox[state[j][i]];
    }
}

__device__ void aes_cuda_byte_sub_inverse(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
{
    // Code
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        #pragma unroll
        for (int j = 0; j < 4; j++)
            state[j][i] = d_sbox_inverse[state[j][i]];
    }
}

__device__ void aes_cuda_shift_rows(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
{
    // Code
    #pragma unroll
    for (int num_shifts = 0; num_shifts < AES_STATE_SIDE; num_shifts++)
    {
        #pragma unroll
        for (int j = 0; j < num_shifts; j++)
        {
            byte_t temp = state[0][num_shifts];
            #pragma unroll
            for (int k = 0; k < AES_STATE_SIDE - 1; k++)
                state[k][num_shifts] = state[k + 1][num_shifts];
            state[AES_STATE_SIDE - 1][num_shifts] = temp;
        }
    }
}

__device__ void aes_cuda_shift_rows_inverse(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
{
    // Code
    #pragma unroll
    for (int num_shifts = 1; num_shifts < AES_STATE_SIDE; num_shifts++)
    {
        #pragma unroll
        for (int j = 0; j < AES_STATE_SIDE - num_shifts; j++)
        {
            byte_t temp = state[0][num_shifts];
            #pragma unroll
            for (int k = 0; k < AES_STATE_SIDE - 1; k++)
                state[k][num_shifts] = state[k + 1][num_shifts];
            state[AES_STATE_SIDE - 1][num_shifts] = temp;
        }
    }
}

__device__ void aes_cuda_mix_columns(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
{
    // Variable Declarations
    byte_t a[AES_STATE_SIDE], b[AES_STATE_SIDE], result[AES_STATE_SIDE];

    // Code
    #pragma unroll
    for (int i = 0; i < AES_STATE_SIDE; i++)
    {
        #pragma unroll
        for (int j = 0; j < AES_STATE_SIDE; j++)
            result[j] = state[i][j];

        #pragma unroll
        for (int k = 0; k < AES_STATE_SIDE; k++)
        {
            a[k] = result[k];
            b[k] = (result[k] << 1) ^ (0x1B * (1 & (result[k] >> 7)));
        }

        result[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
        result[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
        result[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
        result[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0];

        #pragma unroll
        for (int j = 0; j < AES_STATE_SIDE; j++)
            state[i][j] = result[j];
    }
}

__device__ void aes_cuda_mix_columns_inverse(byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
{
    // Variable Declarations
    byte_t u,v;
    byte_t a[AES_STATE_SIDE], b[AES_STATE_SIDE], result[AES_STATE_SIDE];

    // Code
    #pragma unroll
    for (int i = 0; i < AES_STATE_SIDE; i++)
    {
        u = xtime(xtime(state[i][0] ^ state[i][2]));
        v = xtime(xtime(state[i][1] ^ state[i][3]));
        state[i][0] ^= u;
        state[i][1] ^= v;
        state[i][2] ^= u;
        state[i][3] ^= v;
    }

    #pragma unroll
    for (int i = 0; i < AES_STATE_SIDE; i++)
    {
        memcpy(result, state[i], AES_STATE_SIDE * sizeof(byte_t));

        #pragma unroll
        for (int k = 0; k < AES_STATE_SIDE; k++)
        {
            a[k] = result[k];
            b[k] = (result[k] << 1) ^ (0x1B * (1 & (result[k] >> 7)));
        }

        result[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
        result[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
        result[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
        result[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0];

        memcpy(state[i], result, AES_STATE_SIDE * sizeof(byte_t));
    }
}

__device__ void cipher(byte_t input[AES_LENGTH], byte_t output[AES_LENGTH], byte_t* round_key)
{
    // Variable Declarations
    byte_t state[AES_STATE_SIDE][AES_STATE_SIDE];
    byte_t round = 0;

    // Code
    transform1D(input, state);

    aes_cuda_add_round_key(state, round_key, 0);

	// Round 1 (R1) to AES_ROUNDS - 1 (R9)
    #pragma unroll
    for (round = 1; round < AES_ROUNDS; round++)
    {
        aes_cuda_byte_sub(state);
        aes_cuda_shift_rows(state);
        aes_cuda_mix_columns(state);
        aes_cuda_add_round_key(state, round_key, round);
    }

	// Final Round Without Column Mixing
	aes_cuda_byte_sub(state);
	aes_cuda_shift_rows(state);
	aes_cuda_add_round_key(state, round_key, AES_ROUNDS);

    transform2D(output, state);
}

__device__ void inverse_cipher(byte_t input[AES_LENGTH], byte_t output[AES_LENGTH], byte_t* round_key)
{
    // Variable Declarations
    byte_t round = 0;
    byte_t state[AES_STATE_SIDE][AES_STATE_SIDE];

    // Code
    transform1D(input, state);

    aes_cuda_add_round_key(state, round_key, AES_ROUNDS);

	// Round AES_ROUNDS - 1 (R9) to Round 0
    #pragma unroll
    for (round = AES_ROUNDS - 1; round > 0; round--)
    {
        aes_cuda_shift_rows_inverse(state);
        aes_cuda_byte_sub_inverse(state);
        aes_cuda_add_round_key(state, round_key, round);
        aes_cuda_mix_columns_inverse(state); 
    }

	// Final Round Without Column Mixing
	aes_cuda_shift_rows_inverse(state);
    aes_cuda_byte_sub_inverse(state);
    aes_cuda_add_round_key(state, round_key, 0);

    transform2D(output, state);
}

__device__ void transform1D(byte_t input[AES_LENGTH], byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
{
    // Code
    for (int i = 0; i < AES_STATE_SIDE; i++)
    {
        for (int j = 0; j < AES_STATE_SIDE; j++)
            state[i][j] = input[j + AES_STATE_SIDE * i];
    }
}

__device__ void transform2D(byte_t output[AES_LENGTH], byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
{
    // Code
    for (int i = 0; i < AES_STATE_SIDE; i++)
    {
        for (int j = 0; j < AES_STATE_SIDE; j++)
            output[j + AES_STATE_SIDE * i] = state[i][j];
    }
}

// Library Exports
double aes_cuda_encrypt(const char *input_path, const char *output_path, const char* password)
{
    // Variable Declarations
    string user_key;
    string input_file, output_file, output_file_name;
    byte_t *h_plaintext = NULL, *h_ciphertext = NULL;
    byte_t *d_plaintext = NULL, *d_ciphertext = NULL, *d_round_key = NULL;
    byte_t h_round_key[176];
	char key[17];

    // Reading input and output file paths
    input_file = input_path;
    filesystem::path output_file_path = filesystem::path(input_file).filename();
    output_file_name = output_file_path.string();
    
    #if (OS == 1)
        output_file = output_path + ("\\" + output_file_name) + ".enc";
    #else
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

    // Allocate memory to input buffer and output buffer
    h_plaintext = (byte_t *)malloc(sizeof(byte_t) * (file_length + padding));
    if (h_plaintext == NULL)
    {
        cerr << endl << "Error : Failed To Allocate Memory To Input File Buffer ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    h_ciphertext = (byte_t*)malloc(sizeof(byte_t) * (file_length + padding));
    if (h_ciphertext == NULL)
    {
        cerr << endl << "Error : Failed To Allocate Memory To Output File Buffer ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    // Fill Padding with 0s
    for (int i = 0; i < padding - 1; i++)
        h_plaintext[file_length + i] = 0;
    h_plaintext[file_length + padding - 1] = padding;

    // Read Input File
    size_t bytes_read = read_file(input_file.c_str(), h_plaintext, file_length);
    if (bytes_read <= 0)
    {
		cout << "encrypt-input_file = " << input_file << endl;
        cerr << endl << "Error : Empty File ... Please Select A Valid File ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    // Get Password
   	string key_str = get_hash(password);
	aes_cuda_expand_key(h_round_key, strcpy(key, key_str.c_str()));

    // CUDA Kernel Configuration
    int num_blocks = (data_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // CUDA Kernel I/O
    cuda_mem_alloc((void**)&d_plaintext, sizeof(byte_t) * (file_length + padding));
    cuda_mem_alloc((void**)&d_ciphertext, sizeof(byte_t) * (file_length + padding));
    cuda_mem_alloc((void**)&d_round_key, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1));

    cuda_mem_copy(d_round_key, h_round_key, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1), cudaMemcpyHostToDevice);
    cuda_mem_copy(d_plaintext, h_plaintext, sizeof(byte_t) * (file_length + padding), cudaMemcpyHostToDevice);
    
	// CUDA Kernel Call
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    cudaDeviceSynchronize();
    sdkStartTimer(&timer);
    {
        aes_cuda_ecb_encrypt <<<num_blocks, THREADS_PER_BLOCK>>>(d_plaintext, d_ciphertext, d_round_key, data_size);
		cudaDeviceSynchronize();
	}
    sdkStopTimer(&timer);
    double gpu_time = (double) sdkGetTimerValue(&timer);
    
    // Copy Ciphertext from Device to Host
    cuda_mem_copy(h_ciphertext, d_ciphertext, sizeof(byte_t) * (file_length + padding), cudaMemcpyDeviceToHost);

    // Write Encrypted Data
	writeCryptFile(output_file, (const char*)key, h_ciphertext, (file_length + padding));

    sdkDeleteTimer(&timer);
    timer = NULL;

    cuda_mem_free((void**)&d_round_key);
    cuda_mem_free((void**)&d_ciphertext);
    cuda_mem_free((void**)&d_plaintext);

    free(h_ciphertext);
    h_ciphertext = NULL;

    free(h_plaintext);
    h_plaintext = NULL;

    return gpu_time;
}

double aes_cuda_decrypt(const char *input_path, const char *output_path, const char *password)
{
    // Variable Declarations
    string user_key;
    string input_file, output_file;
    int output_file_name_index;
    byte_t *h_plaintext = NULL, *h_ciphertext = NULL;
    byte_t *d_plaintext = NULL, *d_ciphertext = NULL, *d_round_key = NULL;
    byte_t h_round_key[176];
	char key[17];
	int file_length = 0;

    // Code

    // Reading input and output file paths
    input_file = input_path;
    output_file_name_index = input_file.find("enc") - 1;
    output_file = input_file.substr(0, output_file_name_index);
    if (!filesystem::exists(input_file))
    {
        cerr << endl << "Error : Invalid Input File Path ... Exiting !!!" << endl;
		cerr << "File Path - " << input_file << endl;
        exit(AES_FAILURE);
    }

    string key_str = get_hash(password);
	aes_cuda_expand_key(h_round_key, strcpy(key, key_str.c_str()));

	// Read Encrypted File
	int error_type = 0;
	if(!readCryptFile(input_file, (const char*) key, &h_ciphertext, &file_length, &error_type)){
		return error_type;
	}

	// Data Configuration
    int data_size = file_length / AES_LENGTH;

    h_plaintext = (byte_t *)malloc(sizeof(byte_t) * file_length);
    if (h_plaintext == NULL)
    {
        cerr << endl << "Error : Failed To Allocate Memory To Output File Buffer ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    // CUDA Kernel Configuration
    uintmax_t num_blocks = (data_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // CUDA Kernel I/O
    cuda_mem_alloc((void**)&d_ciphertext, sizeof(byte_t) * file_length);
    cuda_mem_alloc((void**)&d_plaintext, sizeof(byte_t) * file_length);
    cuda_mem_alloc((void**)&d_round_key, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1));

    cuda_mem_copy(d_round_key, h_round_key, sizeof(byte_t) * AES_BLOCK_SIZE * (AES_ROUNDS + 1), cudaMemcpyHostToDevice);
    cuda_mem_copy(d_ciphertext, h_ciphertext, sizeof(byte_t) * file_length, cudaMemcpyHostToDevice);
    
    // CUDA Kernel Call
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);

	cudaDeviceSynchronize();
    sdkStartTimer(&timer);
    {
        aes_cuda_ecb_decrypt <<<num_blocks, THREADS_PER_BLOCK>>>(d_ciphertext, d_plaintext, d_round_key, data_size);
		cudaDeviceSynchronize();
    }
    sdkStopTimer(&timer);
    double gpu_time = (double)sdkGetTimerValue(&timer);
    
    // Copy Ciphertext from Device to Host
    cuda_mem_copy(h_plaintext, d_plaintext, sizeof(byte_t) * file_length, cudaMemcpyDeviceToHost);

    // Write Encrypted Data
    int padding = (int)h_plaintext[file_length - 1];
    write_file(output_file.c_str(), h_plaintext, (file_length - padding));

    sdkDeleteTimer(&timer);
    timer = NULL;

    cuda_mem_free((void**)&d_round_key);
    cuda_mem_free((void**)&d_plaintext);
    cuda_mem_free((void**)&d_ciphertext);

    free(h_plaintext);
    h_plaintext = NULL;

    free(h_ciphertext);
    h_ciphertext = NULL;

    return gpu_time;
}
