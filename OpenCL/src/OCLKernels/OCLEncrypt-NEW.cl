__constant const uchar d_sbox[256] = 
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

__constant const uchar d_sbox_inverse[256] = 
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

__constant const uchar d_log_table[256] = 
{
    0,   0,   25,  1,   50,  2,   26,  198, 75,  199, 27,  104, 51,  238, 223,
    3,   100, 4,   224, 14,  52,  141, 129, 239, 76,  113, 8,   200, 248, 105,
    28,  193, 125, 194, 29,  181, 249, 185, 39,  106, 77,  228, 166, 114, 154,
    201, 9,   120, 101, 47,  138, 5,   33,  15,  225, 36,  18,  240, 130, 69,
    53,  147, 218, 142, 150, 143, 219, 189, 54,  208, 206, 148, 19,  92,  210,
    241, 64,  70,  131, 56,  102, 221, 253, 48,  191, 6,   139, 98,  179, 37,
    226, 152, 34,  136, 145, 16,  126, 110, 72,  195, 163, 182, 30,  66,  58,
    107, 40,  84,  250, 133, 61,  186, 43,  121, 10,  21,  155, 159, 94,  202,
    78,  212, 172, 229, 243, 115, 167, 87,  175, 88,  168, 80,  244, 234, 214,
    116, 79,  174, 233, 213, 231, 230, 173, 232, 44,  215, 117, 122, 235, 22,
    11,  245, 89,  203, 95,  176, 156, 169, 81,  160, 127, 12,  246, 111, 23,
    196, 73,  236, 216, 67,  31,  45,  164, 118, 123, 183, 204, 187, 62,  90,
    251, 96,  177, 134, 59,  82,  161, 108, 170, 85,  41,  157, 151, 178, 135,
    144, 97,  190, 220, 252, 188, 149, 207, 205, 55,  63,  91,  209, 83,  57,
    132, 60,  65,  162, 109, 71,  20,  42,  158, 93,  86,  242, 211, 171, 68,
    17,  146, 217, 35,  32,  46,  137, 180, 124, 184, 38,  119, 153, 227, 165,
    103, 74,  237, 222, 197, 49,  254, 24,  13,  99,  140, 128, 192, 247, 112,
    7
};

__constant const uchar d_alog_table[256] = 
{
    1,   3,   5,   15,  17,  51,  85,  255, 26,  46,  114, 150, 161, 248, 19,
    53,  95,  225, 56,  72,  216, 115, 149, 164, 247, 2,   6,   10,  30,  34,
    102, 170, 229, 52,  92,  228, 55,  89,  235, 38,  106, 190, 217, 112, 144,
    171, 230, 49,  83,  245, 4,   12,  20,  60,  68,  204, 79,  209, 104, 184,
    211, 110, 178, 205, 76,  212, 103, 169, 224, 59,  77,  215, 98,  166, 241,
    8,   24,  40,  120, 136, 131, 158, 185, 208, 107, 189, 220, 127, 129, 152,
    179, 206, 73,  219, 118, 154, 181, 196, 87,  249, 16,  48,  80,  240, 11,
    29,  39,  105, 187, 214, 97,  163, 254, 25,  43,  125, 135, 146, 173, 236,
    47,  113, 147, 174, 233, 32,  96,  160, 251, 22,  58,  78,  210, 109, 183,
    194, 93,  231, 50,  86,  250, 21,  63,  65,  195, 94,  226, 61,  71,  201,
    64,  192, 91,  237, 44,  116, 156, 191, 218, 117, 159, 186, 213, 100, 172,
    239, 42,  126, 130, 157, 188, 223, 122, 142, 137, 128, 155, 182, 193, 88,
    232, 35,  101, 175, 234, 37,  111, 177, 200, 67,  197, 84,  252, 31,  33,
    99,  165, 244, 7,   9,   27,  45,  119, 153, 176, 203, 70,  202, 69,  207,
    74,  222, 121, 139, 134, 145, 168, 227, 62,  66,  198, 81,  243, 14,  18,
    54,  90,  238, 41,  123, 141, 140, 143, 138, 133, 148, 167, 242, 13,  23,
    57,  75,  221, 124, 132, 151, 162, 253, 28,  36,  108, 180, 199, 82,  246,
    1
};

// Macros
#define AES_STATE_SIDE  4
#define AES_LENGTH      16
#define AES_ROUNDS      10

#define GF2M(k, b) ((b) == 0 ? 0 : (d_alog_table[(d_log_table[(k)] + d_log_table[(b)]) % 255]))

// Typedefs
typedef unsigned char byte_t;
typedef bool status_t;
typedef byte_t state_t[4][4];

void aes_ocl_add_round_key(size_t block, __global uchar * buffer, __constant const uchar * round_key, size_t round_key_index)
{
	// Code
    for (int i = 0; i < 4; i++)
    {
        for (size_t i = 0; i < AES_LENGTH; i++)
            buffer[block * 16 + i] ^= round_key[round_key_index * 16 + i];
    }
}

void aes_ocl_byte_sub(size_t block, __global uchar *buffer, __constant const uchar *sbox)
{
    // Code
    for (size_t i = 0; i < AES_LENGTH; i++)
        buffer[block * 16 + i] = sbox[buffer[block * 16 + i]];
}

void aes_ocl_shift_rows(size_t block, __global uchar *buffer)
{
    // Code
    uchar temp;

	// Rotate first row 1 columns to left   
	temp = buffer[block * 16 + 4];
	buffer[block * 16 + 4] = buffer[block * 16 + 5];
	buffer[block * 16 + 5] = buffer[block * 16 + 6];
	buffer[block * 16 + 6] = buffer[block * 16 + 7];
	buffer[block * 16 + 7] = temp;

	// Rotate second row 2 columns to left  
	temp = buffer[block * 16 + 8];
	buffer[block * 16 + 8] = buffer[block * 16 + 10];
	buffer[block * 16 + 10] = temp;

	temp = buffer[block * 16 + 9];
	buffer[block * 16 + 9] = buffer[block * 16 + 11];
	buffer[block * 16 + 11] = temp;

	// Rotate third row 3 columns to left
	temp = buffer[block * 16 + 12];
	buffer[block * 16 + 12] = buffer[block * 16 + 15];
	buffer[block * 16 + 15] = buffer[block * 16 + 14];
	buffer[block * 16 + 14] = buffer[block * 16 + 13];
	buffer[block * 16 + 13] = temp;
}

void aes_ocl_mix_columns(size_t block, __global uchar *m)
{
    uchar new_block[16];
	new_block[0] = GF2M(2, m[block * 16 + 0]) ^ GF2M(3, m[block * 16 + 4]) ^ m[block * 16 + 8] ^ m[block * 16 + 12];
	new_block[1] = GF2M(2, m[block * 16 + 1]) ^ GF2M(3, m[block * 16 + 5]) ^ m[block * 16 + 9] ^ m[block * 16 + 13];
    new_block[2] = GF2M(2, m[block * 16 + 2]) ^ GF2M(3, m[block * 16 + 6]) ^ m[block * 16 + 10] ^ m[block * 16 + 14];
	new_block[3] = GF2M(2, m[block * 16 + 3]) ^ GF2M(3, m[block * 16 + 7]) ^ m[block * 16 + 11] ^ m[block * 16 + 15];
	new_block[4] = GF2M(2, m[block * 16 + 4]) ^ GF2M(3, m[block * 16 + 8]) ^ m[block * 16 + 12] ^ m[block * 16 + 0];
	new_block[5] = GF2M(2, m[block * 16 + 5]) ^ GF2M(3, m[block * 16 + 9]) ^ m[block * 16 + 13] ^ m[block * 16 + 1];
	new_block[6] = GF2M(2, m[block * 16 + 6]) ^ GF2M(3, m[block * 16 + 10]) ^ m[block * 16 + 14] ^ m[block * 16 + 2];
	new_block[7] = GF2M(2, m[block * 16 + 7]) ^ GF2M(3, m[block * 16 + 11]) ^ m[block * 16 + 15] ^ m[block * 16 + 3];
	new_block[8] = GF2M(2, m[block * 16 + 8]) ^ GF2M(3, m[block * 16 + 12]) ^ m[block * 16 + 0] ^ m[block * 16 + 4];
	new_block[9] = GF2M(2, m[block * 16 + 9]) ^ GF2M(3, m[block * 16 + 13]) ^ m[block * 16 + 1] ^ m[block * 16 + 5];
	new_block[10] =	GF2M(2, m[block * 16 + 10]) ^ GF2M(3, m[block * 16 + 14]) ^ m[block * 16 + 2] ^ m[block * 16 + 6];
	new_block[11] =	GF2M(2, m[block * 16 + 11]) ^ GF2M(3, m[block * 16 + 15]) ^ m[block * 16 + 3] ^ m[block * 16 + 7];
	new_block[12] =	GF2M(2, m[block * 16 + 12]) ^ GF2M(3, m[block * 16 + 0]) ^ m[block * 16 + 4] ^ m[block * 16 + 8];
	new_block[13] =	GF2M(2, m[block * 16 + 13]) ^ GF2M(3, m[block * 16 + 1]) ^ m[block * 16 + 5] ^ m[block * 16 + 9]; 
	new_block[14] =	GF2M(2, m[block * 16 + 14]) ^ GF2M(3, m[block * 16 + 2]) ^ m[block * 16 + 6] ^ m[block * 16 + 10]; 
	new_block[15] =	GF2M(2, m[block * 16 + 15]) ^ GF2M(3, m[block * 16 + 3]) ^ m[block * 16 + 7] ^ m[block * 16 + 11];
	for (size_t i = 0; i < 16; ++i)
	    m[block * 16 + i] = new_block[i];
}

// void transform1D(__global byte_t input[AES_LENGTH], byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
// {
//     // Code
//     for (int i = 0; i < AES_STATE_SIDE; i++)
//     {
//         for (int j = 0; j < AES_STATE_SIDE; j++)
//             state[i][j] = input[j + AES_STATE_SIDE * i];
//     }
// }

// void transform2D(__global byte_t output[AES_LENGTH], byte_t state[AES_STATE_SIDE][AES_STATE_SIDE])
// {
//     // Code
//     for (int i = 0; i < AES_STATE_SIDE; i++)
//     {
//         for (int j = 0; j < AES_STATE_SIDE; j++)
//             output[j + AES_STATE_SIDE * i] = state[i][j];
//     }
// }

// void ocl_cipher(__global byte_t input[AES_LENGTH], __global byte_t output[AES_LENGTH], __global byte_t* round_key)
// {
//     // Variable Declarations
//     byte_t state[AES_STATE_SIDE][AES_STATE_SIDE];
//     byte_t round = 0;

//     // Code
//     transform1D(input, state);

//     aes_ocl_add_round_key(state, round_key, 0);

// 	// Round 1 (R1) to AES_ROUNDS - 1 (R9)
//     for (round = 1; round < AES_ROUNDS; round++)
//     {
//         aes_ocl_byte_sub(state);
//         aes_ocl_shift_rows(state);
//         aes_ocl_mix_columns(state);
//         aes_ocl_add_round_key(state, round_key, round);
//     }

// 	// Final Round Without Column Mixing
// 	aes_ocl_byte_sub(state);
// 	aes_ocl_shift_rows(state);
// 	aes_ocl_add_round_key(state, round_key, AES_ROUNDS);

//     transform2D(output, state);
// }

// __kernel void aes_ocl_ecb_encrypt(__global unsigned char* d_plaintext, __global unsigned char* d_ciphertext, __global unsigned char* d_round_key, int data_size)
// {
//     // Code
//     int id = get_global_id(0);

//     if (id < data_size)
//     {
//         ocl_cipher(d_plaintext + id * 16, d_ciphertext + id * 16, d_round_key);
//     }  
// }


__kernel void aes_ocl_ecb_encrypt(__global uchar *buffer, const ulong blocks, __constant const uchar *round_key, const uint rounds, const uint round)
{
	size_t global_work_size = get_global_size(0);
	size_t global_id = get_global_id(0);
	/* If there are more work items than blocks, each work items takes AT MOST
	   1 block to process, otherwise it could be several blocks per work item. */
	size_t blocks_per_work_item = global_work_size < blocks ? blocks / global_work_size : 1;
	size_t reminder = global_work_size < blocks ? blocks % global_work_size : 0;
	size_t from_block = global_id * blocks_per_work_item;
	if (global_id < reminder)
		from_block += global_id;
	else
		from_block += reminder;

	/* If the work item should start working from a block outside the input data
	   boundaries, it shall do nothing. */
	if (from_block < blocks) {
		/* The last work item will process the input data from its start block to
		   the very last block. The other work items will just do blocks_per_work_item
		   blocks.
		   Each work item will process any block b such as from_block <= b < to_block. */
		//~ size_t to_block = global_id == global_work_size - 1 ? blocks : from_block + blocks_per_work_item;
		size_t to_block = from_block + blocks_per_work_item;
		if (global_id < reminder)
			to_block += 1;
        
        for (size_t b = from_block; b < to_block; ++b) 
        {
            if (round == 0) 
            {
                aes_ocl_add_round_key(b, buffer, round_key, 0);
            } 
            else if (round == rounds) 
            {
                aes_ocl_byte_sub(b, buffer, d_sbox);
                aes_ocl_shift_rows(b, buffer);
                aes_ocl_add_round_key(b, buffer, round_key, rounds);
            } 
            else 
            {
                aes_ocl_byte_sub(b, buffer, d_sbox);
                aes_ocl_shift_rows(b, buffer);
                aes_ocl_mix_columns(b, buffer);
                aes_ocl_add_round_key(b, buffer, round_key, round);
            }
	    }
    }
}


