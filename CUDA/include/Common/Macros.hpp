#pragma once

// Macros
#define DEBUG_MODE               true

#define AES_BLOCK_SIZE      16
#define AES_LENGTH          16
#define AES_ROUNDS          10
#define AES_BITS            128
#define AES_STATE_SIDE      4
#define AES_SUCCESS         EXIT_SUCCESS
#define AES_FAILURE         EXIT_FAILURE
#define AES_FILE            "444341455547"

// CUDA
#define THREADS_PER_BLOCK   1024

#if defined (_WIN32) || defined (_WIN64) || defined (WIN32) || defined (WIN64)
    #define OS 1
#elif defined(__linux)
    #define OS 2
#endif

// Typedefs
typedef unsigned char byte_t;
typedef byte_t state_t[4][4];
typedef bool status_t;


#ifndef CONSTANTS
#define CONSTANTS

// ------------------- Compression & Decompression Common Macros-------------------

#define MAX_BITS 32
#define MAX_CODEWORD_LENGTH 16
#define MAX_CODE_NUM 256
#define PACK_SIZE 512

#define GAP_LENGTH_MAX 4
#define SEGMENT_SIZE 128

#define WARP_SIZE 32

#define FLAG_A    0x0100000000000000
#define FLAG_P    0x8000000000000000
#define FLAG_MASK 0xFF00000000000000

#define ull unsigned long long int

struct Symbol{
	unsigned char symbol;
	unsigned char length;
	unsigned int num;
};

#define CUERROR  {\
cudaError_t cuError;\
	if( cudaSuccess != (cuError = cudaGetLastError()) ){\
		printf("Error: %s : at %s line %d\n", cudaGetErrorString(cuError), __FILE__, __LINE__);\
		exit(EXIT_FAILURE);\
	} \
}

// ------------------------ Compression Macros -------------------------

#define GAP_ELEMENTS_NUM 8

// For Tesla V100, the number of CUDA blocks per SM is 8,
// the number of CUDA blocks is BLOCKNUM=8*80 
#define BLOCK_NUM 8*80
#define THREAD_NUM 256
#define THREAD_ELEMENT 32
#define TNUM_DIV_WSIZE THREAD_NUM/WARP_SIZE

#define GAP_THREADS 256
#define GAP_BLOCK_RANGE GAP_THREADS*GAP_ELEMENTS_NUM

#define GET_CHAR(value, shift) ((value>>((shift)*8))&0xFF)

#define LEFT 0
#define RIGHT 1
#define ull unsigned long long int

struct Codetable{
	unsigned int code;
	unsigned int length;
};



// ----------------------- Decompression Macros -----------------------

#define LOCAL_SEGMENT_NUM 2
#define LOCAL_SEGMENT_SIZE SEGMENT_SIZE/LOCAL_SEGMENT_NUM

#define NUM_THREADS 128
#define BLOCK_LAST NUM_THREADS/WARP_SIZE

#define GAP_FAC_NUM 8

#define SEPARATE_MIN 9

#define X 0
#define Y 1
#define Z 2
#define W 3
#define O 4

#define LOOP 100

#define UINT_OUT( symbols, symbol, pos ) \
	symbols = (symbols|(symbol<<( pos*8 )));

#define FIXED_PREFIX_BIT 10

struct Dectable{
	unsigned char symbol;
	unsigned char length;
};

struct TableInfo{
	unsigned int l1table_size;
	unsigned int l2table_size;
	unsigned int ptrtable_size;
};


// ------------------ Boundary Package Merge --------------------
struct NodePack{
	ull left_weight;
	ull right_weight;
	ull weight;
	int counter;
	int LR;
	int pack_pos;
	struct NodePack *chain;
};

struct NodePackList{
	struct NodePack *list;
	int current_pos[MAX_CODEWORD_LENGTH];
};
// -------------------------------------------------------------

#endif

enum INVALID_FILE_ERRORS{
	INVALID_SIGNATURE = -1,
	INVALID_PASSWORD = -2,
	INVALID_HMAC = -3,
	UNKNOWN_ERROR = -4
};

#define SIGNATURE_LENGTH  14
#define PASSWORD_LENGTH 16
#define HMAC_LENGTH 64
#define SHA256_HASH_SIZE 32
