#pragma once

#ifndef _MACROS
#define _MACROS

// Macros
#define AES_BLOCK_SIZE  16
#define AES_LENGTH      16
#define AES_COLS        4
#define AES_ROUNDS      10
#define AES_BITS        128
#define AES_STATE_SIDE  4
#define AES_SUCCESS     EXIT_SUCCESS
#define AES_FAILURE     EXIT_FAILURE

#define LOCAL_WORK_SIZE 256

#if defined (_WIN32) || defined (_WIN64) || defined (WIN32) || defined (WIN64)
    #define OS 1
#elif defined(__linux)
    #define OS 2
#endif

// Typedefs
typedef unsigned char byte_t;
typedef bool status_t;
typedef byte_t state_t[4][4];

#define ullint unsigned long long int
#define TOTAL_CHARS 256

struct HuffNode{
	char ch;
	ullint count;
	HuffNode *left, *right;

	HuffNode(ullint count){
		this->ch = 0;
		this->count = count;
		this->left = this->right = nullptr;
	}

	HuffNode(char ch, ullint count){
		this->ch = ch;
		this->count = count;
		this->left = this->right = nullptr;
	}
};

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
