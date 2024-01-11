#include "../../include/AES/AES.hpp"
#include "../../include/Common/Tables.hpp"

// Global Variables
state_t *state;
const byte_t *aes_round_key;

// Function Definitions
byte_t xtime(byte_t x)
{
	return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

byte_t multiply(byte_t x, byte_t y)
{
    return (((y & 1) * x) ^
		((y >> 1 & 1) * xtime(x)) ^
		((y >> 2 & 1) * xtime(xtime(x))) ^
		((y >> 3 & 1) * xtime(xtime(xtime(x)))) ^
		((y >> 4 & 1) * xtime(xtime(xtime(xtime(x))))));
}

void aes_cpu_expand_key(byte_t *round_key, char *key)
{
    // Variable Declrations
    unsigned char aux[4], k;
    size_t i, j;

    // Code

    // 1st round is the key itself
    for (i = 0; i < AES_COLS; i++)
    {
        round_key[(i * 4) + 0] = (byte_t) key[(i * 4) + 0];
        round_key[(i * 4) + 1] = (byte_t) key[(i * 4) + 1];
        round_key[(i * 4) + 2] = (byte_t) key[(i * 4) + 2];
        round_key[(i * 4) + 3] = (byte_t) key[(i * 4) + 3];
    }

    // All other round keys are derived from previous round keys
    while (i < (AES_COLS * (AES_ROUNDS + 1)))
    {
        for (j = 0; j < 4; j++)
            aux[j] = round_key[(i - 1) * AES_COLS + j];
        
        if (i % AES_COLS == 0)
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

            aux[0] = aux[0] ^ round_constants[i / AES_COLS];
        }

        round_key[i * 4 + 0] = round_key[(i - AES_COLS) * 4 + 0] ^ aux[0];
        round_key[i * 4 + 1] = round_key[(i - AES_COLS) * 4 + 1] ^ aux[1];
        round_key[i * 4 + 2] = round_key[(i - AES_COLS) * 4 + 2] ^ aux[2];
        round_key[i * 4 + 3] = round_key[(i - AES_COLS) * 4 + 3] ^ aux[3];

        i++;
    }
}

void aes_cpu_add_round_key(byte_t round)
{
	// Code
    for (byte_t i = 0; i < 4; i++)
    {
        for (byte_t j = 0; j < 4; j++)
            (*state)[i][j] ^= aes_round_key[round * AES_COLS * 4 + i * AES_COLS + j];
    }
}

void aes_cpu_byte_sub(void)
{
    // Code
    for (byte_t i = 0; i < 4; i++)
    {
        for (byte_t j = 0; j < 4; j++)
            (*state)[j][i] = sbox[(*state)[j][i]];
    }
}

void aes_cpu_byte_sub_inverse(void)
{
    // Code
    for (byte_t i = 0; i < 4; i++)
    {
        for (byte_t j = 0; j < 4; j++)
            (*state)[j][i] = sbox_inverse[(*state)[j][i]];
    }
}

void aes_cpu_shift_rows(void)
{
    // Code
    byte_t temp;

    temp           = (*state)[0][1];
	(*state)[0][1] = (*state)[1][1];
	(*state)[1][1] = (*state)[2][1];
	(*state)[2][1] = (*state)[3][1];
	(*state)[3][1] = temp;

	temp           = (*state)[0][2];
	(*state)[0][2] = (*state)[2][2];
	(*state)[2][2] = temp;

	temp           = (*state)[1][2];
	(*state)[1][2] = (*state)[3][2];
	(*state)[3][2] = temp;

	temp           = (*state)[0][3];
	(*state)[0][3] = (*state)[3][3];
	(*state)[3][3] = (*state)[2][3];
	(*state)[2][3] = (*state)[1][3];
	(*state)[1][3] = temp;
}

void aes_cpu_shift_rows_inverse(void)
{
    // Code
    byte_t temp;

    // Inverse row shift operation
    temp           = (*state)[3][1];
    (*state)[3][1] = (*state)[2][1];
    (*state)[2][1] = (*state)[1][1];
    (*state)[1][1] = (*state)[0][1];
    (*state)[0][1] = temp;

    temp           = (*state)[0][2];
    (*state)[0][2] = (*state)[2][2];
    (*state)[2][2] = temp;

    temp           = (*state)[1][2];
    (*state)[1][2] = (*state)[3][2];
    (*state)[3][2] = temp;

    temp           = (*state)[0][3];
    (*state)[0][3] = (*state)[1][3];
    (*state)[1][3] = (*state)[2][3];
    (*state)[2][3] = (*state)[3][3];
    (*state)[3][3] = temp;
}

void aes_cpu_mix_columns(void)
{
    // Variable Declarations
    unsigned char tmp, tm, t;

    // Code
    for (int i = 0; i < 4; i++)
    {
        t = (*state)[i][0];
		tmp = (*state)[i][0] ^ (*state)[i][1] ^ (*state)[i][2] ^ (*state)[i][3];
		
        tm = (*state)[i][0] ^ (*state)[i][1]; 
        tm = xtime(tm);  
        (*state)[i][0] ^= tm ^ tmp;

		tm = (*state)[i][1] ^ (*state)[i][2]; 
        tm = xtime(tm);  
        (*state)[i][1] ^= tm ^ tmp;

		tm = (*state)[i][2] ^ (*state)[i][3]; 
        tm = xtime(tm);  
        (*state)[i][2] ^= tm ^ tmp;

		tm = (*state)[i][3] ^ t;                
        tm = xtime(tm);  
        (*state)[i][3] ^= tm ^ tmp;
    }
}

void aes_cpu_mix_columns_inverse(void)
{
    // Variable Declarations
    int i;
	unsigned char a, b, c, d;

    // Code
	for (i = 0; i < 4; i++) 
    {
		a = (*state)[i][0];
		b = (*state)[i][1];
		c = (*state)[i][2];
		d = (*state)[i][3];

		(*state)[i][0] = multiply(a, 0x0e) ^ multiply(b, 0x0b) ^ multiply(c, 0x0d) ^ multiply(d, 0x09);
		(*state)[i][1] = multiply(a, 0x09) ^ multiply(b, 0x0e) ^ multiply(c, 0x0b) ^ multiply(d, 0x0d);
		(*state)[i][2] = multiply(a, 0x0d) ^ multiply(b, 0x09) ^ multiply(c, 0x0e) ^ multiply(d, 0x0b);
		(*state)[i][3] = multiply(a, 0x0b) ^ multiply(b, 0x0d) ^ multiply(c, 0x09) ^ multiply(d, 0x0e);
	}
}

static void copy_block(unsigned char *input, unsigned char* output)
{
    // Code
    for (unsigned char i = 0; i < AES_LENGTH; i++)
        output[i] = input[i];
}

void cipher(void)
{
    // Variable Declarations
    byte_t round = 0;

    // Code
    aes_cpu_add_round_key(0);

	// Round 1 (R1) to AES_ROUNDS - 1 (R9)
    for (round = 1; round < AES_ROUNDS; round++)
    {
        aes_cpu_byte_sub();
        aes_cpu_shift_rows();
        aes_cpu_mix_columns();
        aes_cpu_add_round_key(round);
    }

	// Final Round Without Column Mixing
	aes_cpu_byte_sub();
	aes_cpu_shift_rows();
	aes_cpu_add_round_key(AES_ROUNDS);
}

void decipher(void)
{
    // Code
    // Variable Declarations
    byte_t round = 0;

    // Code
    aes_cpu_add_round_key(AES_ROUNDS);

	// Round AES_ROUNDS - 1 (R9) to Round 0
    for (round = AES_ROUNDS - 1; round > 0; round--)
    {
        aes_cpu_shift_rows_inverse();
        aes_cpu_byte_sub_inverse();
        aes_cpu_add_round_key(round);
        aes_cpu_mix_columns_inverse(); 
    }

	// Final Round Without Column Mixing
	aes_cpu_shift_rows_inverse();
    aes_cpu_byte_sub_inverse();
    aes_cpu_add_round_key(0);
}

void aes_cpu_encrypt(byte_t* input, const byte_t* round_key, byte_t* output)
{
    // Code
    copy_block(input, output);
    state = (state_t*)output;
    aes_round_key = round_key;

    cipher();
}

void aes_cpu_decrypt(byte_t* input, const byte_t* round_key, byte_t* output)
{
    // Code
    copy_block(input, output);
    state = (state_t*)output;
    aes_round_key = round_key;

    decipher();
}

// Library Exports
double aes_ecb_encrypt(const char *input_path, const char *output_path, const char* password)
{
    // Variable Declarations
    string input_file, output_file, output_file_name;
    byte_t *plaintext = NULL;
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
		cout << "Input file -> " << input_file << endl;
        exit(AES_FAILURE);
    }

    // Allocate memory to input buffer
    int file_length = filesystem::file_size(input_file);
    plaintext = (byte_t *)malloc(sizeof(byte_t) * file_length);
    if (plaintext == NULL)
    {
        cerr << endl << "Error : Failed To Allocate Memory To Input File Buffer ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    // Read Input File
    uintmax_t bytes_read = read_file(input_file.c_str(), plaintext, file_length);
    if (bytes_read <= 0)
    {
        cerr << endl << "Error : Empty File ... Please Select A Valid File ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    uintmax_t plaintext_blocks = (bytes_read + AES_BLOCK_SIZE - 1) / AES_BLOCK_SIZE;
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
            aes_cpu_encrypt(plaintext + i * AES_BLOCK_SIZE, round_key, ciphertext_block);
            memcpy(ciphertext + i * AES_BLOCK_SIZE, ciphertext_block, sizeof(byte_t) * AES_BLOCK_SIZE);
        }
    }
    sdkStopTimer(&timer);
    double cpu_time = (double)sdkGetTimerValue(&timer);

    // Write Encrypted Data
    // write_file(output_file.c_str(), ciphertext, AES_BLOCK_SIZE * plaintext_blocks);
	writeCryptFile(output_file, key_str, ciphertext, AES_BLOCK_SIZE * plaintext_blocks);

    sdkDeleteTimer(&timer);
    timer = NULL;

    free(ciphertext);
    ciphertext = NULL;

    free(plaintext);
    plaintext = NULL;

    return cpu_time;
}

double aes_ecb_decrypt(const char *input_path, const char *output_path, const char* password)
{
    // Variable Declarations
    string input_file, output_file, output_file_name;
    int output_file_name_index=0, input_file_length=0;
    byte_t *plaintext = NULL;
    byte_t *ciphertext = NULL;
    byte_t plaintext_block[AES_BLOCK_SIZE];
    byte_t round_key[176];
    char key[17];
    
    // Code

    // Reading input and output file paths
    input_file = input_path;
    output_file_name_index = input_file.find("enc") - 1;
    #if (OS == 1)
		output_file = output_path + ("\\" + filesystem::path(input_file.substr(0, output_file_name_index)).filename().string());
    #elif (OS == 2)
		output_file = output_path + ("/" + filesystem::path(input_file.substr(0, output_file_name_index)).filename().string());
    #endif
	
    if (!filesystem::exists(input_file))
    {
        cerr << endl << "Error : Invalid Input File ... Exiting !!!" << endl;
		cout << "Input file -> " << input_file << endl;
        exit(AES_FAILURE);
    }

    // Allocate memory to input buffer
    // int file_length = filesystem::file_size(input_file);
    // ciphertext = (byte_t *)malloc(sizeof(byte_t) * file_length);
    // if (ciphertext == NULL)
    // {
    //     cerr << endl << "Error : Failed To Allocate Memory To Input File Buffer ... Exiting !!!" << endl;
    //     exit(AES_FAILURE);
    // }

    // Read Encrypted Input File
    // uintmax_t bytes_read = read_file(input_file.c_str(), ciphertext, file_length);
    // if (bytes_read <= 0)
    // {
    //     cerr << endl << "Error : Empty File ... Please Select A Valid File ... Exiting !!!" << endl;
    //     exit(AES_FAILURE);
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
    double cpu_time = (double)sdkGetTimerValue(&timer);

    write_file(output_file.c_str(), plaintext, AES_BLOCK_SIZE * ciphertext_blocks);

    sdkDeleteTimer(&timer);
    timer = NULL;

    free(plaintext);
    plaintext = NULL;

    free(ciphertext);
    ciphertext = NULL;

    return cpu_time;
}

