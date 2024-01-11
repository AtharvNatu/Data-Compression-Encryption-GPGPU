#include "../../include/Common/Helper.hpp"

const char* AES_FILE_SIGNATURE = "4443414555470A";



size_t read_file(const char *input_file, byte_t *output_buffer, uintmax_t input_data_size)
{
    // Code
    if (input_file == NULL)
    {
        cerr << endl << "Error :  Failed To Read Input File " << input_file << " ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    FILE *fp = NULL;

    #if (OS == 1)
        fopen_s(&fp, input_file, "rb");
    #else
        fp = fopen(input_file, "rb");
    #endif

    if (fp == NULL)
    {
        cerr << endl << "Error :  Failed To Read Input File " << input_file << " ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    size_t bytes_read = fread(output_buffer, sizeof(byte_t), input_data_size, fp);

    fclose(fp);
    fp = NULL;

    return bytes_read;
}

size_t write_file(const char *output_file, byte_t *data, uintmax_t output_data_size)
{
    // Code
    if (output_file == NULL)
    {
        cerr << endl << "Error :  Failed To Read Output File " << output_file << " ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    FILE *fp = NULL;

    #if (OS == 1)
        fopen_s(&fp, output_file, "wb+");
    #else
        fp = fopen(output_file, "wb+");
    #endif

    if (fp == NULL)
    {
        cerr << endl << "Error :  Failed To Read Output File " << output_file << " ... Exiting !!!" << endl;
        exit(AES_FAILURE);
    }

    size_t data_size = fwrite(data, sizeof(byte_t), output_data_size, fp);

    fclose(fp);
    fp = NULL;

    return data_size;
}

string get_hash(const char* input)
{
    hash<string> cpp_hash;
    size_t input_hash = cpp_hash(input);
    string hash_str = to_string(input_hash);
    return hash_str.substr(0, 16); 
}

string calculateHMAC(string file_content, string password){
    std::stringstream ss_result;
    std::vector<uint8_t> out(SHA256_HASH_SIZE);
	
    hmac_sha256(
        password.data(),  password.size(),
        file_content.data(), file_content.size(),
        out.data(),  out.size()
    );

    for (uint8_t x : out) {
        ss_result << std::hex << std::setfill('0') << std::setw(2) << (int)x;
    }

	return ss_result.str();
}

void writeCryptFile(
	string file_path,
	string password,
	unsigned char* ciphertext, 
	unsigned int ciphertext_length
){
	FILE *fp = fopen(file_path.c_str(), "wb");

	fwrite(AES_FILE_SIGNATURE, SIGNATURE_LENGTH, 1, fp);
	putc('\n', fp);

	fwrite(password.c_str(), PASSWORD_LENGTH, 1, fp);
	putc('\n', fp);

	string calculated_hmac = calculateHMAC((char*)(ciphertext), password);
	fwrite(calculated_hmac.c_str(), HMAC_LENGTH, 1, fp);
	putc('\n', fp);

	fwrite(ciphertext, ciphertext_length, 1, fp);

	fclose(fp);
	fp = NULL;
}

bool readCryptFile(
	string file_path, 
	string password, 
	unsigned char **ciphertext, 
	int *ciphertext_length,
	int *error
){
	FILE *fp = fopen(file_path.c_str(), "rb");
	unsigned int offset = 0;
	int i;
	char ch;

	// Comparing Signature
	for(i=0 ; i<SIGNATURE_LENGTH ; i++){
		ch = fgetc(fp);
		if((int)ch != (int)AES_FILE_SIGNATURE[i]){
			*error = INVALID_SIGNATURE;
			return false;
		}
	}
	fgetc(fp);	

	// Comparing Password
	for(i=0 ; i<PASSWORD_LENGTH ; i++){
		ch = fgetc(fp);
		if(ch != password[i]){
			*error = INVALID_PASSWORD;
			return false;
		}
	}
	fgetc(fp);

	unsigned char file_hmac[HMAC_LENGTH];
	for(i=0 ; i<HMAC_LENGTH ; i++){
		file_hmac[i] = fgetc(fp);
	}
	fgetc(fp);	

	offset = ftell(fp);
	fseek(fp, 0L, SEEK_END);
	*ciphertext_length = ftell(fp) - offset;
	fseek(fp, offset, SEEK_SET);

	(*ciphertext) = (unsigned char*) malloc(*ciphertext_length);

	for(offset=0 ; offset < (unsigned int)*ciphertext_length ; offset++){
		(*ciphertext)[offset] = fgetc(fp);
	}

	string calculated_hmac = calculateHMAC((char*)(*ciphertext), password);
	short difference = memcmp(calculated_hmac.c_str(), file_hmac, HMAC_LENGTH);

	if(difference != 0){
		(*error) = INVALID_HMAC;
		return false;
	}

	fclose(fp);
	fp = NULL;

	return true;
}