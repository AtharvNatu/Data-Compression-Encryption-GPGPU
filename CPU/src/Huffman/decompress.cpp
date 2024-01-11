#include "../../include/Huffman/decompress.hpp"

void generateHuffmanTree(HuffNode *const root, const string &codes, const unsigned char ch){
	HuffNode *traverse = root;
	int i=0;


	while(codes[i] != '\0')	{
		if(codes[i] == '0'){
			if(!traverse->left){
				traverse->left = new HuffNode(0);
			}
			traverse = traverse->left;
		}
		else {
			if(!traverse->right){
				traverse->right = new HuffNode(0);
			}
			traverse = traverse->right;
		}
		++i;
	}
	traverse->ch = ch;
}

pair<HuffNode*, pair<unsigned char, int>> decodeHeader(unsigned char* input_file, ullint *offset){
	HuffNode *root = new HuffNode(0);
	int char_count, header_length=1;
	size_t buffer;
	char ch, len;

	char_count = input_file[(*offset)];
	(*offset)++;
	string codes;
	++char_count;
	// cout << "Start: " << char_count << endl;

	// printf("Total characters: %d\n", char_count);

	while(char_count){
		ch = input_file[(*offset)++];
		len = input_file[(*offset)++];
		codes = "";
		buffer = len;

		while(buffer > codes.size()){
			codes.push_back(input_file[(*offset)++]);
		}

		header_length += codes.size() + 2;
		// printf("%c Code = %s len=%d header_length=%d\n", ch, codes.c_str(), len, header_length);

		generateHuffmanTree(root, codes, ch);
		--char_count;
	}
	// cout <<" overrrrrrrr";

	unsigned char padding = input_file[(*offset)++];
	++header_length;
	// printf("Padding = %d\n", padding);
	return {root, {padding, header_length}};
}


double decompress(unsigned char* compressed_file, string output_file_name, ullint input_file_size){
	FILE *o_ptr;
	openFile(&o_ptr, output_file_name, "wb");

	ullint size = 0, compressed_offset = 0;

	double decompression_time = 0.0f;
	StopWatchInterface *timer;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	pair<HuffNode*, pair<unsigned char, int>>header_metadata = decodeHeader(compressed_file, &compressed_offset);
	HuffNode *const root = header_metadata.first;
	const auto [padding, header_size] = header_metadata.second;

	char ch, counter = 7;
	const ullint filesize = input_file_size - header_size;
	HuffNode *traverse = root;

	ch = compressed_file[compressed_offset++];

	while(size != filesize){
		while(counter >= 0){

			traverse = ch & (1<<counter) ? traverse->right : traverse->left;
			ch ^= (1 << counter);
			--counter;
			if(!traverse->left && !traverse->right){
				fputc(traverse->ch, o_ptr);
				if(size == filesize-1 && padding == counter+1){
					break;
				}
				traverse = root;
			}
		}
		++size;
		counter = 7;
		ch = compressed_file[compressed_offset++];
	}

	sdkStopTimer(&timer);
	decompression_time = (double) sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
	
	fclose(o_ptr);

	return decompression_time;
}