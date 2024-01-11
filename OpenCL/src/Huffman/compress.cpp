#include "../../include/Huffman/compress.hpp"

HuffNode* combineNodes(HuffNode *a, HuffNode *b){
	HuffNode *parent = new HuffNode((a?a->count:0) + (b?b->count:0));
	parent->left = b;
	parent->right = a;
	return parent;
}

bool sortByNodeCount(const HuffNode *a, const HuffNode *b){
	return (a->count > b->count);
}

map<char, ullint> coutCharFrequency(string input_file_name, ullint input_file_size){
	FILE *ptr;
	unsigned char ch;
	ullint size = 0;
	vector<ullint> char_count(256, 0);

	openFile(&ptr, input_file_name, "rb");

	while(size != input_file_size){
		ch = fgetc(ptr);
		++char_count[ch];
		++size;
	}

	map<char, ullint> store;
	for(int i=0 ; i<TOTAL_CHARS ; i++){
		if(char_count[i]){
			store[i] = char_count[i];
		}
	}

	fclose(ptr);
	return store;
}

vector<HuffNode*> sortByCharacterCount(const std::map<char, ullint>&value){
	vector<HuffNode*> store;
	for(auto &x: value){
		store.push_back(new HuffNode(x.first, x.second));
	}

	sort(store.begin(), store.end(), sortByNodeCount);
	
	return store;
}

string generateHeader(string *compressed_value, const char padding){
	string header = "";
	// unique character start from -1 (0 means 1, 1 means 2..., to conserve memroy)
	unsigned char unique_character = 255;

	for(int i=0 ; i<TOTAL_CHARS ; i++){
		if(compressed_value[i].size()){
			// cout << "ch = " << (char)i << endl;
			header.push_back(i);
			header.push_back(compressed_value[i].size());
			header += compressed_value[i];
			// cout << header << endl;
			++unique_character;
		}
	}

	char value = unique_character;
	// cout << "Unique Headers = " << value << "->" << (int)value << endl;
	// cout << "Header = " << value + header + (char)padding << endl;
	return value + header + (char)padding;
}

ullint storeHuffmanValue(const HuffNode *root, string &value, string *compressed_value){
	ullint total_compressed_size = 0;

	if(root){
		value.push_back('0');
		// cout << "value = " << value << endl;
		total_compressed_size = storeHuffmanValue(root->left, value, compressed_value);
		value.pop_back();

		if(!root->left && !root->right){
			compressed_value[(unsigned char)root->ch] = value;
			// printf("ch=%c = %s\n", root->ch, value.c_str());
			total_compressed_size += value.size() * root->count;
		}

		value.push_back('1');
		// cout << "value = " << value << endl;
		total_compressed_size += storeHuffmanValue(root->right, value, compressed_value);
		value.pop_back();
	}
	return total_compressed_size;
}

HuffNode* generateHuffmanTree(const map<char, ullint>&value){
	vector<HuffNode*> store = sortByCharacterCount(value);
	HuffNode *one, *two, *parent;
	// for(auto v: store){
	// 	cout << v->ch << " ";
	// }puts("");

	sort(begin(store), end(store), sortByNodeCount);

	// for(auto v: store){
	// 	cout << v->ch << " ";
	// }puts("");

	if(store.size() == 1){
		return combineNodes(store.back(), nullptr);
	}

	while(store.size() > 2){
		one = *(store.end() - 1);
		two = *(store.end() - 2);
		parent = combineNodes(one, two);
		store.pop_back();
		store.pop_back();
		store.push_back(parent);

		vector<HuffNode*>::iterator it1 = store.end() - 2;
		while((*it1)->count < parent->count && it1 != begin(store)){
			--it1;
		}
		sort(it1, store.end(), sortByNodeCount);
	}

	one = *(store.end() - 1);
	two = *(store.end() - 2);
	return combineNodes(one, two);
}

double compress(
	string input_file_name, 
	unsigned char **compressed_file_buffer, 
	ullint compressed_size_wo_header,
	ullint input_file_size,
	ullint *compressed_file_size,
	string *compressed_encodings
){
	const char padding = (8 - ((compressed_size_wo_header) & (7))) & (7);
	FILE *i_ptr;
	string header = generateHeader(compressed_encodings, padding);
	bool write_remaining=true;
	ullint compress_offset = 0, size = 0, i=0;
	unsigned char ch, fch=0;
	char counter = 7;
	
	double compressionTime = 0.0f;
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	*compressed_file_size = header.size() + (compressed_size_wo_header/8) + ((compressed_size_wo_header%8)!=0);
	// cout << "compressed file size = " << *compressed_file_size << endl;
	*compressed_file_buffer = (unsigned char*) malloc(sizeof(char) * (*compressed_file_size) + 1);

	if(*compressed_file_buffer == NULL){
		printf("Cannot allocate memory\n");
		exit(EXIT_FAILURE);
	}
	
	openFile(&i_ptr, input_file_name, "rb");

	i=0;
	while(i < header.size()){
		(*compressed_file_buffer)[compress_offset] = header[i];
		compress_offset++;
		i++;
	}
	
	while(size != input_file_size){
		ch = fgetc(i_ptr);
		i = 0;

		const string &huffman_coded = compressed_encodings[ch];

		while(huffman_coded[i] != '\0'){
			write_remaining = true;
			fch = fch | ((huffman_coded[i] - '0') << counter);
			counter = (counter + 7) & 7;
			
			if(counter == 7){
				write_remaining = false;
				(*compressed_file_buffer)[compress_offset] = fch;
				fch ^= fch;
				compress_offset++;
			}
			++i;
		}
		++size;
	}

	if(write_remaining){
		(*compressed_file_buffer)[compress_offset] = fch;
		compress_offset++;
	}
	(*compressed_file_buffer)[compress_offset] = '\n';
	// cout << "ending compressed file size = " << compress_offset << endl;
	
	sdkStopTimer(&timer);
	compressionTime = (double) sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
	
	fclose(i_ptr);

	return compressionTime;
}
