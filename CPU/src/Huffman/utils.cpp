#include "../../include/Huffman/utils.hpp"

bool checkFileExistance(string file_path){
	if(filesystem::exists(file_path)){
		return true;
	}
	return false;
}

ullint getFileSizeBytes(string file_path){
	if(checkFileExistance(file_path)){
		return filesystem::file_size(file_path);
	}
	cout << "FILE DOES NOT EXIST - " << file_path << endl;
	exit(EXIT_FAILURE);
}

void openFile(FILE **fptr, string file_path, string access_modifier){
	*fptr = fopen(file_path.c_str(), access_modifier.c_str());
	if(*fptr == NULL){
		cout << "CANNOT OPEN FILE - " << file_path.c_str() << endl;
		exit(EXIT_FAILURE) ;
	}
}

string removeNameSuffix(string file_path){
	for(int i=file_path.size()-1 ; i>=0 ; i--){
		if(file_path.at(i) == '.'){
			return file_path.substr(0, i);
		}
	}
	return file_path;
}

void printchar(char c){
	for(int i=7; i>=0 ; i--){
		if(c & (1<<i)){
			printf("1");
		} else{
			printf("0");
		}
	}
	puts("");
}
