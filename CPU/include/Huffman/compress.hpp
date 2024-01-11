#pragma once

#if defined (_WIN32) || defined (_WIN64) || defined (WIN32) || defined (WIN64)
    #include <windows.h>
#endif

#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include "../Common/Macros.hpp"
#include "utils.hpp"

#ifndef _HELPER_TIMER_H
#define _HELPER_TIMER_H
#include "../Common/helper_timer.h"
#endif

using namespace std;

HuffNode* combineNodes(HuffNode *a, HuffNode *b);

map<char, ullint> coutCharFrequency(string input_file_name, ullint input_file_size);

vector<HuffNode*> sortByCharacterCount(const std::map<char, ullint>&value);
	
double compress(
	string input_file_name, 
	unsigned char **compressed_file_buffer, 
	ullint compressed_size_wo_header,
	ullint input_file_size,
	ullint *compressed_file_size,
	string *compressed_value
);

string generateHeader(string *compressed_value, const char padding);

HuffNode* generateHuffmanTree(const map<char, ullint>&value);

ullint storeHuffmanValue(const HuffNode *root, string &value, string *compressed_value);
