#pragma once

#if defined (_WIN32) || defined (_WIN64) || defined (WIN32) || defined (WIN64)
    #include <windows.h>
#endif

#include <iostream>
#include "utils.hpp"
#include "../Common/Macros.hpp"

#ifndef _HELPER_TIMER_H
#define _HELPER_TIMER_H
#include "../Common/helper_timer.h"
#endif

using namespace std;

void generateHuffmanTree(HuffNode *const root, const string &code, const unsigned char ch);

pair<HuffNode*, pair<unsigned char, int>> decodeHeader(unsigned char* input_file, ullint *offset);

double decompress(unsigned char* input_file, string output_file_name, ullint input_file_size);