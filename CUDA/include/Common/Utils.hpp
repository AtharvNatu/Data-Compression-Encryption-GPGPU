#pragma once
#include <iostream>
#include "Macros.hpp"

using namespace std;

void openFile(FILE **fptr, string file_name, string access_modifier);

unsigned int getFileSize(FILE *fptr);

unsigned long long int getOutputFileSize(struct Symbol *symbols, unsigned int symbols_count);