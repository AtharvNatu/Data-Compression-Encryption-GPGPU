#include <iostream>
#include <filesystem>
#include "../Common/Macros.hpp"

using namespace std;

ullint getFileSizeBytes(string file_path);

void openFile(FILE **fptr, string file_path, string access_modifier);

string removeNameSuffix(string file_path);

void printchar(char c);