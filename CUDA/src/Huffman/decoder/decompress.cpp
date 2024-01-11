#if defined (_WIN32) || defined (_WIN64) || defined (WIN32) || defined (WIN64)
    #include <windows.h>
#endif

#include <iostream>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../../../include/Huffman/decoder//huffman.h"
#include "../../../include/Common/helper_timer.h"

using namespace std;

void fatalDecompress(const char* str){
	fprintf( stderr, "%s! at %s in %d\n", str, __FILE__, __LINE__);
	exit(EXIT_FAILURE);
}

double decompress(string input_file, string output_file){
	FILE *inputfile, *outputfile;
	size_t tmp_st;
	int symbol_count;
	struct Symbol *symbols;

	unsigned int* input;
	unsigned int* output;

	inputfile = fopen( input_file.c_str(), "rb");
	outputfile = fopen( output_file.c_str(), "wb" );

	if ( 1 != fread( &tmp_st, sizeof(size_t), 1, inputfile )) fatalDecompress("File read error 1");
	symbol_count = tmp_st;

	cudaMallocHost( &symbols, sizeof(struct Symbol)*symbol_count );

	for(int i=0; i<symbol_count; i++){
		unsigned char tmpsymbol;
		unsigned char tmplength;
		if( 1 != fread( &tmpsymbol, sizeof(tmpsymbol), 1, inputfile) ) fatalDecompress("File read error 2");
		if( 1 != fread( &tmplength, sizeof(tmplength), 1, inputfile) ) fatalDecompress("File read error 3");
		symbols[i].symbol = tmpsymbol;
		symbols[i].length = tmplength;
	}

	unsigned int prefix_bit=0;
	prefix_bit = FIXED_PREFIX_BIT;
	struct TableInfo h_tableinfo;
	void* stage_table;

	int stage_table_size;

	double infosum=0,
		   tablesum=0;
	for(int i=0; i<LOOP; i++){
		// CPU_
		prefix_bit = get_table_info(
			symbols,
			symbol_count,
			prefix_bit,
			h_tableinfo);
		// CPU_
		// infosum += timetableinfo;
	}

	stage_table_size =  sizeof(int)*h_tableinfo.ptrtable_size + sizeof(char)*(MAX_CODE_NUM + h_tableinfo.l1table_size + h_tableinfo.l2table_size);
	cudaMallocHost(	&stage_table, stage_table_size);

	get_twolevel_table(
		stage_table,
		prefix_bit,
		symbols, 
		symbol_count,
		h_tableinfo );

	int compressed_size, original_size, gaparray_size, gap_elements_num;
	if( 1 != fread( &original_size, sizeof(int), 1, inputfile ) ) fatalDecompress("File read error 1");
	if( 1 != fread( &compressed_size, sizeof(int), 1, inputfile ) ) fatalDecompress("File read error 2");
	if( 1 != fread( &gap_elements_num, sizeof(int), 1, inputfile ) ) fatalDecompress("File read error 3");

	gaparray_size = (gap_elements_num + GAP_FAC_NUM-1)/GAP_FAC_NUM;

	cudaMallocHost( &input, sizeof(int)*(gaparray_size + compressed_size) );
	cudaMallocHost( &output, sizeof(int)*((original_size+3)/4) );

	if( (size_t)(gaparray_size+compressed_size) != fread( input, sizeof(unsigned int), gaparray_size + compressed_size, inputfile )) fatalDecompress("File read error");

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	
	cudaDeviceSynchronize();
	sdkStartTimer(&timer);
	decoder_l1_l2(
		input,
		compressed_size,
		output,
		original_size,
		gap_elements_num,
		stage_table,
		stage_table_size,
		prefix_bit,
		symbol_count,
		h_tableinfo	
	);
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);

	double decompression_time = sdkGetTimerValue(&timer);

	sdkDeleteTimer(&timer);
	timer = NULL;


	fwrite(output, sizeof(char), original_size, outputfile);

	fclose(inputfile);

	cudaFreeHost(symbols);
	cudaFreeHost(input);
	cudaFreeHost(output);
	cudaFreeHost(stage_table);

	// printf("Decompression Time = %lf\n", decompression_time);
	return decompression_time;
}
