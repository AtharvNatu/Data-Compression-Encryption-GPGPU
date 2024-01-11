// Usage: bin/encoder input output
#include <iostream>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../../../include/Huffman/encoder/huffman.h"
#include "../../../include/Common/helper_timer.h"

using namespace std;

void fatalCompress(const char* str){
	printf("%s", str);
	fflush(stdout);
	exit(EXIT_FAILURE);
}

int compare( const void *p, const void *q ){
	struct Symbol *P = (struct Symbol *) p;
	struct Symbol *Q = (struct Symbol *) q;
	return P->num - Q->num;
}

double compress(string input_file, string output_file){
	FILE *input, *output;
	unsigned int *inputfile;
	unsigned int *outputfile;
	unsigned int *gap_array;
	unsigned int inputfilesize=0;
	unsigned int outputfilesize=0;
	unsigned long long int outputfilesize_bits=0;
	unsigned int gap_array_size = 0;
	unsigned int gap_array_elements_num = 0;
	unsigned int *num_of_symbols;
	num_of_symbols = (unsigned int *)malloc(sizeof(int)*MAX_CODE_NUM);
	struct Symbol symbols[MAX_CODE_NUM] = {};
	struct Codetable *codetable;

	input = fopen(input_file.c_str(), "rb");
	output = fopen(output_file.c_str(), "wb");

	fseek(input, 0L, SEEK_END);
	inputfilesize = ftell(input);
	fseek(input, 0L, SEEK_SET);

	cudaMallocHost( &codetable, sizeof(struct Codetable)*MAX_CODE_NUM );
	cudaMallocHost( &inputfile, sizeof(int)*((inputfilesize+3)/4) );

	fread(inputfile, sizeof(char), inputfilesize, input);
	// fsync(input->_fileno);
	fflush(input);

	int symbol_count = 0;

	unsigned int *d_num_of_symbols;
	unsigned int *d_inputfile;
	cudaMalloc((void **)&d_inputfile, sizeof(int)*((inputfilesize+3)/4 + 1 ));
	cudaMalloc((void **)&d_num_of_symbols, sizeof(int)*MAX_CODE_NUM);

	cudaMemcpy(d_inputfile, inputfile, sizeof(int)*((inputfilesize+3)/4), cudaMemcpyHostToDevice);
	CUERROR

	// compute the histogram
	histgram(d_num_of_symbols, d_inputfile, inputfilesize);
	cudaMemcpy(num_of_symbols, d_num_of_symbols, sizeof(int)*MAX_CODE_NUM, cudaMemcpyDeviceToHost);

	cudaFree(d_num_of_symbols);

	// make symbols
	symbol_count = store_symbols(num_of_symbols, symbols);
	qsort(symbols, symbol_count, sizeof(struct Symbol) ,compare);

	boundary_PM(symbols, symbol_count, codetable);

	outputfilesize_bits = get_outputfilesize(symbols, symbol_count);
	gap_array_elements_num = (outputfilesize_bits + SEGMENTSIZE-1)/SEGMENTSIZE;
	gap_array_size = gap_array_elements_num/GAP_ELEMENTS_NUM + ((gap_array_elements_num%GAP_ELEMENTS_NUM) != 0);

	outputfilesize = (outputfilesize_bits+MAX_BITS-1)/MAX_BITS;

	cudaMallocHost( &outputfile, sizeof(unsigned int)*(outputfilesize+gap_array_size) );
	gap_array = outputfile + outputfilesize;

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	
	cudaDeviceSynchronize();
	sdkStartTimer(&timer);
	{
		encode(
			outputfile, 
			outputfilesize, 
			d_inputfile, 
			inputfilesize, 
			gap_array_elements_num,
			codetable 
		);
		cudaDeviceSynchronize();
	}
	sdkStopTimer(&timer);
	double compression_time = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;

	CUERROR;
		
	// printf("file,%s \n",argv[1]);
	// printf("SEGMENTSIZE,%d \n",SEGMENTSIZE);
	// printf("THREAD_NUM,%d \n",THREAD_NUM);
	// printf("Bytes_per_THREAD,%d \n",THREAD_ELEMENT);
	// printf("\n");

	//------------------------------------------------------------ 

	size_t tmp_symbol_count = symbol_count;
	fwrite( &tmp_symbol_count, sizeof(size_t), 1, output );
	for(int i=symbol_count-1; i>=0; i--){
		unsigned char tmpsymbol = symbols[i].symbol;
		unsigned char tmplength = symbols[i].length;
		fwrite( &tmpsymbol, sizeof(tmpsymbol), 1, output );
		fwrite( &tmplength, sizeof(tmplength), 1, output );
	}

	fwrite( &inputfilesize, sizeof(inputfilesize), 1, output );
	fwrite( &outputfilesize, sizeof(outputfilesize), 1, output );
	fwrite( &gap_array_elements_num, sizeof(gap_array_elements_num), 1, output );
	fwrite( gap_array, sizeof(int), gap_array_size, output );
	fwrite( outputfile, sizeof(unsigned int), outputfilesize, output);

	// fdatasync(output->_fileno);
	fflush(output);

	int outsize = ftell(output);
	// printf("ratio=outputfile/inputfile,%lf", (double)outsize/inputfilesize);
	// printf(",outputfilesize,%d,inputfilesize,%d\n", outsize, inputfilesize);
	// printf("\n");

	fclose(input);
	fclose(output);

	cudaFreeHost(inputfile);
	cudaFreeHost(outputfile);
	cudaFreeHost(codetable);

	// printf("compression_time = %lf\n", compression_time);
	return compression_time;
}

