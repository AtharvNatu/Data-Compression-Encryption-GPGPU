@echo off

cls

@REM AES ECB Windows

cd "./bin/"
 
echo Compiling Source Files ...
nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o AES-CUDA.obj ^
    "../src/AES/AES-CUDA.cu"

nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o decoder.obj ^
    "../src/Huffman/decoder/decoder.cu"

nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o get_table.obj ^
    "../src/Huffman/decoder/get_table.cpp"

nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o decompress.obj ^
    "../src/Huffman/decoder/decompress.cpp"

nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o compress.obj ^
    "../src/Huffman/encoder/compress.cpp"

nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o encoder.obj ^
    "../src/Huffman/encoder/encoder.cu"

nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o package_merge.obj ^
    "../src/Huffman/encoder/package_merge.cpp"

nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o symbols.obj ^
    "../src/Huffman/encoder/symbols.cpp"

@REM For Executable
@REM nvcc -c --std=c++20 ^
@REM     -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
@REM     -o Main.obj ^
@REM     "../export/main.cu"

@REM For DLL
nvcc -c --std=c++20 ^
    -I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" ^
    -o lib.obj ^
    "../export/lib.cu"

cl.exe /c /EHsc /std:c++20 "../src/Common/Helper.cpp" "../src/Common/sha256.cpp" "../src/Common/hmac.cpp" "../src/Common/utils.cpp"


@REM For Executable
@REM echo Linking Object Files ...
@REM link.exe /DEBUG /OUT:AES.exe main.obj Helper.obj sha256.obj hmac.obj utils.obj AES-CUDA.obj decoder.obj get_table.obj decompress.obj compress.obj encoder.obj package_merge.obj symbols.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" cudart.lib
@REM @copy AES.exe "../" > nul
@REM AES.exe

@REM For DLL
echo Generating DLL ...
link.exe /DLL /OUT:DCAEUG-CUDA.dll lib.obj Helper.obj sha256.obj hmac.obj utils.obj AES-CUDA.obj decoder.obj get_table.obj decompress.obj compress.obj encoder.obj package_merge.obj symbols.obj /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" cudart.lib
@move DCAEUG-CUDA.dll "../" > nul

cd ../


