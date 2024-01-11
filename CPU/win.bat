@echo off

cls

@REM AES ECB Windows

cd "./bin/"

@REM For Executable
cl.exe /c /EHsc /std:c++20 "../export/main.cpp" "../src/AES/AES.cpp" "../src/Common/Helper.cpp" "../src/Common/sha256.cpp" "../src/Common/hmac.cpp" "../src/Huffman/compress.cpp" "../src/Huffman/decompress.cpp" "../src/Huffman/utils.cpp"
link.exe /OUT:App.exe main.obj AES.obj Helper.obj sha256.obj hmac.obj compress.obj decompress.obj utils.obj
@copy App.exe "../" > nul


@REM @REM For DLL
@REM echo Generating DLL ...
@REM cl.exe /c /EHsc /std:c++20 "../export/lib.cpp" "../src/AES/AES.cpp" "../src/Common/Helper.cpp" "../src/Common/sha256.cpp" "../src/Common/hmac.cpp" "../src/Huffman/compress.cpp" "../src/Huffman/decompress.cpp" "../src/Huffman/utils.cpp"
@REM link.exe /DLL /OUT:DCAEUG-CPU.dll lib.obj AES.obj Helper.obj compress.obj decompress.obj utils.obj sha256.obj hmac.obj
@REM @move DCAEUG-CPU.dll "../" > nul

cd ..


