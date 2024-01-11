@echo off

cls

@REM OpenCL Windows

cd "./bin/"

@REM @REM For Executable
@REM cl.exe /c /EHsc /std:c++20 /I "C:\KhronosOpenCL\include" "../export/main.cpp" "../src/CLFW/CLFW.cpp" "../src/Common/Helper.cpp" "../src/Common/sha256.cpp" "../src/Common/hmac.cpp" "../src/AES/AES.cpp" "../src/Huffman/compress.cpp" "../src/Huffman/decompress.cpp" "../src/Huffman/utils.cpp"
@REM link.exe /OUT:AES.exe main.obj AES.obj Helper.obj sha256.obj hmac.obj compress.obj decompress.obj utils.obj CLFW.obj /LIBPATH:"C:\KhronosOpenCL\lib"
@REM @copy AES.exe "../" > nul

@REM For DLL
echo Generating DLL ...
cl.exe /c /EHsc /std:c++20 /I "C:\KhronosOpenCL\include" "../export/lib.cpp" "../src/CLFW/CLFW.cpp" "../src/Common/Helper.cpp" "../src/Common/sha256.cpp" "../src/Common/hmac.cpp" "../src/AES/AES.cpp" "../src/Huffman/compress.cpp" "../src/Huffman/decompress.cpp" "../src/Huffman/utils.cpp"
link.exe /DLL /OUT:DCAEUG-OpenCL.dll lib.obj AES.obj Helper.obj sha256.obj hmac.obj compress.obj decompress.obj utils.obj CLFW.obj /LIBPATH:"C:\KhronosOpenCL\lib"
@move DCAEUG-OpenCL.dll "../" > nul

cd ../


