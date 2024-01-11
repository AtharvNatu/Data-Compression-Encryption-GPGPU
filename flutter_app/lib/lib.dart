import "dart:ffi";
import "package:ffi/ffi.dart";
import "package:logger/logger.dart";

// Global Variables
var logger = Logger(
  printer: PrettyPrinter(),
);

late String platformInfo,
    cpuName,
    cpuCores,
    cpuThreads,
    totalMemory,
    gpuInfo,
    dialogText;

String oclEncryptKernelPath = "", oclDecryptKernelPath = "";
late int operationMode;
bool isCUDA = false, isOpenCL = false;
late double cpuTime, gpuTime;

// Typedefs
typedef DartFunc = Double Function(
    Pointer<Utf8> inputPath, Pointer<Utf8> outputPath, Pointer<Utf8> password);
typedef NativeFunc = double Function(
    Pointer<Utf8> inputPath, Pointer<Utf8> outputPath, Pointer<Utf8> password);

typedef DartOCLFunc = Double Function(
    Pointer<Utf8> inputPath,
    Pointer<Utf8> outputPath,
    Pointer<Utf8> password,
    Pointer<Utf8> oclPlatform);
typedef NativeOCLFunc = double Function(Pointer<Utf8> inputPath,
    Pointer<Utf8> outputPath, Pointer<Utf8> password, Pointer<Utf8> kernelPath);

typedef RunGPUScriptFunc = Void Function();
typedef RunGPUScript = void Function();

late int Function(RunGPUScriptFunc) getCudaSmCount, getOpenCLComputeCount;

late String gpuInfoLibPath, cpuLibPath, oclLibPath, cudaLibPath;
late DynamicLibrary gpuInfoDynamicLib,
    cpuDynamicLib,
    oclDynamicLib,
    cudaDynamicLib;

// CPU
late double Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>) aesCPUEncrypt,
    aesCPUDecrypt,
    aesCPUHuffmanEncrypt,
    aesCPUHuffmanDecrypt;

// CUDA
late double Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>)
    aesCUDAEncrypt,
    aesCUDADecrypt,
    aesCUDAHuffmanEncrypt,
    aesCUDAHuffmanDecrypt;

// OpenCL
late double Function(Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>, Pointer<Utf8>)
    aesOpenCLEncrypt,
    aesOpenCLDecrypt,
    aesOpenCLHuffmanEncrypt,
    aesOpenCLHuffmanDecrypt;
