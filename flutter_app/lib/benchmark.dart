import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'results/results.dart';
import 'package:path/path.dart' as path;
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import "lib.dart";

class FilePickBench extends StatefulWidget {
  final String title;
  final String imagePath;
  final String description;
  final String gpu;

  FilePickBench({
    required this.title,
    required this.imagePath,
    required this.description,
    required this.gpu,
    required String gpuName,
    required int gpuOffset,
  });

  @override
  ChooseFile createState() => ChooseFile();
}

class ChooseFile extends State<FilePickBench> {
  late String title;
  late String imagePath;
  late String description;
  late String gpu;

  @override
  void initState() {
    super.initState();
    title = widget.title;
    imagePath = widget.imagePath;
    description = widget.description;
    gpu = widget.gpu;
  }

  List<String> allowedExtensions = [
    'jpg',
    'png',
    'pdf',
    'txt',
    'mp4',
    'mov',
    'docx',
    'heif',
    'jpeg',
    'tif',
    'enc',
    'obj'
  ];
  String fileExt = "";
  String selectedFileName = "";
  TextEditingController keyController = TextEditingController();
  String submittedKey = "";
  String filePath = "";
  String outputFilePath = "";
  String trimmedPath = "";
  String outputDirPath = "";
  final bool _isIOS = !kIsWeb && defaultTargetPlatform == TargetPlatform.iOS;

  Future<void> _getDirectoryPath() async {
    const String confirmButtonText = 'Choose';
    final String? directoryPath = await getDirectoryPath(
      confirmButtonText: confirmButtonText,
    );
    if (directoryPath == null) {
      // Operation was canceled by the user.
      return;
    }
    logger.i("Output Path --> $directoryPath");
    outputDirPath = directoryPath;
  }

  void _selectPath() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowMultiple: false,
      dialogTitle: 'Select Output Path',
      type: FileType.custom,
      allowedExtensions: [
        'jpg',
        'png',
        'pdf',
        'txt',
        'mp4',
        'mov',
        'docx',
        'heif',
        'jpeg',
        'tif',
        'enc',
        'obj'
      ],
    );

    if (result != null) {
      String selectedExtension =
          result.files.first.extension?.toLowerCase() ?? "";

      if (allowedExtensions.contains(selectedExtension)) {
        setState(() {
          // Store the selected path in the new variable
          outputFilePath = result.files.first.path ?? "";
        });
      } else {
        // Show an alert dialog for inappropriate file extension
        // ignore: use_build_context_synchronously
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: const Text("ERROR ⚠️"),
              content: const Text("Please select an appropriate file."),
              actions: [
                TextButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: const Text("OK"),
                ),
              ],
            );
          },
        );
      }
    }

    _trimPath(outputFilePath);
  }

  void _trimPath(String fullPath) {
    String pathWithoutPrefix = fullPath.replaceFirst("file:///", "");
    String directory = path.dirname(pathWithoutPrefix);

    trimmedPath = path.join(directory, "");
    logger.i(trimmedPath);
  }

  void _openFilePicker() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowMultiple: false,
      dialogTitle: 'Select a File you would like to Compress',
      type: FileType.custom,
      allowedExtensions: [
        'jpg',
        'png',
        'pdf',
        'txt',
        'mp4',
        'mov',
        'docx',
        'heif',
        'jpeg',
        'tif',
        'enc',
        'obj'
      ],
    );

    if (result != null) {
      String selectedExtension =
          result.files.first.extension?.toLowerCase() ?? "";

      if (allowedExtensions.contains(selectedExtension)) {
        setState(() {
          selectedFileName = result.files.first.name;
          fileExt = selectedExtension;
          filePath = result.files.first.path ?? "";
        });
      } else {
        // Show an alert dialog for inappropriate file extension
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: const Text("ERROR ⚠️"),
              content: const Text("Please Select an appropriate file"),
              actions: [
                TextButton(
                  onPressed: () {
                    Navigator.of(context).pop();
                  },
                  child: const Text("OK"),
                ),
              ],
            );
          },
        );
      }
    }
  }

  void _submitKey() {
    //Default value for Benchmarking
    setState(() {
      submittedKey = "0000000000000000";
    });

    var tempDir = outputDirPath; // test/output/
    filePath; // test/output/CPU/abc.txt.enc

    if (Platform.isWindows) {
      tempDir = "$outputDirPath\\CPU\\";
    } else if (Platform.isLinux) {
      tempDir = "$outputDirPath/CPU/";
    }
    Directory(tempDir).create(recursive: false);

    if (Platform.isWindows) {
      tempDir = "$outputDirPath\\GPU\\";
    } else if (Platform.isLinux) {
      tempDir = "$outputDirPath/GPU/";
    }
    Directory(tempDir).create(recursive: false);

    final inputCPUPath = filePath.toNativeUtf8();
    filePath = filePath.replaceAll("CPU", "GPU");

    if (fileExt == "enc" &&
            gpuInfo.contains("NVIDIA") &&
            selectedFileName.contains("txt.enc") ||
        fileExt == "enc" &&
            gpuInfo.contains("NVIDIA") &&
            selectedFileName.contains("obj.enc")) {
      filePath = "${filePath.substring(0, filePath.length - 4)}.huff.enc";
    }
    final inputGPUPath = filePath.toNativeUtf8();

    if (Platform.isWindows) {
      tempDir = "$outputDirPath\\CPU\\";
    } else if (Platform.isLinux) {
      tempDir = "$outputDirPath/CPU/";
    }

    final outputCPUPath = tempDir.toNativeUtf8();
    if (Platform.isWindows) {
      tempDir = "$outputDirPath\\GPU\\";
    } else if (Platform.isLinux) {
      tempDir = "$outputDirPath/GPU/";
    }

    final outputGPUPath = tempDir.toNativeUtf8();

    final password = submittedKey.toNativeUtf8();
    final oclEncPath = oclEncryptKernelPath.toNativeUtf8();
    final oclDecPath = oclDecryptKernelPath.toNativeUtf8();

    if (fileExt != "txt" && fileExt != "enc" && fileExt != "obj") {
      cpuTime = aesCPUEncrypt(inputCPUPath, outputCPUPath, password);
      gpuTime = gpuInfo.contains("NVIDIA")
          ? aesCUDAEncrypt(inputGPUPath, outputGPUPath, password)
          : aesOpenCLEncrypt(inputGPUPath, outputGPUPath, password, oclEncPath);
    } else if (fileExt == "enc" && !selectedFileName.contains("txt.enc")) {
      cpuTime = aesCPUDecrypt(inputCPUPath, outputCPUPath, password);
      gpuTime = gpuInfo.contains("NVIDIA")
          ? aesCUDADecrypt(inputGPUPath, outputGPUPath, password)
          : aesOpenCLDecrypt(inputGPUPath, outputGPUPath, password, oclDecPath);
    } else if (fileExt == "txt" || fileExt == "obj") {
      cpuTime = aesCPUHuffmanEncrypt(inputCPUPath, outputCPUPath, password);
      gpuTime = gpuInfo.contains("NVIDIA")
          ? aesCUDAHuffmanEncrypt(inputGPUPath, outputGPUPath, password)
          : aesOpenCLHuffmanEncrypt(
              inputGPUPath, outputGPUPath, password, oclEncPath);
    } else if ((fileExt == "enc" && selectedFileName.contains("txt.enc")) ||
        (fileExt == "enc" && selectedFileName.contains("obj.enc"))) {
      cpuTime = aesCPUHuffmanDecrypt(inputCPUPath, outputCPUPath, password);
      gpuTime = gpuInfo.contains("NVIDIA")
          ? aesCUDAHuffmanDecrypt(inputGPUPath, outputGPUPath, password)
          : aesOpenCLHuffmanDecrypt(
              inputGPUPath, outputGPUPath, password, oclDecPath);
    }

    cpuTime /= 1000;
    cpuTime = double.parse(cpuTime.toStringAsFixed(2));
    gpuTime /= 1000;
    gpuTime = double.parse(gpuTime.toStringAsFixed(2));

    calloc.free(oclDecPath);
    calloc.free(oclEncPath);
    calloc.free(password);
    calloc.free(outputGPUPath);
    calloc.free(outputCPUPath);
    calloc.free(inputGPUPath);
    calloc.free(inputCPUPath);

    // FOR BENCHMARKING MODE ONLY!!!!!!!
    if (title == "Benchmarking") {
      Navigator.of(context).push(
        MaterialPageRoute(
            builder: (context) =>
                BarChartt(timeTakenCPU: cpuTime, timeTakenGPU: gpuTime)),
      );
    }
  }

  @override
  void dispose() {
    keyController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Color benchmarkingCardColor =
        description.contains("NVIDIA") ? Colors.green : Colors.deepOrange;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Benchmark Mode'),
        backgroundColor: benchmarkingCardColor,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Padding(
              padding: EdgeInsets.all(25.0),
              child: Text(
                'CPU v/s GPU Performance Analysis',
                style: TextStyle(
                  fontFamily: "Cascadia Code",
                  fontSize: 30,
                  fontWeight: FontWeight.bold,
                  color: Colors.black,
                ),
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: _openFilePicker,
                  child: const Text('Select File'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: benchmarkingCardColor,
                    fixedSize: const Size(180, 60),
                    textStyle: const TextStyle(
                      fontFamily: "Cascadia Code",
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                ),
                const SizedBox(width: 20),
                ElevatedButton(
                  onPressed: _isIOS ? null : () => _getDirectoryPath(),
                  child: const Text('Output Path'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: benchmarkingCardColor,
                    fixedSize: const Size(180, 60),
                    textStyle: const TextStyle(
                      fontFamily: "Cascadia Code",
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                ),
                const SizedBox(width: 20),
                ElevatedButton(
                  onPressed: () {
                    _submitKey();
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Entering Benchmarking Mode...'),
                      ),
                    );
                  },
                  child: const Text('Benchmark'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: benchmarkingCardColor,
                    fixedSize: const Size(180, 60),
                    textStyle: const TextStyle(
                      fontFamily: "Cascadia Code",
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(30),
                    ),
                  ),
                ),
              ],
            ),

            //DEBUG
            // //---------------------------------------------------------
            // SizedBox(height: 20),
            // Text(
            //   "FileName: " +selectedFileName,
            //   style: TextStyle(fontSize: 18),
            // ),
            // Text(
            //   "Output Path: " +trimmedPath,
            //   style: TextStyle(fontSize: 18),
            // ),
            // Text(
            //   "Mode: " + title,
            //   style: TextStyle(fontSize: 18),
            // ),
            // Text(
            //   selectedFileName,
            //   style: TextStyle(fontSize: 18),
            // ),
            // Text(
            //   "Extension: " + fileExt,
            //   style: TextStyle(fontSize: 18),
            // ),
            // Text(
            //   "Path: " + filePath,
            //   style: TextStyle(fontSize: 18),
            // ),
            // SizedBox(height: 20),
            // ---------------------------------------------------

            // SizedBox(height: 20),
            // Text(
            //   submittedKey,
            //   style: TextStyle(fontSize: 18),
            // ),
          ],
        ),
      ),
    );
  }
}
