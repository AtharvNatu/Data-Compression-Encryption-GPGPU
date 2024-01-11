import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:file_selector/file_selector.dart';
import "lib.dart";

class FilePickLive extends StatefulWidget {
  final String title;
  final String imagePath;
  final String description;
  final String gpu;

  FilePickLive({
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

class ChooseFile extends State<FilePickLive> {
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

  bool hide = true;

  List<String> allowedExtensions = [
    'jpg',
    'png',
    'bmp',
    'pdf',
    'txt',
    'mp4',
    'mov',
    'docx',
    'heif',
    'jpeg',
    'tif',
    'enc'
  ];
  String fileExt = "";
  String selectedFileName = "";
  TextEditingController keyController = TextEditingController();
  String submittedKey = "";
  String filePath = "";
  String outputDirPath = "";
  final bool _isIOS = !kIsWeb && defaultTargetPlatform == TargetPlatform.iOS;

  Future<void> _getDirectoryPath() async {
    const String confirmButtonText = 'Choose';
    final String? directoryPath = await getDirectoryPath(
      confirmButtonText: confirmButtonText,
    );
    if (directoryPath == null) {
      return;
    }
    logger.i("Output Path --> $directoryPath");
    outputDirPath = directoryPath;
  }

  void _openFilePicker() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
        allowMultiple: false,
        dialogTitle: 'Select A File You Would Like to Secure',
        type: FileType.custom,
        allowedExtensions: [
          'jpg',
          'png',
          'bmp',
          'pdf',
          'txt',
          'mp4',
          'mov',
          'docx',
          'heif',
          'jpeg',
          'tif',
          'enc'
        ]);

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
        // ignore: use_build_context_synchronously
        showDialog(
          context: context,
          builder: (BuildContext context) {
            return AlertDialog(
              title: const Text("😑❌"),
              content: const Text("Please select an appropriate file !!!"),
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
    if (keyController.text.isNotEmpty) {
      setState(() {
        submittedKey = keyController.text;
      });

      final inputPath = filePath.toNativeUtf8();
      final outputPath = outputDirPath.toNativeUtf8();
      final password = submittedKey.toNativeUtf8();
      final oclEncPath = oclEncryptKernelPath.toNativeUtf8();
      final oclDecPath = oclDecryptKernelPath.toNativeUtf8();

      if (fileExt != "txt" && fileExt != "enc") {
        logger.w("1");
        operationMode = 1;
        gpuTime = gpuInfo.contains("NVIDIA")
            ? aesCUDAEncrypt(inputPath, outputPath, password)
            : aesOpenCLEncrypt(inputPath, outputPath, password, oclEncPath);
      } else if (fileExt == "enc" &&
          selectedFileName.contains("txt.huff.enc") &&
          gpuInfo.contains("NVIDIA")) {
        logger.w("2");
        operationMode = 2;
        final huffDecPath =
            "${outputDirPath.substring(0, outputDirPath.length - 4)}.huff.enc"
                .toNativeUtf8();
        gpuTime = aesCUDAHuffmanDecrypt(inputPath, huffDecPath, password);
        calloc.free(huffDecPath);
      } else if (fileExt == "enc" && !selectedFileName.contains("txt.enc")) {
        logger.w("3");
        operationMode = 2;
        gpuTime = gpuInfo.contains("NVIDIA")
            ? aesCUDADecrypt(inputPath, outputPath, password)
            : aesOpenCLDecrypt(inputPath, outputPath, password, oclDecPath);
      } else if (fileExt == "txt") {
        logger.w("4");
        operationMode = 1;
        gpuTime = gpuInfo.contains("NVIDIA")
            ? aesCUDAHuffmanEncrypt(inputPath, outputPath, password)
            : aesOpenCLHuffmanEncrypt(
                inputPath, outputPath, password, oclEncPath);
      } else if (fileExt == "enc" && selectedFileName.contains("txt.enc")) {
        logger.w("5");
        operationMode = 2;
        aesOpenCLHuffmanDecrypt(inputPath, outputPath, password, oclDecPath);
      }

      logger.i("GPU Time : $gpuTime");

      switch (gpuTime.toInt()) {
        case -1:
        case -3:
          showDialog(
            context: context,
            builder: (BuildContext context) {
              return AlertDialog(
                title: const Text("😡❌"),
                content: const Text(
                    "Tampered File ... Please Check Input File Again !!!"),
                actions: [
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                    },
                    child: const Text("Ok"),
                  ),
                ],
              );
            },
          );
          break;

        case -2:
          showDialog(
            context: context,
            builder: (BuildContext context) {
              return AlertDialog(
                title: const Text("😟❌"),
                content: const Text(
                    "Invalid Password ... Please Check Your Password !!!"),
                actions: [
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                    },
                    child: const Text("Ok"),
                  ),
                ],
              );
            },
          );
          break;

        default:
          switch (operationMode) {
            case 1:
              dialogText = "File Encrypted Successfully ...";
              break;
            case 2:
              dialogText = "File Decrypted Successfully ...";
              break;
          }
          showDialog(
            context: context,
            builder: (BuildContext context) {
              return AlertDialog(
                title: const Text("😄✅"),
                content: Text(dialogText),
                actions: [
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                    },
                    child: const Text("Ok"),
                  ),
                ],
              );
            },
          );
          break;
      }

      gpuTime /= 1000;
      gpuTime = double.parse(gpuTime.toStringAsFixed(2));

      calloc.free(oclDecPath);
      calloc.free(oclEncPath);
      calloc.free(password);
      calloc.free(outputPath);
      calloc.free(inputPath);
    } else {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: const Text("😑❌"),
            content: const Text(
                "Empty Password ... Please Enter A Valid Password !!!"),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop();
                },
                child: const Text("Ok"),
              ),
            ],
          );
        },
      );
    }
  }

  @override
  void dispose() {
    keyController.dispose();
    super.dispose();
  }

  bool _isObscured = true;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('GPGPU Based File Security'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: _openFilePicker,
                  style: ElevatedButton.styleFrom(
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
                  child: const Text('Select File'),
                ),
                const SizedBox(width: 20),
                ElevatedButton(
                  onPressed: _isIOS ? null : () => _getDirectoryPath(),
                  style: ElevatedButton.styleFrom(
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
                  child: const Text('Output Path'),
                ),
              ],
            ),

            //  DEBUGGING
            // ------------------------------------------------------------------
            // SizedBox(height: 20),
            // Text(
            //   "Output Path: " +outputP ,
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
            const SizedBox(height: 20),

            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 300, vertical: 20),
              child: TextField(
                controller: keyController,
                obscureText: _isObscured,
                decoration: InputDecoration(
                    border: OutlineInputBorder(),
                    labelText: 'Enter Password',
                    hintText: 'Keep A Strong Password !!!',
                    counterText: '',
                    suffixIcon: GestureDetector(
                      onTap: () {
                        setState(() {
                          _isObscured = !_isObscured;
                        });
                      },
                      child: Icon(_isObscured
                          ? Icons.visibility
                          : Icons.visibility_off),
                    )),
                maxLength: 16,
                minLines: 1,
              ),
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              onPressed: () {
                _submitKey();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('Password Entered'),
                  ),
                );
              },
              style: ElevatedButton.styleFrom(
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
              child: const Text('Process'),
            ),
            const SizedBox(height: 20),

            //DEBUG
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
