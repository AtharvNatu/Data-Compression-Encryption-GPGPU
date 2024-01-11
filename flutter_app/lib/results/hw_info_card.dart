import "package:flutter/material.dart";
import 'package:flutter_app/lib.dart';

Widget labeledText(String label, String value, double fontSize) {
  return RichText(
    text: TextSpan(
      children: <TextSpan>[
        TextSpan(
          text: "$label:  ",
          style: TextStyle(
            fontSize: fontSize,
            fontFamily: "Cascadia Code",
            color: Colors.grey,
          ),
        ),
        TextSpan(
          text: value,
          style: TextStyle(
            fontSize: fontSize,
            fontWeight: FontWeight.bold,
            fontFamily: "Cascadia Code",
            color: Colors.black,
          ),
        )
      ],
    ),
  );
}

// ignore: must_be_immutable
class HwInfoCard extends StatelessWidget {
  HwInfoCard({super.key});

  double hardwareInfoBoxFontSize = 22;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(50.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          labeledText(
              "Operating System", platformInfo, hardwareInfoBoxFontSize),
          const SizedBox(height: 8),
          labeledText("CPU", cpuName, hardwareInfoBoxFontSize),
          const SizedBox(height: 8),
          labeledText("Cores", cpuCores, hardwareInfoBoxFontSize),
          labeledText("Threads", cpuThreads, hardwareInfoBoxFontSize),
          const SizedBox(height: 8),
          labeledText("RAM", "$totalMemory GB", hardwareInfoBoxFontSize),
          const SizedBox(height: 8),
          labeledText("GPU", gpuInfo, hardwareInfoBoxFontSize),
        ],
      ),
    );
  }
}
