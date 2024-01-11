import 'package:flutter/material.dart';
import 'live_mode.dart';
import 'benchmark.dart';
import "lib.dart";

class UserChoice extends StatelessWidget {
  final String title;
  final String imagePath;
  final String gpuName;
  int gpuOffset;

  UserChoice({
    required this.title,
    required this.imagePath,
    required this.gpuName,
    required this.gpuOffset,
  }) {
    gpuInfo = gpuName;
  }
  @override
  Widget build(BuildContext context) {
    Color benchmarkingCardColor =
        gpuName.contains("NVIDIA") ? Colors.green : Colors.deepOrange;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Operation Modes'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              'Select Your Preferred Mode Of Operation',
              style: TextStyle(
                fontFamily: "Cascadia Code",
                fontSize: 30,
                fontWeight: FontWeight.bold,
                color: Colors.black,
              ),
            ),
          ),

          const SizedBox(height: 30),
          // Row of cards
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Secure File Card
              Card(
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(15.0),
                ),
                color: Colors.blue,
                child: InkWell(
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (context) => FilePickLive(
                            title: "Secure File",
                            gpu: title,
                            imagePath: imagePath,
                            description: gpuName,
                            gpuName: gpuName,
                            gpuOffset: gpuOffset),
                      ),
                    );
                  },
                  child: Container(
                    width: 300,
                    height: 300,
                    padding: const EdgeInsets.all(16),
                    child: const Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.security, size: 100, color: Colors.black),
                        SizedBox(height: 30),
                        Text(
                          'Secure File',
                          style: TextStyle(
                            fontFamily: "Cascadia Code",
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: Colors.black,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),

              // Benchmarking Card
              Card(
                color: benchmarkingCardColor,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(15.0),
                ),
                child: InkWell(
                  onTap: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (context) => FilePickBench(
                            title: "Benchmarking",
                            gpu: title,
                            imagePath: imagePath,
                            description: gpuName,
                            gpuName: gpuName,
                            gpuOffset: gpuOffset),
                      ),
                    );
                  },
                  child: Container(
                    width: 300,
                    height: 300,
                    padding: const EdgeInsets.all(16),
                    child: const Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.speed, size: 100, color: Colors.black),
                        SizedBox(height: 30),
                        Text(
                          'Benchmarking',
                          style: TextStyle(
                            fontFamily: "Cascadia Code",
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: Colors.black,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
