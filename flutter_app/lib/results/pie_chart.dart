import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';


class InfoPieChart extends StatefulWidget{
  final double timeTakenCPU;
  final double timeTakenGPU;

  const InfoPieChart({
    Key? key, 
    required this.timeTakenCPU, 
    required this.timeTakenGPU
  }) : super(key: key);

  @override
  State<InfoPieChart> createState() => _InfoPieChart();
}

class _InfoPieChart extends State<InfoPieChart>{
  int touchedIndex = -1;

  List<PieChartSectionData> showingSections() {
    double cpuTime = widget.timeTakenCPU;
    double gpuTime = widget.timeTakenGPU;
      
    return List.generate(2, (i) {
      final isTouched = i == touchedIndex;
      final fontSize = isTouched ? 25.0 : 16.0;
      final radius = isTouched ? 60.0 : 50.0;
      const shadows = [Shadow(color: Colors.black, blurRadius: 2)];

      switch (i) {
        case 0:
          return PieChartSectionData(
            color: Colors.deepOrange,
            value: cpuTime,
            title: '$cpuTime Seconds',
            radius: radius,
            titleStyle: TextStyle(
              fontSize: fontSize,
              fontWeight: FontWeight.bold,
              color: Colors.black,
              shadows: shadows,
            ),
          );
        case 1:
          return PieChartSectionData(
            color: Colors.green,
            value: gpuTime,
            title: '$gpuTime Seconds',
            radius: radius,
            titleStyle: TextStyle(
              fontSize: fontSize,
              fontWeight: FontWeight.bold,
              color: Colors.black,
              shadows: shadows,
            ),
          );
        default:
          throw Error();
      }
    });
  }

  @override
  Widget build(BuildContext context){
    return PieChart(
      PieChartData(
        pieTouchData: PieTouchData(
          touchCallback: (FlTouchEvent event, pieTouchResponse) {
            setState(() {
              if (!event.isInterestedForInteractions ||
                  pieTouchResponse == null ||
                  pieTouchResponse.touchedSection == null) {
                touchedIndex = -1;
                return;
              }
              touchedIndex = pieTouchResponse
                  .touchedSection!.touchedSectionIndex;
            });
          },
        ),
        borderData: FlBorderData(
          show: false,
        ),
        sectionsSpace: 0,
        centerSpaceRadius: 80,
        sections: showingSections(),
      ),
    );
  }
}