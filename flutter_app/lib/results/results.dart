import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'hw_info_card.dart';
import 'pie_chart.dart';
import 'bar_chart.dart';

class BarChartt extends StatefulWidget {
  final double timeTakenCPU;
  final double timeTakenGPU;

  const BarChartt(
      {Key? key, required this.timeTakenCPU, required this.timeTakenGPU})
      : super(key: key);

  @override
  State<StatefulWidget> createState() => PlotCharts();
}

class PlotCharts extends State<BarChartt> {
  int touchedIndex = -1;

  List<PieChartSectionData> showingSections() {
    return List.generate(2, (i) {
      final isTouched = i == touchedIndex;
      final fontSize = isTouched ? 25.0 : 16.0;
      final radius = isTouched ? 60.0 : 50.0;
      const shadows = [Shadow(color: Colors.black, blurRadius: 2)];

      switch (i) {
        case 0:
          return PieChartSectionData(
            color: Colors.deepOrange,
            value: widget.timeTakenCPU,
            title: '${widget.timeTakenCPU} Seconds',
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
            value: widget.timeTakenGPU,
            title: '${widget.timeTakenGPU} Seconds',
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
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Results'),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(vertical: 50, horizontal: 220),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: <Widget>[
            Row(
              mainAxisAlignment: MainAxisAlignment.start,
              children: [
                Container(
                  width: 1000,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(0.5),
                        spreadRadius: 5,
                        blurRadius: 7,
                        offset: const Offset(
                            0, 3), // changes the position of the shadow
                      ),
                    ],
                  ),
                  child: HwInfoCard(),
                )
              ],
            ),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: <Widget>[
                Container(
                  width: 1000,
                  height: 561,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(0.5),
                        spreadRadius: 5,
                        blurRadius: 5,
                        offset: const Offset(0, 3),
                      ),
                    ],
                  ),
                  child: InfoBarChart(
                      timeTakenCPU: widget.timeTakenCPU,
                      timeTakenGPU: widget.timeTakenGPU),
                ),
                Container(
                  width: 450,
                  height: 561,
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(10),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.grey.withOpacity(0.5),
                        spreadRadius: 5,
                        blurRadius: 5,
                        offset: const Offset(0, 3),
                      ),
                    ],
                  ),
                  child: InfoPieChart(
                    timeTakenCPU: widget.timeTakenCPU,
                    timeTakenGPU: widget.timeTakenGPU,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
