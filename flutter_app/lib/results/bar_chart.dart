import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import "package:flutter_app/lib.dart";

class InfoBarChart extends StatelessWidget {
  final double timeTakenCPU;
  final double timeTakenGPU;

  const InfoBarChart(
      {Key? key, required this.timeTakenCPU, required this.timeTakenGPU})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(30.0),
      child: BarChart(
        BarChartData(
          barTouchData: barTouchData,
          titlesData: titlesData,
          borderData: borderData,
          barGroups: barGroups,
          gridData: const FlGridData(show: true),
          alignment: BarChartAlignment.spaceAround,
          maxY: timeTakenCPU,
        ),
      ),
    );
  }

  BarTouchData get barTouchData => BarTouchData(
        enabled: true,
        touchTooltipData: BarTouchTooltipData(
          tooltipBgColor: Colors.transparent,
          tooltipPadding: EdgeInsets.zero,
          tooltipMargin: 8,
          getTooltipItem: (
            BarChartGroupData group,
            int groupIndex,
            BarChartRodData rod,
            int rodIndex,
          ) {
            return BarTooltipItem(
              "${rod.toY} Seconds",
              const TextStyle(
                color: Colors.black,
                fontWeight: FontWeight.bold,
              ),
            );
          },
        ),
      );

  Widget getTitles(double value, TitleMeta meta) {
    const style = TextStyle(
      fontFamily: "Cascadia Code",
      color: Colors.black,
      fontWeight: FontWeight.bold,
      fontSize: 15,
    );
    String text;

    switch (value.toInt()) {
      case 0:
        text = cpuName;
        break;
      case 1:
        text = gpuInfo;
        break;
      default:
        text = '';
        break;
    }
    return SideTitleWidget(
      axisSide: meta.axisSide,
      space: 14,
      child: Text(text, style: style),
    );
  }

  FlTitlesData get titlesData => FlTitlesData(
        show: true,
        bottomTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 30,
            getTitlesWidget: getTitles,
          ),
        ),
        topTitles: const AxisTitles(
          sideTitles: SideTitles(showTitles: false),
        ),
        rightTitles: const AxisTitles(
          sideTitles: SideTitles(showTitles: false),
        ),
      );

  FlBorderData get borderData => FlBorderData(
        show: false,
      );

  List<BarChartGroupData> get barGroups => [
        BarChartGroupData(
          x: 0,
          barRods: [
            BarChartRodData(
              toY: timeTakenCPU,
              color: Colors.deepOrange,
              width: 17,
            )
          ],
          showingTooltipIndicators: [1],
        ),
        BarChartGroupData(
          x: 1,
          barRods: [
            BarChartRodData(toY: timeTakenGPU, color: Colors.green, width: 17)
          ],
          showingTooltipIndicators: [2],
        ),
      ];
}

// class InfoBarChart extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return const Padding(
//       padding: EdgeInsets.all(30.0),
//       child: _BarChart(),
//     );
//   }
// }

// class InfoBarChart extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return const Padding(
//       padding: EdgeInsets.all(30.0),
//       child: _BarChart(),
//     );
//   }
// }