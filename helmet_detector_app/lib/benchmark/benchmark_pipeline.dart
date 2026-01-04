// Benchmarking purposes
import 'dart:io';

import 'package:path_provider/path_provider.dart';

class PipelineLatencyCsvLogger {
  final List<_Row> _rows = [];
  int receivedFrames = 0;
  int processedFrames = 0;

  bool enabled = false;

  // optional warmup: ignore first N processed frames
  final int warmupProcessedFrames;
  PipelineLatencyCsvLogger({this.warmupProcessedFrames = 0});

  void start() {
    _rows.clear();
    receivedFrames = 0;
    processedFrames = 0;
    enabled = true;
  }

  void stop() => enabled = false;

  void onFrameReceived() {
    if (!enabled) return;
    receivedFrames++;
  }

  void onFrameProcessed({required int pipelineUs}) {
    if (!enabled) return;
    processedFrames++;

    if (processedFrames <= warmupProcessedFrames) return;

    _rows.add(
      _Row(tUs: DateTime.now().microsecondsSinceEpoch, pipelineUs: pipelineUs),
    );
  }

  /// Saves CSV in app documents directory and returns the file.
  Future<File> saveCsv({String fileName = "pipeline_latency.csv"}) async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File("${dir.path}/$fileName");

    final sb = StringBuffer();
    sb.writeln("t_us,pipeline_us");

    for (final r in _rows) {
      sb.writeln("${r.tUs},${r.pipelineUs}");
    }

    await file.writeAsString(sb.toString(), flush: true);
    return file;
  }
}

class _Row {
  final int tUs;
  final int pipelineUs;
  _Row({required this.tUs, required this.pipelineUs});
}

Future<void> stopAndExport() async {
  logger.stop();

  final ts = DateTime.now().millisecondsSinceEpoch;
  final file = await logger.saveCsv(fileName: "pipeline_latency_S24+_$ts.csv");

  // print path (useful for sanity)
  // ignore: avoid_print
  print("Saved CSV to: ${file.path}");
}

final logger = PipelineLatencyCsvLogger(warmupProcessedFrames: 30);
