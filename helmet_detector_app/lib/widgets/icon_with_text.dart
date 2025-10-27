// Reusable custom widget for having Icon and text within the same row.
import 'package:flutter/material.dart';

class IconWithText extends StatelessWidget {
  final IconData iconData;
  final String text;

  const IconWithText({super.key, required this.iconData, required this.text});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(iconData, color: Colors.green, size: 35),
        const SizedBox(width: 8),
        Text(text, style: const TextStyle(fontSize: 16)),
      ],
    );
  }
}
