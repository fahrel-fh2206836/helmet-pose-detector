// Reusable custom widget for having Icon and text within the same row.
import 'package:flutter/material.dart';

class IconWithText extends StatelessWidget {
  final IconData iconData;
  final String text;
  final Color textColor;
  final Color iconColor;

  const IconWithText({
    super.key,
    required this.iconData,
    required this.text,
    this.iconColor = Colors.green,
    this.textColor = Colors.black,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(iconData, color: iconColor, size: 35),
        const SizedBox(width: 8),
        Text(text, style: TextStyle(fontSize: 16, color: textColor)),
      ],
    );
  }
}
