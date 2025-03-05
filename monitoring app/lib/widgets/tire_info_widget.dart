import 'package:flutter/material.dart';

Widget buildTireInfo(String temp, String psi, String label, bool isError) {
  return Container(
    decoration: BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(10),
      boxShadow: [
        BoxShadow(
          color: Colors.grey.shade300,
          blurRadius: 10,
          offset: const Offset(0, 5),
        ),
      ],
    ),
    padding: const EdgeInsets.all(10.0),
    child: Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(temp, style: TextStyle(fontSize: 24, color: isError ? Colors.red : Colors.black)),
        const SizedBox(height: 10),
        Text(psi, style: const TextStyle(fontSize: 18)),
        const SizedBox(height: 10),
        Text(label, style: const TextStyle(fontSize: 16)),
      ],
    ),
  );
}
