import 'package:byte_force/pages/login_page.dart';
import 'package:flutter/material.dart';
import 'package:get/get.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      title: 'Vehicle Monitoring',
      theme: ThemeData(
        primaryColor: const Color(0xFFFFFAF3),
        scaffoldBackgroundColor: const Color(0xFFFFFAF3),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFFFFAF3),
          iconTheme: IconThemeData(color: Color(0xFF111111)),
          titleTextStyle: TextStyle(
            color: Color(0xFF111111),
            fontSize: 20,
          ),
        ), // Set the default font for the entire app
      ),
      home: const LoginPage(), // Set the LoginPage as the initial route
      debugShowCheckedModeBanner: false,
    );
  }
}
