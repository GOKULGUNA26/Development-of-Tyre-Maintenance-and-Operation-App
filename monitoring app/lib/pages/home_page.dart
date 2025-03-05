import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'profile_page.dart';
import '../widgets/tire_info_widget.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String selectedVehicle = 'Tata Motors';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: DropdownButton<String>(
          value: selectedVehicle,
          dropdownColor: const Color(0xFFFFFAF3),
          items: <String>['Tata Motors', 'Mahindra'].map((String value) {
            return DropdownMenuItem<String>(
              value: value,
              child: Text(value, style: const TextStyle(color: Color(0xFF111111))),
            );
          }).toList(),
          onChanged: (value) {
            setState(() {
              selectedVehicle = value!;
            });
          },
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications),
            onPressed: () {
              // Handle notifications
            },
          )
        ],
        leading: IconButton(
          icon: const Icon(Icons.menu),
          onPressed: () {
            Get.to(const ProfilePage(), transition: Transition.leftToRight);
          },
        ),
      ),
      body: buildVehicleMonitoringScreen(),
    );
  }

  Widget buildVehicleMonitoringScreen() {
    return Column(
      children: [
        Image.asset(
          selectedVehicle == 'Tata Motors'
              ? 'assets/images/tata_motors.png'
              : 'assets/images/mahindra.png',
          height: 200,
        ),
        Expanded(
          child: GridView.count(
            crossAxisCount: 2,
            padding: const EdgeInsets.all(20.0),
            crossAxisSpacing: 10.0,
            mainAxisSpacing: 10.0,
            children: [
              buildTireInfo('32°C', 'PSI: 30', 'Front Left', false),
              buildTireInfo('31°C', 'PSI: 31', 'Front Right', false),
              buildTireInfo('80°C', 'PSI: 28', 'Rear Left', true),
              buildTireInfo('28°C', 'PSI: 30', 'Rear Right', false),
            ],
          ),
        ),
      ],
    );
  }
}
