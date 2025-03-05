import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class ProfilePage extends StatelessWidget {
  const ProfilePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile', style: TextStyle(color: Color(0xFF111111))),
      ),
      body: Drawer(
        child: Column(
          children: [
            const UserAccountsDrawerHeader(
              accountName: Text('Hi, Amarnath'),
              accountEmail: Text('amarnath@example.com'),
              currentAccountPicture: CircleAvatar(
                backgroundImage: AssetImage('assets/images/profile.png'),
              ),
              decoration: BoxDecoration(
                color: Color(0xFFFFFAF3),
              ),
            ),
            ListTile(
              leading: const Icon(Icons.home, color: Color(0xFF111111)),
              title: const Text('Home', style: TextStyle(color: Color(0xFF111111))),
              onTap: () {
                // Navigate to Home
              },
            ),
            ListTile(
              leading: const Icon(Icons.settings, color: Color(0xFF111111)),
              title: const Text('Settings', style: TextStyle(color: Color(0xFF111111))),
              onTap: () {
                // Navigate to Settings
              },
            ),
            ListTile(
              leading: const Icon(FontAwesomeIcons.signOutAlt, color: Color(0xFF111111)),
              title: const Text('Logout', style: TextStyle(color: Color(0xFF111111))),
              onTap: () {
                // Handle logout
              },
            ),
          ],
        ),
      ),
    );
  }
}
