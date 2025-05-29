import 'package:flutter/material.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            // Headings
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16.0),
              child: Text(
                'DriveSafe',
                style: TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),

            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'Stay Focused, Drive Safe',
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: Colors.deepOrange),
              ),
            ),

            const SizedBox(height: 8),

            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'Press the button to monitor your driving attentiveness and receive alerts for prolonged phone usage.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16),
              ),
            ),

            const SizedBox(height: 16),

            // Camera container
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 16.0),
              height: 250,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.green, width: 2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Center(
                child: Text('Camera Display'),
              ),
            ),

            const SizedBox(height: 24),

            // Activation Button
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: Color(0xFF12EB66),
                foregroundColor: Colors.black,
                minimumSize: Size(250, 60),
                padding:
                    const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
              onPressed: () {
                // add logic
              },
              icon: const Icon(Icons.power_settings_new),
              label: const Text(
                'Activate AI & Camera',
                style: TextStyle(fontSize: 18),
              ),
            ),

            const SizedBox(height: 16),

            // Speed indicator
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                Icon(Icons.motorcycle, color: Colors.green),
                SizedBox(width: 8),
                Text(
                  'Current Speed: 0 KM/h', // Dummy
                  style: TextStyle(fontSize: 16),
                ),
              ],
            ),
          ],
        ),
      ),

      // Bottom Navbar
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: '',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.location_on),
            label: '',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: '',
          ),
        ],
        onTap: (index) {
          // handle navigation
        },
      ),
    );
  }
}
