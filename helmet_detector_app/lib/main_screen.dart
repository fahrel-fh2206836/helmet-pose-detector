import 'package:flutter/material.dart';
import 'package:helmet_detector_app/widgets/camera_widget.dart';
import 'package:helmet_detector_app/widgets/speed_widget.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  bool _isActivated = false; // Initial button state
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
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
            ),

            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'Stay Focused, Drive Safe',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: Colors.deepOrange,
                ),
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
              child: Center(child: CameraWidget()),
            ),

            const SizedBox(height: 24),

            // Activation Button
            SizedBox(
              width: 300,
              height: 50,
              child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isActivated
                      ? const Color(0xFF12EB66)
                      : Colors.grey,
                  foregroundColor: Colors.black,
                  // minimumSize: Size(300, 60),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 12,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                onPressed: () {
                  // add logic
                  setState(() {
                    _isActivated = !_isActivated;
                  });
                },
                icon: const Icon(Icons.power_settings_new),
                label: Expanded(
                  child: Text(
                    _isActivated
                        ? 'Activated Head Tracking'
                        : 'Activate AI & Camera',
                    style: TextStyle(fontSize: 18),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Speed indicator
            const SpeedWidget(),
          ],
        ),
      ),

      // Bottom Navbar
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.location_on), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: ''),
        ],
        onTap: (index) {
          // handle navigation
        },
      ),
    );
  }
}
