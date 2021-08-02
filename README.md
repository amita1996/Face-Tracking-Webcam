# Face-Tracking-Webcam
The project was made using a robotic arm, a webcam and a servo (MG996R) connected to a Raspberry Pi.

# How it works:
Using OpenCV Haar cascades the camera detects faces in the current frame. For each face it checks if its my face using "face recognition" library. Once my face is detected the program will now perform Object Tracking. If my face is near the image boundaries the camera will move (using the servo and the raspberry pi) to the location my face is centered.


Sadly one of the servo's wouldnt stop jittering (tried to fix it for a couple of days) so the camera only moves up and down and will not move left and right.



Video to demonstrate how it works:



[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/5j7RGh3648M/0.jpg)](https://www.youtube.com/watch?v=5j7RGh3648M)

